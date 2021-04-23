# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

import torch
import onnxruntime
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch.utils.cpp_extension import load_inline


def _ortvalue_to_torch_tensor(ortvalue):
    # PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
    # and convert the config to torch.uint8 tensor duing from_dlpack().
    # So we need to convert the torch tensor to torch.bool type if OrtValue is bool tensor.
    torch_tensor = from_dlpack(ortvalue._ortvalue.to_dlpack())
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == 'tensor(bool)' else torch_tensor


def _ortvalue_from_torch_tensor(torch_tensor):
    return OrtValue(C.OrtValue.from_dlpack(to_dlpack(torch_tensor), torch_tensor.dtype == torch.bool))


def _load_torch_gpu_allocator_cpp_extension(verbosity, is_rocm_pytorch):
    gpu_identifier = "hip" if is_rocm_pytorch else "cuda"
    gpu_allocator_header = "HIPCachingAllocator" if is_rocm_pytorch else "CUDACachingAllocator"
    torch_gpu_allocator_addresses_cpp_source = f'''
        #include <torch/extension.h>
        #include <c10/{gpu_identifier}/{gpu_allocator_header}.h>

        size_t gpu_caching_allocator_raw_alloc_address() {{
            return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_alloc);
        }}

        size_t gpu_caching_allocator_raw_delete_address() {{
            return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_delete);
        }}
    '''

    return load_inline(name='inline_extension',
                       cpp_sources=[torch_gpu_allocator_addresses_cpp_source],
                       extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                       functions=['gpu_caching_allocator_raw_alloc_address',
                                  'gpu_caching_allocator_raw_delete_address'],
                       verbose=verbosity,
                       with_cuda=True)


def _check_same_device(device, argument_str, *args):
    '''Check that all tensor arguments in *args reside on the same device as the input device'''

    assert isinstance(device, torch.device), '`device` must be a valid `torch.device` object'
    for arg in args:
        if arg is not None and isinstance(arg, torch.Tensor):
            arg_device = torch.device(arg.device)
            if arg_device != device:
                raise RuntimeError(
                    f"{argument_str} found on device {arg_device}, but expected it to be on module device {device}.")


def get_device_from_module(module):
    '''Returns the first device found in the `module`'s parameters or None'''
    device = None
    try:
        device = next(module.parameters()).device
        for param in module.parameters():
            if param.device != device:
                raise RuntimeError('ORTModule supports a single device per model for now')
    except StopIteration:
        # Model doesn't have a device set to any of the model parameters
        pass
    return device


def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_ortvalue_input(value_info.name, _ortvalue_from_torch_tensor(inputs[idx]))

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type, device_id=_utils.get_device_index(device))


def _load_aten_functions_cpp_extension(verbosity, is_rocm_pytorch):
    aten_functions_cpp_source = """
#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <tuple>

DLManagedTensor* aten_embedding(const DLManagedTensor* weight, const DLManagedTensor* indices, int64_t padding_idx,
                                bool scale_grad_by_freq) {
  return at::toDLPack(
      at::embedding(at::fromDLPack(weight), at::fromDLPack(indices), padding_idx, scale_grad_by_freq, false));
}

DLManagedTensor* aten_embedding_backward(const DLManagedTensor* grad, const DLManagedTensor* weight,
                                         const DLManagedTensor* indices, int64_t padding_idx, bool scale_grad_by_freq) {
  return at::toDLPack(at::embedding_backward(at::fromDLPack(grad), at::fromDLPack(indices),
                                             at::fromDLPack(weight).size(0), padding_idx, scale_grad_by_freq, false));
}

std::tuple<DLManagedTensor*, DLManagedTensor*, DLManagedTensor*, DLManagedTensor*> aten_cudnn_batch_norm(
    const DLManagedTensor* input, const DLManagedTensor* weight, const DLManagedTensor* bias,
    const DLManagedTensor* running_mean, const DLManagedTensor* running_var, float momentum, float eps) {
  auto torch_result =
      at::cudnn_batch_norm(at::fromDLPack(input), at::fromDLPack(weight), at::fromDLPack(bias),
                           at::fromDLPack(running_mean), at::fromDLPack(running_var), true /*training*/, momentum, eps);
  return std::make_tuple(at::toDLPack(std::get<0>(torch_result)), at::toDLPack(std::get<1>(torch_result)),
                         at::toDLPack(std::get<2>(torch_result)), at::toDLPack(std::get<3>(torch_result)));
}

std::tuple<DLManagedTensor*, DLManagedTensor*, DLManagedTensor*> aten_cudnn_batch_norm_backward(
    const DLManagedTensor* grad_output, const DLManagedTensor* input, const DLManagedTensor* weight,
    const DLManagedTensor* running_mean, const DLManagedTensor* running_var, const DLManagedTensor* save_mean,
    const DLManagedTensor* save_var, const DLManagedTensor* reserve_space, float eps) {
  auto torch_input = at::fromDLPack(input);
  auto torch_reserve_space = reserve_space->dl_tensor.data ? at::fromDLPack(reserve_space)
                                                           : at::empty({0}, torch_input.options().dtype(c10::kByte));
  auto torch_result = at::cudnn_batch_norm_backward(torch_input, at::fromDLPack(grad_output),
                                                    at::fromDLPack(weight), at::fromDLPack(running_mean),
                                                    at::fromDLPack(running_var), at::fromDLPack(save_mean),
                                                    at::fromDLPack(save_var), eps, torch_reserve_space);
  return std::make_tuple(at::toDLPack(std::get<0>(torch_result)), at::toDLPack(std::get<1>(torch_result)),
                         at::toDLPack(std::get<2>(torch_result)));
}

size_t aten_embedding_address() { return reinterpret_cast<size_t>(&aten_embedding); }
size_t aten_embedding_backward_address() { return reinterpret_cast<size_t>(&aten_embedding_backward); }
size_t aten_cudnn_batch_norm_address() { return reinterpret_cast<size_t>(&aten_cudnn_batch_norm); }
size_t aten_cudnn_batch_norm_backward_address() { return reinterpret_cast<size_t>(&aten_cudnn_batch_norm_backward); }
    """

    aten_functions_cpp_extension = load_inline(name='inline_extension_aten_functions', cpp_sources=[aten_functions_cpp_source],
                                               extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                                               functions=['aten_embedding_address',
                                                          'aten_embedding_backward_address',
                                                          'aten_cudnn_batch_norm_address',
                                                          'aten_cudnn_batch_norm_backward_address'],
                                               verbose=verbosity, with_cuda=True)

    onnxruntime.register_external_function("aten::embedding",
                                           str(aten_functions_cpp_extension.aten_embedding_address()),
                                           str(aten_functions_cpp_extension.aten_embedding_backward_address()))
    onnxruntime.register_external_function("aten::cudnn_batch_norm",
                                           str(aten_functions_cpp_extension.aten_cudnn_batch_norm_address()),
                                           str(aten_functions_cpp_extension.aten_cudnn_batch_norm_backward_address()))

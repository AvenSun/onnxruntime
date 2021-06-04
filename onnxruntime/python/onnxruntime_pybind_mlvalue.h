// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocatormgr.h"
#include "core/session/environment.h"
#include "core/framework/ml_value.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
class SparseTensor;
namespace python {

namespace py = pybind11;

extern const char* PYTHON_ORTVALUE_OBJECT_NAME;
extern const char* PYTHON_ORTVALUE_NATIVE_OBJECT_ATTR;

bool IsNumericNumpyType(int npy_type);

bool IsNumericNumpyArray(const py::object& py_object);

bool IsNumpyArray(py::object& obj);

int GetNumpyArrayType(const py::object& obj);

bool IsNumericDType(const py::dtype& dtype);

TensorShape GetShape(const py::array& arr);

int OnnxRuntimeTensorToNumpyType(const DataTypeImpl* tensor_type);

MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type);

MLDataType NumpyToOnnxRuntimeTensorType(int numpy_type);

using MemCpyFunc = void (*)(void*, const void*, size_t);

void CpuToCpuMemCpy(void*, const void*, size_t);

void CopyDataToTensor(const py::array& py_array, int npy_type, Tensor& tensor, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

py::object AddTensorAsPyObj(const OrtValue& val, const DataTransferManager* data_transfer_manager,
                      const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

py::object GetPyObjectFromSparseTensor(size_t pos, const OrtValue& ort_value, const DataTransferManager* data_transfer_manager);

py::object AddNonTensorAsPyObj(const OrtValue& val,
                         const DataTransferManager* data_transfer_manager,
                         const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions);

OrtMemoryInfo GetMemoryInfoPerDeviceType(const OrtDevice& ort_device);

#ifdef USE_CUDA

void CpuToCudaMemCpy(void* dst, const void* src, size_t num_bytes);

void CudaToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetCudaToHostMemCpyFunction();

bool IsCudaDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetCudaAllocator(OrtDevice::DeviceId id);

std::unique_ptr<IDataTransfer> GetGPUDataTransfer();

#endif

#ifdef USE_ROCM

bool IsRocmDeviceIdValid(const onnxruntime::logging::Logger& logger, int id);

AllocatorPtr GetRocmAllocator(OrtDevice::DeviceId id);

void CpuToRocmMemCpy(void* dst, const void* src, size_t num_bytes);

void RocmToCpuMemCpy(void* dst, const void* src, size_t num_bytes);

const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* GetRocmToHostMemCpyFunction();

#endif


void CreateGenericMLValue(const onnxruntime::InputDefList* input_def_list, const AllocatorPtr& alloc,
                          const std::string& name_input, const py::object& value, OrtValue* p_mlvalue,
                          bool accept_only_numpy_array = false, bool use_numpy_data_memory = true, MemCpyFunc mem_cpy_to_device = CpuToCpuMemCpy);

void GetPyObjFromTensor(const Tensor& rtensor, py::object& obj,
                        const DataTransferManager* data_transfer_manager = nullptr,
                        const std::unordered_map<OrtDevice::DeviceType, MemCpyFunc>* mem_cpy_to_host_functions = nullptr);

template <class T>
struct DecRefFn {
  void operator()(T* pyobject) const {
    Py_XDECREF(pyobject);
  }
};

template <class T>
using UniqueDecRefPtr = std::unique_ptr<T, DecRefFn<T>>;

// This class exposes SparseTensor to Python
// The class serves two major purposes
// - to be able to map numpy arrays memory and use it on input, this serves as a reference holder
//   so incoming arrays do not disappear
// - to be able to expose SparseTensor returned from run method
class PySparseTensor {
 public:
  /// <summary>
  /// Use this constructor when you created a SparseTensor instance which is backed
  /// by python array storage and it important that they stay alive while this object is
  /// alive
  /// </summary>
  /// <param name="instance">a fully constructed and populated instance of SparseTensor</param>
  /// <param name="storage">a collection reference guards</param>
  PySparseTensor(std::unique_ptr<SparseTensor>&& instance,
                 std::vector<py::object>&& storage)
      : backing_storage_(std::move(storage)), ort_value_() {
    Init(std::move(instance));
  }

  /// <summary>
  /// Same as above but no backing storage as SparseTensor owns the memory
  /// </summary>
  /// <param name="instance"></param>
  explicit PySparseTensor(std::unique_ptr<SparseTensor>&& instance)
      : backing_storage_(), ort_value_() {
    Init(std::move(instance));
  }

  explicit PySparseTensor(const OrtValue& ort_value) 
   : backing_storage_(), ort_value_(ort_value) {}

  PySparseTensor(const PySparseTensor&) = delete;
  PySparseTensor& operator=(const PySparseTensor&) = delete;

  PySparseTensor(PySparseTensor&& o) noexcept {
    *this = std::move(o);
  }

  PySparseTensor& operator=(PySparseTensor&& o) noexcept {
    ort_value_ = std::move(o.ort_value_);
    backing_storage_ = std::move(o.backing_storage_);
    return *this;
  }

  ~PySparseTensor();

  const SparseTensor& Instance() const {
    return ort_value_.Get<SparseTensor>();
  }

  SparseTensor* Instance() {
    return ort_value_.GetMutable<SparseTensor>();
  }

  std::unique_ptr<OrtValue> AsOrtValue() const {
    return std::make_unique<OrtValue>(ort_value_);
  }

 private:

  void Init(std::unique_ptr<SparseTensor>&& instance);

  OrtValue ort_value_;
  // These will hold references to underpinning python array objects
  // when they serve as a backing storage for a feeding SparseTensor
  std::vector<py::object> backing_storage_;
};


}  // namespace python
}  // namespace onnxruntime

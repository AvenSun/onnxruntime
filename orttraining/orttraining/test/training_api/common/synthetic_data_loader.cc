// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <memory>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

#include "synthetic_data_loader.h"

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

namespace {

void RandomFloats(std::vector<float>& rets, size_t num_element) {
  const float scale = 1.f;
  const float mean = 0.f;
  const float seed = 123.f;
  static std::default_random_engine generator{static_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  std::generate_n(std::back_inserter(rets), num_element,
                  [&distribution]() -> float { return distribution(generator); });
}

template <typename IntType>
void RandomInts(std::vector<IntType>& rets, size_t num_element, IntType low, IntType high) {
  static std::random_device rd;
  static std::mt19937 generator(rd());
  std::uniform_int_distribution<IntType> distribution(low, high);

  std::generate_n(std::back_inserter(rets), num_element,
                  [&distribution]() -> IntType { return distribution(generator); });
}

}  // namespace

template <typename T>
void SyntheticSampleBatch::AddIntInput(gsl::span<const int64_t> shape, T low, T high) {
  data_vector_.push_back(SyntheticInput(shape));

  std::vector<T> values;
  auto num_of_element = data_vector_.back().NumOfElements();
  values.reserve(num_of_element);
  RandomInts(values, num_of_element, low, high);

  SyntheticDataVector& data = data_vector_.back().GetData();
  data = values;
}

void SyntheticSampleBatch::AddInt64Input(gsl::span<const int64_t> shape, int64_t low, int64_t high) {
  AddIntInput(shape, low, high);
}

void SyntheticSampleBatch::AddInt32Input(gsl::span<const int64_t> shape, int32_t low, int32_t high) {
  AddIntInput(shape, low, high);
}

void SyntheticSampleBatch::AddBoolInput(gsl::span<const int64_t> shape) {
  // Use uint8_t to store the bool value by intention, because vector<bool> is specialized, we can not create a
  // Tensor leveraging C APIs to reuse the data buffer.
  data_vector_.push_back(SyntheticInput(shape));

  std::vector<int32_t> values;
  auto num_of_element = data_vector_.back().NumOfElements();
  values.reserve(num_of_element);

  // Need random with int32_t first because MSVC compiler complains uint8_t usage for uniform_int_distribution.
  RandomInts(values, num_of_element, static_cast<int32_t>(0), static_cast<int32_t>(1));

  SyntheticDataVector& data = data_vector_.back().GetData();
  std::vector<uint8_t> uint8_values;
  std::transform(values.begin(), values.end(), std::back_inserter(uint8_values),
                 [](int32_t x) { return static_cast<uint8_t>(x); });
  data = uint8_values;
}

void SyntheticSampleBatch::AddFloatInput(gsl::span<const int64_t> shape) {
  data_vector_.push_back(SyntheticInput(shape));

  std::vector<float> values;
  auto num_of_element = data_vector_.back().NumOfElements();
  values.reserve(num_of_element);
  RandomFloats(values, num_of_element);

  SyntheticDataVector& data = data_vector_.back().GetData();
  data = values;
}

#define ORT_RETURN_ON_ERROR(expr)                              \
  do {                                                         \
    OrtStatus* onnx_status = (expr);                           \
    if (onnx_status != NULL) {                                 \
      auto code = ort_api->GetErrorCode(onnx_status);          \
      const char* msg = ort_api->GetErrorMessage(onnx_status); \
      printf("Run failed with error code :%d\n", code);        \
      printf("Error message :%s\n", msg);                      \
      ort_api->ReleaseStatus(onnx_status);                     \
      return false;                                            \
    }                                                          \
  } while (0);

bool SyntheticSampleBatch::GetBatch(std::vector<OrtValue*>& batches) {
  batches.clear();
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const auto* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  for (size_t i = 0; i < data_vector_.size(); ++i) {
    SyntheticInput& input = data_vector_[i];

    const bool ret = std::visit([&batches, &input, &ort_api, &memory_info](auto&& arg) -> bool {
      ONNXTensorElementDataType elem_data_type;
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<typename T::value_type, uint8_t>) {
        elem_data_type = Ort::TypeToTensorType<bool>::type;
      } else {
        elem_data_type = Ort::TypeToTensorType<typename T::value_type>::type;
      }

      OrtValue* value = nullptr;
      const auto& shape_vector = input.ShapeVector();
      // Be noted: the created OrtValue won't clean the raw data after its lifetime ended.
      ORT_RETURN_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          memory_info,
          arg.data(), (input.NumOfElements() * sizeof(typename T::value_type)),
          shape_vector.data(), shape_vector.size(),
          elem_data_type,
          &value));

      batches.emplace_back(value);
      return true;
    },
                                input.GetData());

    if (!ret) {
      return false;
    }
  }

  return true;
}

bool SyntheticDataLoader::GetNextSampleBatch(std::vector<OrtValue*>& batches) {
  if (sample_batch_iter_index_ >= NumOfSampleBatches()) {
    return false;
  }

  auto& sample = sample_batch_collections_[sample_batch_iter_index_];
  sample.GetBatch(batches);
  sample_batch_iter_index_ += 1;
  return true;
}

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime

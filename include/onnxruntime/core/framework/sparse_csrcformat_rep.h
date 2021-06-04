// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This class represents Compressed Storage Row(Column) sparse format
/// CSS for Row Major Matrices
/// CSC for Column Major Matrices
/// </summary>
class SparseCsrcFormatRep : public SparseRep {
 public:
  enum Order {
    kRowMajor = 1,
    kColMajor = 2
  };

  /// <summary>
  /// This constructor allocates memory for indices using the provided allocator
  /// </summary>
  /// <param name="order">matrix order represented</param>
  /// <param name="inner_shape">inner indices shape. Normally 1-D, but we would like to support batches in the future</param>
  /// <param name="outer_shape">outer indices shape. Normally 1-D, but we would like to support batches in the future</param>
  /// <param name="allocator"></param>
  SparseCsrcFormatRep(Order order, size_t inner_size,
                      size_t outer_size, const AllocatorPtr& allocator)
      : major_(order),
        inner_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(inner_size)}, allocator),
        outer_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(outer_size)}, allocator) {}

  SparseCsrcFormatRep(Order order, size_t inner_size, size_t outer_size,
                      int64_t* inner_data, int64_t* outer_data,
                      const OrtMemoryInfo& info)
      : major_(order),
        inner_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(inner_size)}, inner_data, info, 0),
        outer_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(outer_size)}, outer_data, info, 0) {}

  ~SparseCsrcFormatRep() override;

  /// <summary>
  /// Returns the matrix order currently represented
  /// </summary>
  /// <returns></returns>
  Order Major() const noexcept {
    return major_;
  }

  const Tensor& Inner() const noexcept {
    return inner_indecies_;
  }

  const Tensor& Outer() const noexcept {
    return outer_indecies_;
  }

  Tensor& MutableInner() noexcept {
    return inner_indecies_;
  }

  Tensor& MutableOuter() noexcept {
    return outer_indecies_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

 private:
  Order major_;
  // This represents indices of nnz within a row/column
  Tensor inner_indecies_;
  // This represents indices into values and inner_indecies where each row/column data starts
  Tensor outer_indecies_;
};

class SparseCsrcBuilder {
 public:
  SparseCsrcBuilder(AllocatorPtr allocator, SparseTensor& sp, std::unique_ptr<SparseRep>& rep) noexcept
      : allocator_(std::move(allocator)),
        sp_(&sp),
        rep_(&rep) {}

  ~SparseCsrcBuilder() = default;

  Status Create(SparseCsrcFormatRep::Order major,
                     size_t inner_size, size_t outer_size,
                     SparseCsrcFormatRep*& result);

  Status Create(SparseCsrcFormatRep::Order major,
                     size_t inner_size, size_t outer_size,
                     int64_t* inner_data, int64_t* outer_data);

 private:
  AllocatorPtr allocator_;
  SparseTensor* sp_;
  std::unique_ptr<SparseRep>* rep_;
};

template <>
inline const SparseCsrcFormatRep* SparseTensor::GetRep<SparseCsrcFormatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCsrc), "Expecting CSR(C) format");
  return static_cast<const SparseCsrcFormatRep*>(rep_.get());
}

template <>
inline SparseCsrcBuilder SparseTensor::RepBuilder<SparseCsrcBuilder>() {
  if (!rep_) {
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCsrc);
  } else {
    ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCsrc), "Expecting CSR(C) format set");
  }
  return SparseCsrcBuilder(allocator_, *this, rep_);
}

}  // namespace onnxruntime

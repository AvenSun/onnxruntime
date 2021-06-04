// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_csrcformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {

SparseCsrcFormatRep::~SparseCsrcFormatRep() = default;

Status SparseCsrcFormatRep::Copy(const DataTransferManager& data_transfer_manager,
                                 const AllocatorPtr& allocator,
                                 int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = std::make_unique<SparseCsrcFormatRep>(Major(),
                                                        static_cast<size_t>(inner_indecies_.Shape().Size()),
                                                        static_cast<size_t>(outer_indecies_.Shape().Size()),
                                                        allocator);
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(inner_indecies_, rep_copy->MutableInner(), exec_q_id));
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(outer_indecies_, rep_copy->MutableOuter(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCsrcFormatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
                                 int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = std::make_unique<SparseCsrcFormatRep>(Major(),
                                                        static_cast<size_t>(inner_indecies_.Shape().Size()),
                                                        static_cast<size_t>(outer_indecies_.Shape().Size()),
                                                        allocator);
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(inner_indecies_, rep_copy->MutableInner(), exec_q_id));
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(outer_indecies_, rep_copy->MutableOuter(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCsrcBuilder::Create(SparseCsrcFormatRep::Order major,
                                 size_t inner_size, size_t outer_size,
                                 SparseCsrcFormatRep*& result) {
  ORT_RETURN_IF_NOT(*rep_ == nullptr, "The instance is not empty");
  ORT_RETURN_IF_NOT(allocator_ != nullptr, "Must have an allocator set with Sparse Tensor instance");
  result = new SparseCsrcFormatRep(major, inner_size, outer_size, allocator_);
  rep_->reset(result);
  return Status::OK();
}

Status SparseCsrcBuilder::Create(SparseCsrcFormatRep::Order major, size_t inner_size, size_t outer_size,
                                 int64_t* inner_data, int64_t* outer_data) {
  ORT_RETURN_IF_NOT(*rep_ == nullptr, "The instance is not empty");
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Must have NOT an allocator set with Sparse Tensor instance");
  rep_->reset(new SparseCsrcFormatRep(major, inner_size, outer_size, inner_data, outer_data, sp_->Location()));
  return Status::OK();
}

}  // namespace onnxruntime
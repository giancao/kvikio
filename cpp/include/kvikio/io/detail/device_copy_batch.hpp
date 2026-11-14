/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file device_copy_batch.hpp
 * @brief Several memory copies handed to the CUDA driver in one call.
 *
 * N small copies issued one by one pay N driver round trips. Accumulate them
 * here and @ref kvikio::io::detail::device_copy_batch::enqueue once: the batch
 * reaches the driver as a single `cuMemcpyBatchAsync`.
 *
 * The batch is direction-agnostic and owns no stream or device state.
 *
 * Each source is read when the stream reaches its copy, not when `enqueue`
 * returns, so every source must stay alive and unmodified until then.
 *
 * `enqueue` falls back to one `cuMemcpyAsync` per copy when the driver has no
 * batch call, or when the stream is a default stream (which the batch call
 * rejects). The batch is issued with "prefer overlap with compute" when every
 * copy is 128 KiB or smaller. This is the behaviour of cuCascade's
 * `cuda::device_copy_batch`, written against the driver API.
 */

#include <kvikio/shim/cuda.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace kvikio::io::detail {

class device_copy_batch {
 public:
  /// Copies at or below this size are latency-bound small transfers that
  /// overlap well with compute; one larger copy makes the whole batch use the
  /// default flag.
  static constexpr std::size_t overlap_with_compute_max_bytes = 128ULL << 10;

  void reserve(std::size_t n)
  {
    _dsts.reserve(n);
    _srcs.reserve(n);
    _sizes.reserve(n);
  }

  /// Add a copy of @p bytes from @p src to @p dst. Null pointers and empty
  /// copies are ignored: the batch call rejects them.
  void add(void* dst, void const* src, std::size_t bytes)
  {
    if (bytes == 0 || dst == nullptr || src == nullptr) { return; }
    _dsts.push_back(reinterpret_cast<CUdeviceptr>(dst));
    _srcs.push_back(reinterpret_cast<CUdeviceptr>(src));
    _sizes.push_back(bytes);
    _bytes += bytes;
  }

  [[nodiscard]] std::size_t count() const noexcept { return _dsts.size(); }
  [[nodiscard]] std::size_t bytes() const noexcept { return _bytes; }
  [[nodiscard]] bool empty() const noexcept { return _dsts.empty(); }

  /// Whether the batch call would be used for @p stream (false: one call per copy).
  [[nodiscard]] static bool batches_on(CUstream stream)
  {
#if CUDA_VERSION >= 13000
    return !is_default_stream(stream) && static_cast<bool>(cudaAPI::instance().MemcpyBatchAsync);
#else
    (void)stream;
    return false;
#endif
  }

  /// Queue every copy on @p stream. The calling thread's current context must
  /// be able to access the device memory.
  [[nodiscard]] CUresult enqueue(CUstream stream)
  {
    if (_dsts.empty()) { return CUDA_SUCCESS; }
#if CUDA_VERSION >= 13000
    if (batches_on(stream)) {
      CUmemcpyAttributes attrs{};
      attrs.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
      attrs.flags =
        std::all_of(_sizes.begin(),
                    _sizes.end(),
                    [](std::size_t size) { return size <= overlap_with_compute_max_bytes; })
          ? CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE
          : CU_MEMCPY_FLAG_DEFAULT;
      // One attribute for the whole batch, starting at copy 0.
      std::size_t attrs_idxs[] = {0};
      return cudaAPI::instance().MemcpyBatchAsync(_dsts.data(),
                                                  _srcs.data(),
                                                  _sizes.data(),
                                                  _dsts.size(),
                                                  &attrs,
                                                  attrs_idxs,
                                                  static_cast<std::size_t>(1),
                                                  stream);
    }
#endif
    for (std::size_t i = 0; i < _dsts.size(); ++i) {
      auto const status = cudaAPI::instance().MemcpyAsync(_dsts[i], _srcs[i], _sizes[i], stream);
      if (status != CUDA_SUCCESS) { return status; }
    }
    return CUDA_SUCCESS;
  }

  void clear() noexcept
  {
    _dsts.clear();
    _srcs.clear();
    _sizes.clear();
    _bytes = 0;
  }

 private:
  [[nodiscard]] static bool is_default_stream(CUstream stream) noexcept
  {
    return stream == nullptr || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD;
  }

  std::vector<CUdeviceptr> _dsts;
  std::vector<CUdeviceptr> _srcs;
  std::vector<std::size_t> _sizes;
  std::size_t _bytes{0};
};

}  // namespace kvikio::io::detail

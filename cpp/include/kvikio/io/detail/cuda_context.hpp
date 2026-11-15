/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file cuda_context.hpp
 * @brief Making a device's CUDA context current, cheaply, on engine threads.
 *
 * The engines address a device by its ordinal and use the driver API, which
 * works on the calling thread's current context. An engine thread starts with
 * no current context.
 */

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace kvikio::io::detail {

/**
 * @brief The primary context of device @p ordinal.
 *
 * `kvikio::get_primary_cuda_context` takes a process-wide lock on every call.
 * This remembers the answer per thread, so only a thread's first request for a
 * device takes that lock.
 */
[[nodiscard]] inline CUcontext primary_context(int ordinal)
{
  constexpr std::size_t cached_devices = 16;
  thread_local std::array<CUcontext, cached_devices> cache{};
  if (ordinal < 0) { throw std::invalid_argument("primary_context: negative device ordinal"); }
  auto const index = static_cast<std::size_t>(ordinal);
  if (index >= cached_devices) { return get_primary_cuda_context(ordinal); }
  if (cache[index] == nullptr) { cache[index] = get_primary_cuda_context(ordinal); }
  return cache[index];
}

/// The ordinal of the device of the calling thread's current context, or -1 if
/// the thread has no current context.
[[nodiscard]] inline int current_device_ordinal()
{
  CUcontext ctx{};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx), std::runtime_error);
  if (ctx == nullptr) { return -1; }
  CUdevice device{};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetDevice(&device), std::runtime_error);
  int count = 0;
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGetCount(&count), std::runtime_error);
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice candidate{};
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGet(&candidate, ordinal), std::runtime_error);
    if (candidate == device) { return ordinal; }
  }
  throw std::runtime_error("current_device_ordinal: the current context's device is unknown");
}

/**
 * @brief Makes a context current for a scope.
 *
 * Nothing is switched when the context is already current. Otherwise it is
 * pushed and popped again at the end of the scope, except in one case: with
 * @c adopt_if_none, a thread that has no current context keeps the new one. An
 * engine thread uses that, so that from its second device operation on it only
 * pays the check.
 */
class scoped_context {
 public:
  enum class policy { restore, adopt_if_none };

  explicit scoped_context(CUcontext ctx, policy how = policy::restore)
  {
    CUcontext current{};
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&current), std::runtime_error);
    if (current == ctx) { return; }
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxPushCurrent(ctx), std::runtime_error);
    _pop = !(how == policy::adopt_if_none && current == nullptr);
  }

  ~scoped_context() noexcept
  {
    if (!_pop) { return; }
    CUcontext popped{};
    cudaAPI::instance().CtxPopCurrent(&popped);
  }

  scoped_context(scoped_context const&)            = delete;
  scoped_context& operator=(scoped_context const&) = delete;

 private:
  bool _pop{false};
};

}  // namespace kvikio::io::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file cuda_event.hpp
 * @brief An owned CUDA event, on the driver API.
 *
 * An engine records an event after queueing a copy and polls it with
 * @ref kvikio::io::detail::cuda_event::query, so its thread never blocks on a
 * stream. The event belongs to the context that is current when it is created;
 * record, query and destroy it with that context current.
 */

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

#include <stdexcept>
#include <utility>

namespace kvikio::io::detail {

class cuda_event {
 public:
  /// Creates the event in the current context, without timing.
  cuda_event()
  {
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&_event, CU_EVENT_DISABLE_TIMING),
                           std::runtime_error);
  }

  ~cuda_event() noexcept { reset(); }

  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;
  cuda_event(cuda_event&& other) noexcept : _event(std::exchange(other._event, nullptr)) {}
  cuda_event& operator=(cuda_event&& other) noexcept
  {
    if (this != &other) {
      reset();
      _event = std::exchange(other._event, nullptr);
    }
    return *this;
  }

  [[nodiscard]] CUevent get() const noexcept { return _event; }

  /// Mark the point @p stream has reached. Returns the driver's status.
  [[nodiscard]] CUresult record(CUstream stream) const noexcept
  {
    return cudaAPI::instance().EventRecord(_event, stream);
  }

  /// `CUDA_SUCCESS` when the recorded point has been reached,
  /// `CUDA_ERROR_NOT_READY` while it has not, another code on error. Never blocks.
  [[nodiscard]] CUresult query() const noexcept { return cudaAPI::instance().EventQuery(_event); }

  /// Block until the recorded point has been reached. For failure and shutdown
  /// paths only.
  [[nodiscard]] CUresult synchronize() const noexcept
  {
    return cudaAPI::instance().EventSynchronize(_event);
  }

 private:
  void reset() noexcept
  {
    if (_event != nullptr) {
      cudaAPI::instance().EventDestroy(_event);
      _event = nullptr;
    }
  }

  CUevent _event{nullptr};
};

}  // namespace kvikio::io::detail

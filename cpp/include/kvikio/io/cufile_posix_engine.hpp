/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file cufile_posix_engine.hpp
 * @brief The built-in engine for local files: cuFile where it is available, POSIX otherwise.
 *
 * It is the default engine of @c engine_route::local, and what a selection
 * falls back to when another engine for local files cannot be used. It opens
 * files with @c FileHandle, so the cuFile registration and the compatibility
 * mode decision are KvikIO's usual ones.
 *
 * Unlike @c FileHandle::pread it never blocks the caller, does not use the
 * process-wide thread pool, bounce buffer pool or @c kvikio::defaults settings,
 * and reads fragmented buffers one fragment per operation.
 *
 * It has no priorities: high and low requests share one queue.
 */

#include <kvikio/compat_mode.hpp>
#include <kvikio/io/engine.hpp>

#include <cstddef>
#include <memory>

namespace kvikio::io {

/// Name the engine is registered under.
inline constexpr char const* cufile_posix_engine_name = "cufile_posix";

/// Options of the cufile_posix engine.
struct cufile_posix_options : engine_options {
  /// Worker threads of this engine instance. They do the blocking reads and
  /// writes, so this is also the number of operations in flight.
  unsigned int num_threads{8};
  /// AUTO uses cuFile where the system supports it and POSIX elsewhere.
  CompatMode compat_mode{CompatMode::AUTO};
  /// Largest single read or write of a contiguous or device range, in bytes.
  std::size_t max_operation_bytes{16UL << 20};
};

/// Create the engine. A plain @c engine_options gets the defaults above.
[[nodiscard]] std::shared_ptr<engine> make_cufile_posix_engine(engine_options const& options);

}  // namespace kvikio::io

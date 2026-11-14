/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file staging_source.hpp
 * @brief Where an engine gets the host memory it stages bytes in.
 *
 * An engine never allocates page-locked memory itself. The layer above injects
 * a @c staging_source, so the memory an engine stages through is accounted in
 * that layer's budget.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace kvikio::io {

/**
 * @brief A group of staging blocks on loan to an engine.
 *
 * Destroying the lease returns the blocks to their source.
 */
class staging_lease {
 public:
  staging_lease()                                = default;
  staging_lease(staging_lease const&)            = delete;
  staging_lease& operator=(staging_lease const&) = delete;
  virtual ~staging_lease();

  /// The blocks; each is @ref block_size bytes.
  [[nodiscard]] virtual std::span<std::byte* const> blocks() const noexcept = 0;

  /// Size of one block in bytes.
  [[nodiscard]] virtual std::size_t block_size() const noexcept = 0;

  /// Total bytes on loan.
  [[nodiscard]] std::size_t size_bytes() const noexcept { return blocks().size() * block_size(); }
};

/// Outcome of @c staging_source::acquire.
enum class staging_status : std::uint8_t {
  ok,           ///< the lease covers the request
  unavailable,  ///< no blocks right now; the engine parks the operation and retries later
  failed,       ///< the source cannot serve the request at all
};

/// Result of @c staging_source::acquire. @c lease is non-null exactly when
/// @c status is @c staging_status::ok.
struct staging_result {
  staging_status status{staging_status::failed};
  std::unique_ptr<staging_lease> lease;
};

/**
 * @brief A source of fixed-size, page-locked host blocks.
 *
 * Implementations are thread-safe. An engine calls @ref acquire without holding
 * any of its own locks.
 */
class staging_source {
 public:
  staging_source()                                 = default;
  staging_source(staging_source const&)            = delete;
  staging_source& operator=(staging_source const&) = delete;
  virtual ~staging_source();

  /// Size of one block in bytes.
  [[nodiscard]] virtual std::size_t block_size() const noexcept = 0;

  /**
   * @brief Lend blocks covering at least @p bytes.
   *
   * Never blocks and never throws: a source that has no blocks at the moment
   * answers @c staging_status::unavailable, so the calling engine thread keeps
   * serving its other operations.
   */
  [[nodiscard]] virtual staging_result acquire(std::size_t bytes) noexcept = 0;
};

}  // namespace kvikio::io

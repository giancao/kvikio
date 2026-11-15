/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file uring_http_api.hpp
 * @brief Name, options and extra interface of the `uring_http` engine.
 *
 * All a host needs to select the engine by name and to use what is beyond the
 * engine interface. Nothing here needs the engine plugin at link time.
 */

#include <kvikio/io/engine.hpp>
#include <kvikio/io/nic_topology.hpp>
#include <kvikio/io/object_store_config.hpp>
#include <kvikio/io/rest/authorizer.hpp>
#include <kvikio/io/rest/config.hpp>
#include <kvikio/io/rest/rest_reactor.hpp>
#include <kvikio/io/rest/s3/list_parser.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace kvikio::io::rest {

/// Name the engine is registered under.
inline constexpr char const* uring_http_engine_name = "uring_http";

/**
 * @brief Options of the `uring_http` engine.
 *
 * `staging` (inherited) serves device reads, which are staged through pinned
 * host blocks; without it they fail.
 */
struct uring_http_options : engine_options {
  /// Number of reactors; 0 sizes the pool from @ref nic.
  std::size_t n_reactors{0};
  /// Per-reactor configuration.
  config cfg{};
  /// Signs every request. When null, the engine makes a SigV4 authorizer from
  /// @ref object_store, whose endpoint, region and static credentials must
  /// then be set.
  std::shared_ptr<request_authorizer> authorizer;
  /// The object store to sign for when @ref authorizer is null.
  object_store_config object_store{};
  /// The egress interface, as the host discovered it. Without one the pool has
  /// the default size and its reactors are not pinned.
  std::optional<nic_topology> nic;
};

/**
 * @brief What the `uring_http` engine offers beyond the engine interface.
 *
 * Reach it with `dynamic_cast<uring_http_api*>(engine)`.
 */
class uring_http_api {
 public:
  virtual ~uring_http_api() = default;

  /// Counters of the whole pool: the reactors' @c transport_stats summed. A
  /// snapshot; safe while the pool runs.
  [[nodiscard]] virtual transport_stats stats() const noexcept = 0;

  /// Stream a bucket's ListObjectsV2 pages under @p prefix to @p sink, one call
  /// per page, until it returns false or the listing ends.
  virtual void list_objects_paged(std::string_view bucket,
                                  std::string_view prefix,
                                  std::size_t page_size,
                                  std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                                  std::optional<std::size_t> max_scanned = std::nullopt) = 0;

  /// The entries of a bucket under @p prefix.
  [[nodiscard]] virtual std::vector<s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt) = 0;

  /// The configured cap on the matches of one listing.
  [[nodiscard]] virtual std::size_t list_max_matches() const = 0;

  /// Resolve the size and footer bytes of a batch of objects, reporting each
  /// through @p on_result. Blocks until the batch is done or @p stop is set.
  virtual void resolve_footer_objects(std::span<std::string const> paths,
                                      std::function<void(footer_resolve_result)> const& on_result,
                                      std::stop_token stop = {}) = 0;

  /// How many reactors one read is spread over: the configured fan-out, clamped
  /// to the size of the pool.
  [[nodiscard]] virtual std::size_t fan_out() const noexcept = 0;

  /// Bytes of footer stash currently reserved, and their peak.
  [[nodiscard]] virtual std::size_t footer_stash_reserved_bytes() const noexcept      = 0;
  [[nodiscard]] virtual std::size_t footer_stash_reserved_peak_bytes() const noexcept = 0;
};

}  // namespace kvikio::io::rest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file object_listing.hpp
 * @brief Listing the objects of a bucket, for engines that can.
 *
 * Not part of the engine interface: an engine that lists implements this, and a
 * host reaches it with `dynamic_cast<object_listing*>(engine)`. Header-only, so
 * it needs no engine plugin at link time.
 */

#include <kvikio/io/rest/s3/list_parser.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace kvikio::io {

class object_listing {
 public:
  virtual ~object_listing() = default;

  /// Stream a bucket's ListObjectsV2 pages under @p prefix to @p sink, one call
  /// per page, until it returns false or the listing ends.
  virtual void list_objects_paged(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size,
    std::function<bool(rest::s3::list_objects_v2_page const&)> const& sink,
    std::optional<std::size_t> max_scanned = std::nullopt) = 0;

  /// The entries of a bucket under @p prefix.
  [[nodiscard]] virtual std::vector<rest::s3::list_entry> list_objects(
    std::string_view bucket,
    std::string_view prefix,
    std::size_t page_size               = 1000,
    std::optional<std::size_t> max_keys = std::nullopt) = 0;

  /// The configured cap on the matches of one listing.
  [[nodiscard]] virtual std::size_t list_max_matches() const = 0;
};

}  // namespace kvikio::io

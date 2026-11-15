/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file cufile_options.hpp
 * @brief Name and options of the `cufile_reactor` engine.
 *
 * All a host needs to select the engine by name; it pulls in neither cufile.h
 * nor the reactor classes.
 */

#include <kvikio/io/cufile/config.hpp>
#include <kvikio/io/engine.hpp>

#include <cstddef>

namespace kvikio::io::cufile {

/**
 * @brief What a host can ask about an object the `cufile_reactor` engine opened.
 *
 * Reach it with `dynamic_cast<cufile_object_view const*>(&object)`; that needs
 * this header but not the plugin at link time.
 */
class cufile_object_view {
 public:
  virtual ~cufile_object_view() = default;

  /// The descriptor opened without O_DIRECT; always valid.
  [[nodiscard]] virtual int buffered_fd() const noexcept = 0;
  /// The O_DIRECT descriptor, or -1 when the file has none.
  [[nodiscard]] virtual int direct_fd() const noexcept = 0;
  /// Whether the file is registered with cuFile.
  [[nodiscard]] virtual bool cufile_registered() const noexcept = 0;
};

/// Name the engine is registered under.
inline constexpr char const* cufile_reactor_engine_name = "cufile_reactor";

/// Options of the `cufile_reactor` engine. It stages through pinned memory of
/// its own, so `staging` (inherited) is not used.
struct cufile_reactor_options : engine_options {
  /// Number of reactors. Each serves one synchronous transfer at a time.
  std::size_t n_reactors{4};
  /// Per-reactor configuration.
  config cfg{};
};

}  // namespace kvikio::io::cufile

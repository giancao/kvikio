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

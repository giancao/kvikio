/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string>
#include <system_error>

namespace kvikio::io {

/**
 * @brief The engine exists, but this operation of it is not implemented.
 *
 * Reported by every entry point an engine declares unsupported in its
 * @c engine_capabilities (for example the write and flush entry points of a
 * read-only engine). A @c std::system_error with @c std::errc::not_supported,
 * so callers that catch @c std::system_error and compare its code keep working;
 * the message names the engine and operation and says "not yet implemented".
 */
class not_implemented_error : public std::system_error {
 public:
  /// @p what names the engine and the operation, e.g. "curl: host_write".
  explicit not_implemented_error(std::string const& what)
    : std::system_error(std::make_error_code(std::errc::not_supported),
                        what + ": not yet implemented")
  {
  }
};

/**
 * @brief Selecting an engine failed: the name is unknown, its plugin could not
 *        be loaded, or the engine could not be created or started.
 *
 * The selection that was in effect before the failed call stays in effect.
 */
class engine_selection_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace kvikio::io

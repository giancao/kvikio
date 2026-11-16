/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

/**
 * @file kernel_version.hpp
 * @brief The running kernel's major.minor (cached `uname`, test override), which decides the
 * TLS version policy and whether a TLS 1.3 KeyUpdate can re-key kTLS (Linux 6.14).
 * Implemented in ktls.cpp.
 */
namespace kvikio::io::tls::ktls {

/// Kernel major.minor as reported by uname(2).
struct kernel_release {
  unsigned major{0};
  unsigned minor{0};

  [[nodiscard]] constexpr bool at_least(unsigned maj, unsigned min) const noexcept
  {
    return major > maj || (major == maj && minor >= min);
  }
  friend constexpr bool operator==(kernel_release, kernel_release) noexcept = default;
};

/// Parse "7.0.0-38-generic" style release strings; {0, 0} when unparsable.
[[nodiscard]] kernel_release parse_kernel_release(std::string_view release) noexcept;

/// @return the running kernel (uname, cached), or the test override when one is set.
[[nodiscard]] kernel_release running_kernel() noexcept;

}  // namespace kvikio::io::tls::ktls

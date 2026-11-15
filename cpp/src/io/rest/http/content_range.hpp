/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file content_range.hpp
/// @brief `Content-Range` value parser.
/// Forms (RFC 9110 §14.4): `bytes first-last/complete`, `bytes first-last/*`, `bytes */complete`.
/// S3 sends `bytes a-b/total` on 206 and `bytes */total` on 416.

#pragma once

#include "io/rest/http/message.hpp"

#include <string_view>
#include <system_error>

namespace kvikio::io::http {

/// Parse a `Content-Range` field value (surrounding whitespace tolerated, unit case-insensitive).
/// @param v the field value.
/// @param out [out] written only on success; `complete == unknown_length` for `/*`;
///        `unsatisfied` (with `first == last == 0`) for `*/complete`.
/// @return `errc::http_bad_message` on syntax errors, overflow, `first > last` or
///         `last >= complete`; success otherwise.
[[nodiscard]] std::error_code parse_content_range(std::string_view v, content_range& out) noexcept;

}  // namespace kvikio::io::http

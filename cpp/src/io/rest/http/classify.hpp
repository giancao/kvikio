/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file classify.hpp
/// @brief S3 error-body helpers: the drain cap for error bodies and the `<Code>` scan.
///
/// Retry, re-sign and connection-reuse policy is not decided here: the transport and the REST
/// reactor own it.

#pragma once

#include "io/rest/http/message.hpp"

#include <cstddef>
#include <string_view>

namespace kvikio::io::http {

/// Error bodies up to this size are drained into the header buffer to keep the connection warm;
/// larger ones close it.
inline constexpr std::size_t max_error_body = std::size_t{64} * 1024;

/// Tiny XML scan of an S3 error body: copies the text of the first `<Code>...</Code>` element
/// (whitespace-trimmed) found within the first `max_error_body` bytes into `meta.amz_error_code`.
/// @return true when a code was found.
bool extract_error_code(std::string_view body, response_meta& meta) noexcept;

}  // namespace kvikio::io::http

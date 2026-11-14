/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace kvikio::io::rest::s3 {

/// Percent-encode @p s the way AWS Signature Version 4 canonical requests
/// require: every byte except `A-Z a-z 0-9 - _ . ~` becomes `%XX` (upper-case
/// hex); `/` is kept when @p encode_slash is false.
[[nodiscard]] std::string uri_encode(std::string_view s, bool encode_slash);

}  // namespace kvikio::io::rest::s3

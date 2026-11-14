/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <string_view>

#include <kvikio/io/rest/s3/uri_encode.hpp>

namespace kvikio::io::rest::s3 {

std::string uri_encode(std::string_view s, bool encode_slash)
{
  static constexpr char kHexUpper[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(s.size());
  for (char raw : s) {
    auto c          = static_cast<unsigned char>(raw);
    bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                      c == '-' || c == '_' || c == '.' || c == '~';
    if (unreserved || (c == '/' && !encode_slash)) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(kHexUpper[(c >> 4) & 0xF]);
      out.push_back(kHexUpper[c & 0xF]);
    }
  }
  return out;
}

}  // namespace kvikio::io::rest::s3

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/rest/http/classify.hpp"

#include "io/rest/http/rules.hpp"

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace kvikio::io::http {

bool extract_error_code(std::string_view body, response_meta& meta) noexcept
{
  body                             = body.substr(0, std::min(body.size(), max_error_body));
  constexpr std::string_view open  = "<Code>";
  constexpr std::string_view close = "</Code>";
  std::size_t const a              = body.find(open);
  if (a == std::string_view::npos) return false;
  std::size_t const from = a + open.size();
  std::size_t const b    = body.find(close, from);
  if (b == std::string_view::npos) return false;
  std::string_view code = body.substr(from, b - from);
  while (!code.empty() &&
         (detail::is_ws(code.front()) || code.front() == '\r' || code.front() == '\n')) {
    code.remove_prefix(1);
  }
  while (!code.empty() &&
         (detail::is_ws(code.back()) || code.back() == '\r' || code.back() == '\n')) {
    code.remove_suffix(1);
  }
  if (code.empty()) return false;
  if (!meta.amz_error_code.assign(code)) meta.truncated_fields = true;
  return true;
}

}  // namespace kvikio::io::http

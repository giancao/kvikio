/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file codec.hpp
/// @brief Umbrella header for the HTTP/1.1 codec (no I/O).

#pragma once

#include "io/rest/http/chunked_decoder.hpp"
#include "io/rest/http/classify.hpp"
#include "io/rest/http/content_range.hpp"
#include "io/rest/http/fields.hpp"
#include "io/rest/http/iovec_sink.hpp"
#include "io/rest/http/message.hpp"
#include "io/rest/http/request_head.hpp"
#include "io/rest/http/response_parser.hpp"

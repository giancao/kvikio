/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/rest/common.hpp"
#include "io/rest/errc.hpp"

#include <cstdio>
#include <cstdlib>

/**
 * @file common.hpp
 * @brief Vocabulary of the net layer (`kvikio::io::net`): the REST transport's error codes,
 *        `result<T>`, priorities and byte units, and the debug-only invariant check of the
 *        remote path.
 */

/// Debug-only invariant check of the remote path (release builds compile it out; the pool keeps
/// counters for the same invariants so tests can assert them in any build type).
#ifndef KVIKIO_NET_ASSERT
#ifdef NDEBUG
#define KVIKIO_NET_ASSERT(cond) ((void)0)
#else
#define KVIKIO_NET_ASSERT(cond)                                                             \
  ((cond)                                                                                      \
     ? (void)0                                                                                 \
     : (std::fprintf(                                                                          \
          stderr, "cucascade net: %s:%d: assertion `%s' failed\n", __FILE__, __LINE__, #cond), \
        std::abort()))
#endif
#endif

namespace kvikio::io::net {

using rest::errc;
using rest::GiB;
using rest::KiB;
using rest::MiB;
using rest::priority;
using rest::result;

}  // namespace kvikio::io::net

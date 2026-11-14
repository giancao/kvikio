/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file concurrent_queue.hpp
 * @brief Aliases of moodycamel's lock-free queues, for the engines.
 *
 * The moodycamel headers are not part of KvikIO and `libkvikio.so` does not use
 * them: this header is included only by engine plugins and by code that embeds
 * the reactor building blocks, which supply the include path themselves.
 *
 * Some hosts ship a copy whose namespace was renamed (DuckDB's is
 * `duckdb_moodycamel`). Define `KVIKIO_MOODYCAMEL_NAMESPACE` to that name; every
 * module that exchanges these queue types must use the same one.
 */

#ifndef KVIKIO_MOODYCAMEL_NAMESPACE
#define KVIKIO_MOODYCAMEL_NAMESPACE moodycamel
#endif

#include <blockingconcurrentqueue.h>
#include <concurrentqueue.h>
#include <lightweightsemaphore.h>

namespace kvikio::io::detail {

namespace moodycamel_ns = ::KVIKIO_MOODYCAMEL_NAMESPACE;

template <typename T, typename Traits = moodycamel_ns::ConcurrentQueueDefaultTraits>
using concurrent_queue = moodycamel_ns::ConcurrentQueue<T, Traits>;

template <typename T, typename Traits = moodycamel_ns::ConcurrentQueueDefaultTraits>
using blocking_concurrent_queue = moodycamel_ns::BlockingConcurrentQueue<T, Traits>;

using lightweight_semaphore = moodycamel_ns::LightweightSemaphore;

}  // namespace kvikio::io::detail

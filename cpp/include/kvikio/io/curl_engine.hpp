/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file curl_engine.hpp
 * @brief The `curl` engine: remote objects (http://, https://, s3://) through libcurl.
 *
 * Built into `libkvikio.so` when KvikIO has remote support; the default engine
 * for object-store paths. It is the fallback, not the fast path: it has no
 * request priority, no footer probe, no connection warm-up and no writes, and
 * its capabilities say so. It lists buckets (@c object_listing).
 *
 * It reuses what `RemoteHandle` uses (endpoints, the process-wide multi-poll
 * reactors, the HTTP retry policy and the request limit, all configured through
 * `kvikio::defaults`) but not `RemoteHandle::pread` itself: a request is
 * submitted without blocking, device reads are staged through the injected
 * staging source, and a read settles when its bytes are in place, with nothing
 * left for the caller to wait on.
 */

#include <kvikio/io/engine.hpp>
#include <kvikio/io/object_store_config.hpp>

#include <cstddef>
#include <memory>

namespace kvikio::io {

/// Name the engine is registered under.
inline constexpr char const* curl_engine_name = "curl";

/**
 * @brief Options of the `curl` engine.
 *
 * `staging` (inherited) serves device reads, one staging block per request;
 * without it they fail.
 */
struct curl_options : engine_options {
  /// A host read larger than this is split into several range requests.
  std::size_t max_operation_bytes{16UL << 20};
  /**
   * @brief Where `s3://` objects live and how to sign for them.
   *
   * Each non-empty field of `endpoint`, `region`, `access_key`, `secret_key`
   * and `session_token` overrides the corresponding environment variable
   * (`AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`,
   * `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`); an empty field leaves the
   * environment variable in charge. With every field empty the engine opens
   * objects exactly as `RemoteHandle::open` does. The other fields are not used.
   */
  object_store_config object_store{};
};

/// Create the engine. A plain @c engine_options gets the defaults of @ref curl_options.
[[nodiscard]] std::shared_ptr<engine> make_curl_engine(engine_options const& options);

}  // namespace kvikio::io

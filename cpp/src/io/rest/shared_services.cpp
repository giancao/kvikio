/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/rest/shared_services.hpp"

#include "io/rest/net/uring_transport.hpp"

#include <chrono>
#include <utility>

namespace kvikio::io::rest::detail {

namespace {

net::dns_resolver::options resolver_options(config const& cfg)
{
  net::dns_resolver::options o;
  o.ttl           = std::chrono::duration_cast<std::chrono::milliseconds>(cfg.dns_ttl);
  o.max_addresses = static_cast<unsigned>(cfg.dns_max_addresses);
  o.accumulate    = cfg.dns_accumulate;
  return o;
}

}  // namespace

shared_services::shared_services(config const& cfg)
  : _tls_policy(cfg.effective_tls()),
    _dns(std::make_unique<net::dns_resolver>(resolver_options(cfg))),
    _on_ec2(net::ec2_platform())
{
}

shared_services::~shared_services() = default;

std::shared_ptr<tls::provider const> shared_services::tls()
{
  std::lock_guard lock{_mutex};
  if (!_tls) { _tls = tls::make_openssl_provider(_tls_policy); }
  return _tls;
}

}  // namespace kvikio::io::rest::detail

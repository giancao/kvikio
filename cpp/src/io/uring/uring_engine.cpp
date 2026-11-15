/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <kvikio/io/engine_plugin.hpp>
#include <kvikio/io/uring/uring_engine.hpp>

namespace kvikio::io::uring {

namespace {

/// Builds the reactors of a pool, all sharing @p context.
auto reactor_factory(std::shared_ptr<uring_reactor::reactor_context> context)
{
  return [context = std::move(context), i = 0]() mutable {
    return std::make_unique<uring_reactor>(context, "reactor-" + std::to_string(i++));
  };
}

std::shared_ptr<uring_reactor::reactor_context> make_context(
  config const& cfg, std::shared_ptr<staging_source> const& staging)
{
  if (staging == nullptr) {
    throw std::invalid_argument(
      "the uring engine needs a staging source (engine_options::staging)");
  }
  return std::make_shared<uring_reactor::reactor_context>(cfg, staging.get());
}

}  // namespace

uring_engine::uring_engine(std::size_t n_reactors,
                           config const& cfg,
                           std::shared_ptr<staging_source> staging)
  : detail::uring_engine_resources{staging, make_context(cfg, staging)},
    kvikio::io::detail::reactor_engine<uring_reactor>(
      uring_engine_name, n_reactors, reactor_factory(context))
{
}

uring_engine::uring_engine(std::size_t n_reactors,
                           std::shared_ptr<uring_reactor::reactor_context> ctx)
  : detail::uring_engine_resources{nullptr, std::move(ctx)},
    kvikio::io::detail::reactor_engine<uring_reactor>(
      uring_engine_name, n_reactors, reactor_factory(context))
{
}

uring_engine::~uring_engine() { shutdown(); }

std::vector<uring_reactor::gauges> uring_engine::reactor_gauges() noexcept
{
  std::vector<uring_reactor::gauges> out;
  try {
    out.reserve(_reactors.size());
    for (auto& reactor : _reactors) {
      out.push_back(reactor->take_gauges());
    }
  } catch (...) {
    out.clear();
  }
  return out;
}

std::shared_ptr<engine> make_uring_engine(engine_options const& options)
{
  if (auto const* typed = dynamic_cast<uring_options const*>(&options); typed != nullptr) {
    if (typed->n_reactors == 0) {
      throw std::invalid_argument("the uring engine needs at least one reactor");
    }
    return std::make_shared<uring_engine>(typed->n_reactors, typed->cfg, typed->staging);
  }
  uring_options const defaults;
  return std::make_shared<uring_engine>(defaults.n_reactors, defaults.cfg, options.staging);
}

}  // namespace kvikio::io::uring

KVIKIO_DEFINE_ENGINE_PLUGIN("uring", &kvikio::io::uring::make_uring_engine)

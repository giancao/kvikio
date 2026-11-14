/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One source, built several times: KVIKIO_TEST_PLUGIN_KIND selects which kind
// of test plugin this is. 0: a working plugin. 1: built for another engine ABI.
// 2: its header names a different engine. 3: its engine fails to start.

#include <memory>

#include <kvikio/io/engine_plugin.hpp>

#include "../utils/fake_engine.hpp"

#ifndef KVIKIO_TEST_PLUGIN_KIND
#error "KVIKIO_TEST_PLUGIN_KIND must be defined"
#endif

namespace {

std::shared_ptr<kvikio::io::engine> create(kvikio::io::engine_options const&)
{
  return std::make_shared<kvikio::test::fake_engine>(KVIKIO_TEST_PLUGIN_NAME,
                                                     KVIKIO_TEST_PLUGIN_KIND == 3);
}

}  // namespace

#if KVIKIO_TEST_PLUGIN_KIND == 1

extern "C" KVIKIO_EXPORT const kvikio::io::engine_plugin_descriptor* kvikio_engine_plugin()
{
  static const kvikio::io::engine_plugin_descriptor descriptor{
    {KVIKIO_ENGINE_PLUGIN_ABI_VERSION + 1,
     KVIKIO_ENGINE_PLUGIN_BUILD_FLAGS,
     KvikIO_VERSION,
     KVIKIO_TEST_PLUGIN_NAME},
    &create};
  return &descriptor;
}

#elif KVIKIO_TEST_PLUGIN_KIND == 2

KVIKIO_DEFINE_ENGINE_PLUGIN("some_other_engine", &create)

#else

KVIKIO_DEFINE_ENGINE_PLUGIN(KVIKIO_TEST_PLUGIN_NAME, &create)

#endif

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>

#include <kvikio/io/detail/local_descriptor.hpp>

namespace kvikio::io::detail {
namespace {

// The installed seam lives in a static slot; the pointer is what readers load,
// so the hot path costs one relaxed-acquire load when no test installed one.
reopen_hook_for_testing installed_hook{};
std::atomic<reopen_hook_for_testing const*> installed{nullptr};

}  // namespace

void set_reopen_hook_for_testing(reopen_hook_for_testing hook) noexcept
{
  installed.store(nullptr, std::memory_order_release);
  if (hook.hook == nullptr) return;
  installed_hook = hook;
  installed.store(&installed_hook, std::memory_order_release);
}

reopen_hook_for_testing const* reopen_hook_for_testing_installed() noexcept
{
  return installed.load(std::memory_order_acquire);
}

}  // namespace kvikio::io::detail

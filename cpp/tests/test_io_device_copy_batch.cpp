/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/io/detail/device_copy_batch.hpp>

#include "utils/utils.hpp"

using kvikio::io::detail::device_copy_batch;

namespace {

std::vector<std::uint8_t> to_host(void* device, std::size_t size)
{
  std::vector<std::uint8_t> host(size);
  KVIKIO_CHECK_CUDA(cudaMemcpy(host.data(), device, size, cudaMemcpyDeviceToHost));
  return host;
}

}  // namespace

TEST(DeviceCopyBatchTest, ignores_empty_and_null_copies)
{
  device_copy_batch batch;
  std::uint8_t byte{};
  batch.add(&byte, &byte, 0);
  batch.add(nullptr, &byte, 1);
  batch.add(&byte, nullptr, 1);
  EXPECT_TRUE(batch.empty());
  EXPECT_EQ(batch.enqueue(nullptr), CUDA_SUCCESS);
  batch.add(&byte, &byte, 1);
  EXPECT_EQ(batch.count(), 1u);
  EXPECT_EQ(batch.bytes(), 1u);
  batch.clear();
  EXPECT_TRUE(batch.empty());
}

TEST(DeviceCopyBatchTest, default_streams_are_never_batched)
{
  EXPECT_FALSE(device_copy_batch::batches_on(nullptr));
  EXPECT_FALSE(device_copy_batch::batches_on(CU_STREAM_LEGACY));
  EXPECT_FALSE(device_copy_batch::batches_on(CU_STREAM_PER_THREAD));
}

// Host-to-device and device-to-host, small copies (overlap flag) and one large copy (default
// flag), on an explicit stream (batched where the driver supports it) and on the default stream.
TEST(DeviceCopyBatchTest, copies_arrive_on_explicit_and_default_streams)
{
  constexpr std::size_t small = 4096;
  constexpr std::size_t large = 512 * 1024;
  std::vector<std::uint8_t> a(small), b(small), c(large);
  std::iota(a.begin(), a.end(), std::uint8_t{1});
  std::iota(b.begin(), b.end(), std::uint8_t{77});
  std::iota(c.begin(), c.end(), std::uint8_t{5});
  kvikio::test::DevBuffer<std::uint8_t> device(2 * small + large);
  auto* base = static_cast<std::uint8_t*>(device.ptr);

  cudaStream_t explicit_stream{};
  KVIKIO_CHECK_CUDA(cudaStreamCreate(&explicit_stream));

  for (CUstream stream : {static_cast<CUstream>(explicit_stream), static_cast<CUstream>(nullptr)}) {
    KVIKIO_CHECK_CUDA(cudaMemset(device.ptr, 0, device.nbytes));
    for (bool with_large : {false, true}) {
      device_copy_batch batch;
      batch.reserve(3);
      batch.add(base, a.data(), small);
      batch.add(base + small, b.data(), small);
      if (with_large) { batch.add(base + 2 * small, c.data(), large); }
      ASSERT_EQ(batch.enqueue(stream), CUDA_SUCCESS);
      KVIKIO_CHECK_CUDA(cudaStreamSynchronize(stream));
      auto const got = to_host(device.ptr, device.nbytes);
      EXPECT_TRUE(std::equal(a.begin(), a.end(), got.begin()));
      EXPECT_TRUE(std::equal(b.begin(), b.end(), got.begin() + small));
      if (with_large) { EXPECT_TRUE(std::equal(c.begin(), c.end(), got.begin() + 2 * small)); }
    }

    // Device to host through the same batch type.
    std::vector<std::uint8_t> back(small);
    device_copy_batch d2h;
    d2h.add(back.data(), base + small, small);
    ASSERT_EQ(d2h.enqueue(stream), CUDA_SUCCESS);
    KVIKIO_CHECK_CUDA(cudaStreamSynchronize(stream));
    EXPECT_EQ(back, b);
  }
  KVIKIO_CHECK_CUDA(cudaStreamDestroy(explicit_stream));
  // Say which path this machine exercised.
  RecordProperty("batched_on_explicit_stream",
                 device_copy_batch::batches_on(static_cast<CUstream>(explicit_stream)) ? 1 : 0);
}

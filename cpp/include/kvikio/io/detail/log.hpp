/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file log.hpp
 * @brief Logging for the engines.
 *
 * `KVIKIO_IO_LOG_*` take a printf-style format and arguments. Unlike the plain
 * logger call, which formats the message before looking at the level, the
 * arguments are not evaluated when the level is disabled, so a log statement
 * costs one level check where nothing is logged.
 */

#include <kvikio/logger.hpp>

#define KVIKIO_IO_LOG(level, ...)                          \
  do {                                                     \
    auto& kvikio_io_logger_ = ::kvikio::default_logger();  \
    if (kvikio_io_logger_.should_log(level)) {             \
      try {                                                \
        kvikio_io_logger_.log(level, __VA_ARGS__);         \
      } catch (...) { /* logging never fails the caller */ \
      }                                                    \
    }                                                      \
  } while (false)

#define KVIKIO_IO_LOG_DEBUG(...) KVIKIO_IO_LOG(::rapids_logger::level_enum::debug, __VA_ARGS__)
#define KVIKIO_IO_LOG_INFO(...)  KVIKIO_IO_LOG(::rapids_logger::level_enum::info, __VA_ARGS__)
#define KVIKIO_IO_LOG_WARN(...)  KVIKIO_IO_LOG(::rapids_logger::level_enum::warn, __VA_ARGS__)
#define KVIKIO_IO_LOG_ERROR(...) KVIKIO_IO_LOG(::rapids_logger::level_enum::error, __VA_ARGS__)

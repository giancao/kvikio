# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

enable_testing()

include(rapids-test)
rapids_test_init()

#[=======================================================================[.rst:
kvikio_add_test
--------------------

Create a KvikIO test.

.. code-block:: cmake

   kvikio_add_test(NAME <name> SOURCES <sources>)

   ``NAME``
   Test name. Single-value argument.

   ``SOURCES``
   List of source files for the test. Multi-value argument.
#]=======================================================================]
function(kvikio_add_test)
  cmake_parse_arguments(
    _KVIKIO # prefix
    "" # optional
    "NAME" # single value
    "SOURCES" # multi-value
    ${ARGN}
  )

  if(DEFINED _KVIKIO_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown argument: ${_KVIKIO_UNPARSED_ARGUMENTS}")
  endif()

  add_executable(${_KVIKIO_NAME} ${_KVIKIO_SOURCES})
  set_target_properties(
    ${_KVIKIO_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${KvikIO_BINARY_DIR}/gtests>"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    ${_KVIKIO_NAME}
    PRIVATE kvikio::kvikio GTest::gmock GTest::gmock_main GTest::gtest GTest::gtest_main
            CUDA::cudart_static $<TARGET_NAME_IF_EXISTS:CURL::libcurl>
  )

  rapids_test_add(
    NAME ${_KVIKIO_NAME}
    COMMAND ${_KVIKIO_NAME}
    GPUS 1
    INSTALL_COMPONENT_SET testing
  )
endfunction()

#[=======================================================================[.rst:
kvikio_add_nsys_plugin_test
---------------------------

Create a test for the support code of a KvikIO Nsight Systems plugin.

Plugin support code (under ``cpp/nsys_plugins/<plugin dir>``) deliberately does not depend on
libkvikio or CUDA, so this test links only GoogleTest and compiles the plugin sources directly,
rather than going through ``kvikio_add_test``.

.. code-block:: cmake

   kvikio_add_nsys_plugin_test(NAME <name> SOURCES <sources> PLUGIN_SOURCES <sources>)

   ``NAME``
   Test name. Single-value argument.

   ``SOURCES``
   List of test source files, relative to ``cpp/tests``. Multi-value argument.

   ``PLUGIN_SOURCES``
   List of plugin source files, relative to ``cpp/nsys_plugins`` (for example
   ``nic/nic_monitor.cpp``). The parent directory of each file is added as an include
   directory. Multi-value argument.
#]=======================================================================]
function(kvikio_add_nsys_plugin_test)
  cmake_parse_arguments(
    _KVIKIO # prefix
    "" # optional
    "NAME" # single value
    "SOURCES;PLUGIN_SOURCES" # multi-value
    ${ARGN}
  )

  if(DEFINED _KVIKIO_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown argument: ${_KVIKIO_UNPARSED_ARGUMENTS}")
  endif()

  set(_plugin_root "${KvikIO_SOURCE_DIR}/nsys_plugins")
  set(_plugin_sources "")
  set(_plugin_include_dirs "")
  foreach(_src IN LISTS _KVIKIO_PLUGIN_SOURCES)
    list(APPEND _plugin_sources "${_plugin_root}/${_src}")
    cmake_path(GET _src PARENT_PATH _src_dir)
    list(APPEND _plugin_include_dirs "$<BUILD_INTERFACE:${_plugin_root}/${_src_dir}>")
  endforeach()
  list(REMOVE_DUPLICATES _plugin_include_dirs)

  add_executable(${_KVIKIO_NAME} ${_KVIKIO_SOURCES} ${_plugin_sources})
  set_target_properties(
    ${_KVIKIO_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${KvikIO_BINARY_DIR}/gtests>"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
  )
  target_include_directories(${_KVIKIO_NAME} PRIVATE ${_plugin_include_dirs})
  target_link_libraries(
    ${_KVIKIO_NAME} PRIVATE GTest::gmock GTest::gmock_main GTest::gtest GTest::gtest_main
  )

  rapids_test_add(
    NAME ${_KVIKIO_NAME}
    COMMAND ${_KVIKIO_NAME}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

kvikio_add_test(NAME BASIC_IO_TEST SOURCES test_basic_io.cpp utils/env.cpp)

kvikio_add_test(NAME BOUNCE_BUFFER_TEST SOURCES test_bounce_buffer.cpp)

kvikio_add_test(NAME BOUNCE_BUFFER_CACHE_TEST SOURCES test_bounce_buffer_cache.cpp)

kvikio_add_test(NAME DEFAULTS_TEST SOURCES test_defaults.cpp utils/env.cpp)

kvikio_add_test(NAME DETAIL_UTILS_TEST SOURCES test_detail_utils.cpp)

kvikio_add_test(NAME ERROR_TEST SOURCES test_error.cpp)

kvikio_add_test(NAME EVENT_POOL_TEST SOURCES test_event_pool.cpp)

kvikio_add_test(NAME FILE_UTILS_TEST SOURCES test_file_utils.cpp)

kvikio_add_test(NAME IO_EVENT_BARRIER_TEST SOURCES test_io_event_barrier.cpp)

kvikio_add_test(NAME OBSERVATION_TEST SOURCES test_observation.cpp)

kvikio_add_test(NAME SUMMARY_TEST SOURCES test_summary.cpp)

kvikio_add_test(NAME STRING_UTILS_TEST SOURCES test_string_utils.cpp)

kvikio_add_test(NAME COUNTERS_TEST SOURCES test_counters.cpp)

kvikio_add_test(NAME MMAP_TEST SOURCES test_mmap.cpp)

kvikio_add_test(NAME CONCURRENT_REQUEST_LIMITER_TEST SOURCES test_concurrent_request_limiter.cpp)

kvikio_add_test(NAME EXEC_SEMI_FUTURE_TEST SOURCES test_exec_semi_future.cpp)

kvikio_add_test(NAME IO_TYPES_TEST SOURCES test_io_types.cpp)

# Engine plugins used by IO_ENGINE_REGISTRY_TEST: one source built in several kinds (see the file).
set(_kvikio_test_plugin_dir "${KvikIO_BINARY_DIR}/gtests/engine_plugins")
function(kvikio_add_test_engine_plugin name kind)
  add_library(kvikio_engine_${name} MODULE engine_plugins/test_engine_plugin.cpp)
  set_target_properties(
    kvikio_engine_${name}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_kvikio_test_plugin_dir}"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
  )
  target_compile_definitions(
    kvikio_engine_${name} PRIVATE KVIKIO_TEST_PLUGIN_NAME="${name}"
                                  KVIKIO_TEST_PLUGIN_KIND=${kind}
  )
  target_link_libraries(kvikio_engine_${name} PRIVATE kvikio::kvikio)
endfunction()
kvikio_add_test_engine_plugin(test_good 0)
kvikio_add_test_engine_plugin(test_bad_abi 1)
kvikio_add_test_engine_plugin(test_wrong_name 2)
kvikio_add_test_engine_plugin(test_start_fails 3)

kvikio_add_test(NAME IO_ENGINE_REGISTRY_TEST SOURCES test_io_engine_registry.cpp)
target_compile_definitions(
  IO_ENGINE_REGISTRY_TEST PRIVATE KVIKIO_TEST_ENGINE_PLUGIN_DIR="${_kvikio_test_plugin_dir}"
)
add_dependencies(
  IO_ENGINE_REGISTRY_TEST kvikio_engine_test_good kvikio_engine_test_bad_abi
  kvikio_engine_test_wrong_name kvikio_engine_test_start_fails
)

kvikio_add_test(NAME IO_CUFILE_POSIX_ENGINE_TEST SOURCES test_io_cufile_posix_engine.cpp)

kvikio_add_nsys_plugin_test(
  NAME NIC_MONITOR_TEST SOURCES test_nic_monitor.cpp PLUGIN_SOURCES nic/nic_monitor.cpp
)

if(KvikIO_REMOTE_SUPPORT)
  kvikio_add_test(NAME HTTP_RETRY_TEST SOURCES test_http_retry.cpp utils/env.cpp)
  kvikio_add_test(NAME REMOTE_HANDLE_TEST SOURCES test_remote_handle.cpp utils/env.cpp)
  kvikio_add_test(NAME HDFS_TEST SOURCES test_hdfs.cpp utils/hdfs_helper.cpp)
  kvikio_add_test(NAME TLS_TEST SOURCES test_tls.cpp utils/env.cpp)
  kvikio_add_test(NAME URL_TEST SOURCES test_url.cpp)
  kvikio_add_test(NAME MULTI_POLL_REACTOR_TEST SOURCES test_multi_poll_reactor.cpp utils/env.cpp)
endif()

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/libkvikio)

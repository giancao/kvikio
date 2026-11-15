# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  KvikIO
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Set logging level
set(KVIKIO_LOGGING_LEVEL
    "TRACE"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE KVIKIO_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL"
                                      "OFF"
)
message(VERBOSE "KVIKIO: KVIKIO_LOGGING_LEVEL = '${KVIKIO_LOGGING_LEVEL}'.")

# Write the version header
rapids_cmake_write_version_file(include/kvikio/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ##################################################################################################
# * build options ----------------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build KvikIO shared library" ON)
option(KvikIO_BUILD_BENCHMARKS "Configure CMake to build benchmarks" OFF)
option(KvikIO_BUILD_EXAMPLES "Configure CMake to build examples" ON)
option(KvikIO_BUILD_TESTS "Configure CMake to build tests" ON)
option(KvikIO_REMOTE_SUPPORT "Configure CMake to build with remote IO support" ON)
option(KvikIO_BUILD_NSYS_PLUGIN "Configure CMake to build the Nsight Systems NIC plugin" ON)
option(KvikIO_USE_ABSEIL_INVOCABLE
       "Back kvikio::exec::invocable with absl::AnyInvocable instead of the built-in implementation"
       OFF
)
set(KvikIO_ABSEIL_INVOCABLE_TARGET
    ""
    CACHE STRING
          "abseil any_invocable CMake target to compile against (default: absl::any_invocable)"
)
option(KvikIO_BUILD_URING_ENGINE
       "Build the io_uring engine plugin (libkvikio_engine_uring.so); needs liburing" ON
)
option(KvikIO_BUILD_URING_HTTP_ENGINE
       "Build the io_uring HTTP engine plugin for object stores; needs liburing and OpenSSL" ON
)
option(KvikIO_BUILD_CUFILE_REACTOR_ENGINE
       "Build the cuFile reactor engine plugin (libkvikio_engine_cufile_reactor.so); needs cufile.h"
       ON
)
# The reactor engines use moodycamel's lock-free queues, which libkvikio.so itself does not.
# Resolution order: an existing CMake target, a directory holding concurrentqueue.h, else
# cameron314/concurrentqueue is fetched.
set(KvikIO_MOODYCAMEL_TARGET
    ""
    CACHE STRING "CMake target that carries the moodycamel concurrentqueue headers"
)
set(KvikIO_MOODYCAMEL_INCLUDE_DIR
    ""
    CACHE PATH "Directory containing the moodycamel concurrentqueue headers"
)
set(KvikIO_MOODYCAMEL_NAMESPACE
    "moodycamel"
    CACHE STRING "C++ namespace of the moodycamel headers (DuckDB's copy uses duckdb_moodycamel)"
)

# ##################################################################################################
# * conda environment ------------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------

rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET kvikio-exports INSTALL_EXPORT_SET kvikio-exports)
create_logger_macros(KVIKIO "kvikio::default_logger()" include/kvikio)

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

if(KvikIO_REMOTE_SUPPORT)
  include(cmake/thirdparty/get_libcurl.cmake)
  if(TARGET libcurl_static)
    set_target_properties(libcurl_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endif()

# CUDA is now required
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)
include(cmake/thirdparty/get_nvtx.cmake)

if(NOT TARGET CUDA::cuFile)
  set(cuFile_FOUND 0)
  message(
    WARNING "Cannot find cuFile - KvikIO will still work but won't use GPUDirect Storage (GDS)"
  )
else()
  set(cuFile_FOUND 1)

  # Check API support
  try_compile(
    cuFile_BATCH_API_FOUND SOURCE_FROM_CONTENT
    batch.cpp
    [[#include <cufile.h>
    int main() {
      cuFileBatchIOSetUp(nullptr, 0);
      return 0;
    }
    ]]
    LINK_LIBRARIES CUDA::cuFile rt ${CMAKE_DL_LIBS}
    OUTPUT_VARIABLE batch_output
  )
  message(STATUS "Found cuFile Batch API: ${cuFile_BATCH_API_FOUND}")
  if(NOT cuFile_BATCH_API_FOUND)
    message(FATAL_ERROR "Missing cuFile Batch API")
  endif()

  try_compile(
    cuFile_STREAM_API_FOUND SOURCE_FROM_CONTENT
    stream.cpp
    [[#include <cufile.h>
    int main() {
      CUfileHandle_t fh;
      CUstream stream;
      cuFileReadAsync(fh, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
      return 0;
    }
    ]]
    LINK_LIBRARIES CUDA::cuFile rt ${CMAKE_DL_LIBS}
    OUTPUT_VARIABLE stream_output
  )
  message(STATUS "Found cuFile Stream API: ${cuFile_STREAM_API_FOUND}")
  if(NOT cuFile_STREAM_API_FOUND)
    message(FATAL_ERROR "Missing cuFile Stream API")
  endif()

  try_compile(
    cuFile_VERSION_API_FOUND SOURCE_FROM_CONTENT
    version.cpp
    [[#include <cufile.h>
    int main() {
      int version;
      cuFileGetVersion(&version);
      return 0;
    }
    ]]
    LINK_LIBRARIES CUDA::cuFile rt ${CMAKE_DL_LIBS}
    OUTPUT_VARIABLE version_output
  )
  message(STATUS "Found cuFile Version API: ${cuFile_VERSION_API_FOUND}")
endif()

include(cmake/thirdparty/get_thread_pool.cmake)

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

set(SOURCES
    "src/batch.cpp"
    "src/bounce_buffer.cpp"
    "src/buffer.cpp"
    "src/compat_mode.cpp"
    "src/compat_mode_manager.cpp"
    "src/http_status_codes.cpp"
    "src/cufile/config.cpp"
    "src/cufile/driver.cpp"
    "src/defaults.cpp"
    "src/error.cpp"
    "src/file_handle.cpp"
    "src/file_utils.cpp"
    "src/logger.cpp"
    "src/mmap.cpp"
    "src/observation.cpp"
    "src/statistics/counters.cpp"
    "src/statistics/summary.cpp"
    "src/detail/bounce_buffer_cache.cpp"
    "src/detail/concurrent_request_limiter.cpp"
    "src/detail/env.cpp"
    "src/detail/event.cpp"
    "src/detail/io_event_barrier.cpp"
    "src/detail/nvtx.cpp"
    "src/detail/observation_recorder.cpp"
    "src/detail/posix_io.cpp"
    "src/detail/stream.cpp"
    "src/detail/string_utils.cpp"
    "src/detail/utils.cpp"
    "src/exec/admission_control.cpp"
    "src/exec/thread_util.cpp"
    "src/io/rest/config.cpp"
    "src/io/rest/reactor_placement.cpp"
    "src/io/cufile_posix_engine.cpp"
    "src/io/detail/local_descriptor.cpp"
    "src/io/engine.cpp"
    "src/io/engine_registry.cpp"
    "src/io/object_identity.cpp"
    "src/io/rest/object_store_lister.cpp"
    "src/io/rest/s3/list_parser.cpp"
    "src/io/rest/s3/uri_encode.cpp"
    "src/io/uri_parser.cpp"
    "src/shim/cuda.cpp"
    "src/shim/cufile.cpp"
    "src/shim/utils.cpp"
    "src/stream.cpp"
    "src/utils.cpp"
)

if(KvikIO_REMOTE_SUPPORT)
  list(
    APPEND
    SOURCES
    "src/hdfs.cpp"
    "src/remote_handle.cpp"
    "src/detail/http_retry.cpp"
    "src/detail/multi_poll_reactor.cpp"
    "src/detail/remote_callback.cpp"
    "src/detail/tls.cpp"
    "src/detail/url.cpp"
    "src/shim/libcurl.cpp"
  )
endif()

add_library(kvikio ${SOURCES})

# To avoid symbol conflicts when statically linking to libcurl.a (see get_libcurl.cmake) and its
# dependency OpenSSL, we exclude them when building libkvikio.so. This way, libkvikio.so will not
# expose any OpenSSL symbols that could conflict with downstream users like CPython that also links
# to (another version of) OpenSSL.
target_link_options(kvikio PRIVATE "LINKER:--exclude-libs,ALL")

add_library(kvikio::kvikio ALIAS kvikio)

target_include_directories(
  kvikio
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
         "$<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
  # Private headers shared between the library and its engine plugins are included as "io/...".
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Define logging level
target_compile_definitions(
  kvikio PRIVATE "KVIKIO_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${KVIKIO_LOGGING_LEVEL}"
)

# Notice, we do not link to cuda or cufile since KvikIO opens them manually using `dlopen()`.
target_link_libraries(
  kvikio
  PUBLIC Threads::Threads BS::thread_pool ${CMAKE_DL_LIBS} $<TARGET_NAME_IF_EXISTS:nvtx3::nvtx3-cpp>
         rapids_logger::rapids_logger
  PRIVATE $<TARGET_NAME_IF_EXISTS:CURL::libcurl>
)

target_compile_definitions(
  kvikio
  PUBLIC $<$<BOOL:${KvikIO_REMOTE_SUPPORT}>:KVIKIO_LIBCURL_FOUND>
         $<$<BOOL:${cuFile_FOUND}>:KVIKIO_CUFILE_FOUND>
         $<$<BOOL:${cuFile_VERSION_API_FOUND}>:KVIKIO_CUFILE_VERSION_API_FOUND>
)

# kvikio::exec::invocable backend. The definition is public: libkvikio, its engine plugins and
# every consumer must resolve the type to the same implementation. Only the usage requirements of
# abseil are taken (COMPILE_ONLY), so libkvikio.so gains no link dependency.
if(KvikIO_USE_ABSEIL_INVOCABLE)
  if(KvikIO_ABSEIL_INVOCABLE_TARGET)
    set(_kvikio_absl_target ${KvikIO_ABSEIL_INVOCABLE_TARGET})
  elseif(TARGET absl::any_invocable)
    set(_kvikio_absl_target absl::any_invocable)
  else()
    find_package(absl CONFIG REQUIRED)
    set(_kvikio_absl_target absl::any_invocable)
  endif()
  message(
    STATUS
      "kvikio::exec::invocable backed by absl::AnyInvocable (target '${_kvikio_absl_target}')"
  )
  target_link_libraries(kvikio PUBLIC $<BUILD_INTERFACE:$<COMPILE_ONLY:${_kvikio_absl_target}>>)
  target_compile_definitions(kvikio PUBLIC KVIKIO_USE_ABSEIL_INVOCABLE)
endif()

# What the cuFile library of this process can do (kvikio::io::cufile::probe_platform). It is in
# libkvikio.so rather than in the cufile_reactor plugin so that a host can ask without loading the
# plugin. libcufile is opened with dlopen(), so this adds no link dependency.
if(cuFile_FOUND)
  target_sources(kvikio PRIVATE src/io/cufile/cufile_api.cpp src/io/cufile/platform.cpp)
endif()

set_target_properties(
  kvikio
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

# ##################################################################################################
# * engine plugins --------------------------------------------------------------------------------

# An engine plugin is a shared library next to libkvikio.so, found there by name by the engine
# registry. It can also be linked like any library by code that uses the engine's classes directly.
if(KvikIO_BUILD_URING_ENGINE
   OR KvikIO_BUILD_CUFILE_REACTOR_ENGINE
   OR KvikIO_BUILD_URING_HTTP_ENGINE
)
  add_library(kvikio_engine_thirdparty INTERFACE)
  if(KvikIO_MOODYCAMEL_TARGET)
    target_link_libraries(
      kvikio_engine_thirdparty INTERFACE $<BUILD_INTERFACE:${KvikIO_MOODYCAMEL_TARGET}>
    )
  elseif(KvikIO_MOODYCAMEL_INCLUDE_DIR)
    if(NOT EXISTS "${KvikIO_MOODYCAMEL_INCLUDE_DIR}/concurrentqueue.h")
      message(
        FATAL_ERROR
          "KvikIO_MOODYCAMEL_INCLUDE_DIR does not contain concurrentqueue.h: ${KvikIO_MOODYCAMEL_INCLUDE_DIR}"
      )
    endif()
    target_include_directories(
      kvikio_engine_thirdparty INTERFACE $<BUILD_INTERFACE:${KvikIO_MOODYCAMEL_INCLUDE_DIR}>
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      concurrentqueue
      GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
      GIT_TAG v1.0.4
    )
    FetchContent_MakeAvailable(concurrentqueue)
    target_include_directories(
      kvikio_engine_thirdparty INTERFACE $<BUILD_INTERFACE:${concurrentqueue_SOURCE_DIR}>
    )
  endif()
  target_compile_definitions(
    kvikio_engine_thirdparty
    INTERFACE KVIKIO_MOODYCAMEL_NAMESPACE=${KvikIO_MOODYCAMEL_NAMESPACE}
  )
endif()

if(KvikIO_BUILD_URING_ENGINE OR KvikIO_BUILD_URING_HTTP_ENGINE)
  find_package(PkgConfig REQUIRED)
  # GLOBAL: a project embedding KvikIO links the plugin from its own directories.
  pkg_check_modules(KVIKIO_LIBURING REQUIRED IMPORTED_TARGET GLOBAL liburing)
endif()

if(KvikIO_BUILD_URING_ENGINE)

  add_library(kvikio_engine_uring SHARED src/io/uring/uring_engine.cpp src/io/uring/uring_reactor.cpp)
  add_library(kvikio::engine_uring ALIAS kvikio_engine_uring)
  set_target_properties(
    kvikio_engine_uring
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(kvikio_engine_uring PRIVATE -Wall -Wextra -Wconversion -Wsign-conversion)
  target_compile_definitions(
    kvikio_engine_uring
    PRIVATE "KVIKIO_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${KVIKIO_LOGGING_LEVEL}"
  )
  target_link_libraries(
    kvikio_engine_uring
    PUBLIC kvikio::kvikio $<BUILD_INTERFACE:kvikio_engine_thirdparty>
           $<BUILD_INTERFACE:PkgConfig::KVIKIO_LIBURING>
  )
endif()

# The io_uring HTTP engine for object stores: the REST reactor with the layers under it (TLS, the
# HTTP codec, the network transport, S3 request signing). OpenSSL and liburing are dependencies of
# the plugin only.
if(KvikIO_BUILD_URING_HTTP_ENGINE)
  find_package(OpenSSL REQUIRED)
  add_library(
    kvikio_engine_uring_http SHARED
    src/io/rest/errc.cpp
    src/io/rest/rest_reactor.cpp
    src/io/rest/shared_services.cpp
    src/io/rest/uring_http_engine.cpp
    src/io/rest/http/chunked_decoder.cpp
    src/io/rest/http/classify.cpp
    src/io/rest/http/content_range.cpp
    src/io/rest/http/iovec_sink.cpp
    src/io/rest/http/request_head.cpp
    src/io/rest/http/response_parser.cpp
    src/io/rest/net/dns_resolver.cpp
    src/io/rest/net/tls_stream.cpp
    src/io/rest/net/uring_transport.cpp
    src/io/rest/s3/sigv4.cpp
    src/io/rest/s3/sigv4_authorizer.cpp
    src/io/tls/evp_aead.cpp
    src/io/tls/key_schedule.cpp
    src/io/tls/ktls.cpp
    src/io/tls/openssl_context.cpp
    src/io/tls/openssl_session.cpp
    src/io/tls/openssl_session_cache.cpp
    src/io/tls/openssl_trust_store.cpp
    src/io/tls/record_opener.cpp
    src/io/tls/tls.cpp
  )
  add_library(kvikio::engine_uring_http ALIAS kvikio_engine_uring_http)
  set_target_properties(
    kvikio_engine_uring_http
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
  )
  # The sources include each other's private headers as "io/...", relative to src/. Code that
  # tests those layers directly gets the same path.
  target_include_directories(
    kvikio_engine_uring_http PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  )
  target_compile_options(
    kvikio_engine_uring_http PRIVATE -Wall -Wextra -Wconversion -Wsign-conversion
  )
  target_compile_definitions(
    kvikio_engine_uring_http
    PRIVATE "KVIKIO_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${KVIKIO_LOGGING_LEVEL}"
  )
  target_link_libraries(
    kvikio_engine_uring_http
    PUBLIC kvikio::kvikio $<BUILD_INTERFACE:kvikio_engine_thirdparty>
           $<BUILD_INTERFACE:PkgConfig::KVIKIO_LIBURING> $<BUILD_INTERFACE:OpenSSL::SSL>
           $<BUILD_INTERFACE:OpenSSL::Crypto>
  )
endif()

# The cuFile reactor loads libcufile at run time: building it needs only cufile.h.
if(KvikIO_BUILD_CUFILE_REACTOR_ENGINE AND NOT cuFile_FOUND)
  message(STATUS "cufile.h not found: the cufile_reactor engine plugin is not built")
endif()
if(KvikIO_BUILD_CUFILE_REACTOR_ENGINE AND cuFile_FOUND)
  add_library(
    kvikio_engine_cufile_reactor SHARED
    src/io/cufile/cufile_engine.cpp src/io/cufile/cufile_reactor.cpp
  )
  add_library(kvikio::engine_cufile_reactor ALIAS kvikio_engine_cufile_reactor)
  set_target_properties(
    kvikio_engine_cufile_reactor
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(
    kvikio_engine_cufile_reactor PRIVATE -Wall -Wextra -Wconversion -Wsign-conversion
  )
  target_compile_definitions(
    kvikio_engine_cufile_reactor
    PRIVATE "KVIKIO_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${KVIKIO_LOGGING_LEVEL}"
  )
  target_link_libraries(
    kvikio_engine_cufile_reactor PUBLIC kvikio::kvikio $<BUILD_INTERFACE:kvikio_engine_thirdparty>
  )
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(KvikIO_BUILD_BENCHMARKS)
  # Find or install GoogleBench
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(BUILD_STATIC)

  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * add examples -----------------------------------------------------------------------------------

# optionally build examples
if(KvikIO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##################################################################################################
# * add Nsight Systems plugin ----------------------------------------------------------------------

if(KvikIO_BUILD_NSYS_PLUGIN)
  add_subdirectory(nsys_plugins/nic)
endif()

if(KvikIO_BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(cmake/thirdparty/get_gtest.cmake)

  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * install targets --------------------------------------------------------------------------------

rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME kvikio)

install(
  TARGETS kvikio
  DESTINATION ${lib_dir}
  EXPORT kvikio-exports
)

if(TARGET kvikio_engine_uring)
  install(TARGETS kvikio_engine_uring DESTINATION ${lib_dir})
endif()
if(TARGET kvikio_engine_cufile_reactor)
  install(TARGETS kvikio_engine_cufile_reactor DESTINATION ${lib_dir})
endif()
if(TARGET kvikio_engine_uring_http)
  install(TARGETS kvikio_engine_uring_http DESTINATION ${lib_dir})
endif()

install(DIRECTORY include/kvikio/ DESTINATION include/kvikio)
install(FILES ${KvikIO_BINARY_DIR}/include/kvikio/version_config.hpp DESTINATION include/kvikio)

set(doc_string
    [=[
Provide targets for KvikIO.
]=]
)

set(final_code_string
    "
set(KvikIO_CUFILE_SUPPORT [=[${cuFile_FOUND}]=])
set(KvikIO_REMOTE_SUPPORT [=[${KvikIO_REMOTE_SUPPORT}]=])
"
)
string(
  APPEND
  final_code_string
  [=[
find_package(CUDAToolkit REQUIRED QUIET)
target_include_directories(kvikio::kvikio INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

if(KvikIO_CUFILE_SUPPORT AND NOT TARGET CUDA::cuFile)
  message(FATAL_ERROR "Compiled with cuFile support but cuFile not found")
endif()
]=]
)

rapids_export(
  INSTALL kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  NAMESPACE kvikio::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK final_code_string
)

rapids_export(
  BUILD kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  NAMESPACE kvikio::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK final_code_string
)

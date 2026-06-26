cmake_minimum_required(VERSION 3.18)
project(parakeet_cpp LANGUAGES C CXX)

set(PARAKEET_VERSION "0.0.1" CACHE STRING "Version string returned by parakeet_version()")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PARAKEET_BUILD_TESTS "Build ctest targets" OFF)
option(PARAKEET_BUILD_CLI   "Build parakeet-cli" ON)
option(PARAKEET_BUILD_SERVER "Build parakeet-server (OpenAI-compatible example)" ON)
option(PARAKEET_SHARED      "Build libparakeet as shared" OFF)
option(PARAKEET_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(PARAKEET_GGML_METAL  "Forward GGML_METAL" OFF)
option(PARAKEET_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(PARAKEET_GGML_HIP "Forward GGML_HIP (ROCm)" OFF)

set(GGML_CUDA    ${PARAKEET_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL   ${PARAKEET_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${PARAKEET_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIP ${PARAKEET_GGML_HIP} CACHE BOOL "" FORCE)

# CUDA graphs: ggml defaults this OFF, but on CUDA builds it captures/replays the
# compute graph for a small free speedup (~1% measured on GB10, never negative).
# Turn it on whenever we forward CUDA so every CUDA build (docker/release/local)
# gets it. The runtime kill-switch GGML_CUDA_DISABLE_GRAPHS=1 still works.
if(PARAKEET_GGML_CUDA)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()

# Performance: -march=native (GGML_NATIVE) and tinyBLAS SGEMM (GGML_LLAMAFILE)
# give meaningful free speedups (~30% and ~25% per the rt-detr.cpp benchmarks).
# Force them on unless the caller explicitly overrides.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

# Apply any in-tree ggml patches (broadcast-fold, etc.) before configuring the
# submodule so the patched source is compiled in. Idempotent: already-applied
# patches are silently skipped.
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh")
  execute_process(
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _patch_result
    OUTPUT_VARIABLE _patch_output
    ERROR_VARIABLE  _patch_errors
  )
  if(_patch_output)
    message(STATUS "${_patch_output}")
  endif()
  if(_patch_result)
    message(WARNING "apply_ggml_patches.sh returned ${_patch_result}: ${_patch_errors}")
  endif()
endif()

add_subdirectory(third_party/ggml)

set(PARAKEET_SRC
    src/parakeet.cpp
    src/model.cpp
    src/parakeet_capi.cpp
    src/common.cpp
    src/audio_io.cpp
    src/model_loader.cpp
    src/backend.cpp
    src/ggml_graph.cpp
    src/fft.cpp
    src/mel.cpp
    src/mel_gpu.cpp
    src/subsampling.cpp
    src/relpos_attention.cpp
    src/conformer.cpp
    src/pos_enc.cpp
    src/encoder.cpp
    src/streaming_encoder.cpp
    src/streaming.cpp
    src/ctc_decoder.cpp
    src/prediction.cpp
    src/joint.cpp
    src/prompt_kernel.cpp
    src/tdt.cpp
    src/rnnt.cpp
    src/transducer_batch.cpp
    src/tokenizer.cpp
    src/search.cpp
    src/transcription.cpp)

if(PARAKEET_SHARED)
  add_library(parakeet SHARED ${PARAKEET_SRC})
  # The C-API header marks symbols with plain extern "C", no __declspec(dllexport).
  # On MSVC that yields a DLL with no exported symbols (and no import .lib), which
  # is useless for dlopen/ctypes consumers. Auto-export every symbol so the DLL is
  # usable the same way the ELF/Mach-O shared libs are. No effect on Linux/macOS.
  set_target_properties(parakeet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(parakeet STATIC ${PARAKEET_SRC})
endif()
target_include_directories(parakeet PUBLIC include PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_compile_definitions(parakeet PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_compile_definitions(parakeet PRIVATE PARAKEET_VERSION="${PARAKEET_VERSION}")
target_link_libraries(parakeet PUBLIC ggml)

if(PARAKEET_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(PARAKEET_BUILD_SERVER)
  add_subdirectory(examples/server)
endif()
if(PARAKEET_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

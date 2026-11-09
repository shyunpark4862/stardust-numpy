cmake_minimum_required(VERSION 3.27)
project(stardust_numpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.15 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC
  csrc/bind/module.cpp)

target_include_directories(_core PRIVATE csrc)

# The only target is this Mac (Apple M4 Pro): -mcpu=native resolves to apple-m4.
# -ffast-math is deliberately absent (DESIGN.md §2.4): it assumes no NaN/inf.
target_compile_options(_core PRIVATE -mcpu=native -Wall -Wextra -Wpedantic)

# Accelerate: new LAPACK interface with 64-bit integers (DESIGN.md §2.5).
target_compile_definitions(_core PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_link_libraries(_core PRIVATE "-framework Accelerate")

install(TARGETS _core LIBRARY DESTINATION stardust_numpy)

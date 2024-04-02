cmake_minimum_required(VERSION 3.27)
project(lagrange_polynomial_solver)

add_executable(lagrange_polynomial_solver src/main.cc
        src/galois.h
        src/lagrange.h)

target_compile_features(lagrange_polynomial_solver PRIVATE cxx_std_17)
set_target_properties(lagrange_polynomial_solver PROPERTIES CXX_EXTENSIONS OFF)

target_compile_options(lagrange_polynomial_solver PRIVATE
        "-fno-rtti"
        "$<$<CONFIG:DEBUG>:-O0;-g;-Wall;-Wextra;-Wpedantic>"
        "$<$<CONFIG:RELEASE>:-Ofast;-flto>")
target_compile_definitions(lagrange_polynomial_solver PRIVATE
        "$<$<CONFIG:RELEASE>:NDEBUG>")

target_include_directories(lagrange_polynomial_solver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")

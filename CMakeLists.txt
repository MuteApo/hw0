cmake_minimum_required(VERSION 3.4)
project(simple_ml_ext)

find_package(pybind11)
pybind11_add_module(simple_ml_ext src/simple_ml_ext.cpp)

target_compile_features(simple_ml_ext PUBLIC cxx_std_11)
SET_TARGET_PROPERTIES(simple_ml_ext PROPERTIES RUNTIME_OUTPUT_DIRECTORY ./src)

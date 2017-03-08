cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(vigra_fastfilters VERSION 1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vigra_cmake")

enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/vigra_cmake")
include(VigraAddDep)

vigra_add_dep(vigra_core LIVE)

add_library(vigra_fastfilters INTERFACE)
target_link_libraries(vigra_fastfilters INTERFACE vigra_core)

target_include_directories(vigra_fastfilters INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)

add_subdirectory(test)

cmake_minimum_required(VERSION 3.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(main)

find_package(MPI REQUIRED)
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP SCALAPACK_32BIT_OMP)

add_executable(main src/main.cpp)
add_executable(blacs src/blacs.cpp)

set_property(TARGET blacs PROPERTY CXX_STANDARD 17)

target_link_libraries(main PRIVATE ${MPI_LIBRARIES} mkl::scalapack_32bit_omp mkl::blas_32bit_omp)
target_link_libraries(blacs PRIVATE ${MPI_LIBRARIES} mkl::scalapack_32bit_omp mkl::blas_32bit_omp)


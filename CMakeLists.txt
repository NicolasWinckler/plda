cmake_minimum_required(VERSION 3.5)
project(plda)

set(CMAKE_CXX_STANDARD 11)
if (NOT MPI_CXX_FOUND)
    find_package(MPI)
endif ()

if (NOT MPI_CXX_FOUND)
    message(STATUS "WARNING: MPI_CXX not found... Parallel LDA version will not be built")
endif ()

set(INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include)
set(DEPENDENCIES )
set(SOURCE_FILES
        src/accumulative_model.cc
        src/cmd_flags.cc
        src/common.cc
        src/document.cc
        src/model.cc
        src/sampler.cc)

if (MPI_CXX_FOUND)
   set(INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH} ${INCLUDE_DIRECTORIES})
   set(DEPENDENCIES ${MPI_CXX_LIBRARIES} ${DEPENDENCIES})
endif ()

include_directories(${INCLUDE_DIRECTORIES})


# ############################################################
# lda : dump word-topic model only (sequential)
set(SRCS run/lda.cc)
set(EXE_NAME lda)

add_executable(${EXE_NAME} ${SRCS} ${SOURCE_FILES})

# ############################################################
# infer : infer new unseend doc (sequential)
set(SRCS run/infer.cc)
set(EXE_NAME infer)

add_executable(${EXE_NAME} ${SRCS} ${SOURCE_FILES})

# ############################################################
# mpi_lda : mpi version of lda (parallel)
if (MPI_CXX_FOUND)
    set(SRCS run/mpi_lda.cc)
    set(EXE_NAME mpi_lda)

    add_executable(${EXE_NAME} ${SRCS} ${SOURCE_FILES})
    target_link_libraries(${EXE_NAME} ${DEPENDENCIES})
endif()
# ############################################################
# lda_with_doc : dump word topic and doc-topic model (sequential)
set(SRCS run/lda_with_doc.cc)
set(EXE_NAME lda_with_doc)

add_executable(${EXE_NAME} ${SRCS} ${SOURCE_FILES})
target_link_libraries(${EXE_NAME} ${DEPENDENCIES})

# ############################################################
# mpi_lda_with_doc : mpi version of lda_with_doc (parallel)
if (MPI_CXX_FOUND)
    set(SRCS run/mpi_lda_with_doc.cc)
    set(EXE_NAME mpi_lda_with_doc)

    add_executable(${EXE_NAME} ${SRCS} ${SOURCE_FILES})
    target_link_libraries(${EXE_NAME} ${DEPENDENCIES})
endif()


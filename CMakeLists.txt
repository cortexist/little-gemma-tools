cmake_minimum_required(VERSION 3.16)
project(little-gemma-tools C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# mmcat — multimodal cat for little-gemma's -s socket. v1 shells out to the
# ffmpeg/ffprobe CLI at runtime (no link-time media dependency); it only needs
# sockets. ffmpeg must be on PATH when mmcat runs.
add_executable(mmcat src/mmcat.c)
target_include_directories(mmcat PRIVATE proto)

if(WIN32)
    target_link_libraries(mmcat PRIVATE ws2_32)   # winsock / AF_UNIX
elseif(UNIX AND NOT APPLE)
    target_link_libraries(mmcat PRIVATE m)
endif()

# The planned in-process build (links libavformat/libavcodec/libavutil/libswscale/
# libswresample, demux+decode without spawning ffmpeg) will be gated here behind a
# WITH_LIBAV option once it lands — see README.

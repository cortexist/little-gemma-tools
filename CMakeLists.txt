cmake_minimum_required(VERSION 3.16)
project(little-gemma-tools C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# mmcat — multimodal cat for little-gemma's -s socket. v1 shells out to the
# ffmpeg/ffprobe CLI at runtime (no link-time media dependency); it only needs
# sockets. ffmpeg must be on PATH when mmcat runs.
add_executable(mmcat src/mmcat.c)
target_include_directories(mmcat PRIVATE proto)

# voicecat — the live sibling: open microphone (ffmpeg capture), energy VAD,
# streaming whisper transcripts committed into the OPEN turn (LocalAgreement-2),
# barge-in over a streaming reply (the runner's LG_BARGE byte).
add_executable(voicecat src/voicecat.c)
target_include_directories(voicecat PRIVATE proto)

if(WIN32)
    target_link_libraries(mmcat PRIVATE ws2_32)   # winsock / AF_UNIX
    target_link_libraries(voicecat PRIVATE ws2_32)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(mmcat PRIVATE m)
    target_link_libraries(voicecat PRIVATE m)
endif()

# The planned in-process build (links libavformat/libavcodec/libavutil/libswscale/
# libswresample, demux+decode without spawning ffmpeg) will be gated here behind a
# WITH_LIBAV option once it lands — see README.

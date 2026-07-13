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

# clausecat — the presentation cat: raw reply stream in, one speakable clause
# per line out (thought spans and special tokens stripped). Plain stdio, no
# sockets; sits between voicecat and a line-based TTS.
add_executable(clausecat src/clausecat.c)

if(WIN32)
    target_link_libraries(mmcat PRIVATE ws2_32)   # winsock / AF_UNIX
    target_link_libraries(voicecat PRIVATE ws2_32)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(mmcat PRIVATE m)
    target_link_libraries(voicecat PRIVATE m)
endif()

# far-field-service — the audio authority: owns both directions of the sound
# card (libpulse-simple), cancels the speaker out of the mic (aec.h seam:
# webrtc AEC3 when webrtc-audio-processing-1 AND a C++ compiler are found,
# pass-through otherwise), serves clean/raw taps + one mouth over an AF_UNIX
# socket. Linux-only, and the ONLY place C++ is even probed for — the rest
# of the repo stays plain C on every platform.
if(UNIX AND NOT APPLE)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PULSE libpulse-simple)
        pkg_check_modules(WAP webrtc-audio-processing-1)
        if(PULSE_FOUND)
            set(FF_AEC "")
            if(WAP_FOUND)
                include(CheckLanguage)
                check_language(CXX)
                if(CMAKE_CXX_COMPILER)
                    enable_language(CXX)
                    set(CMAKE_CXX_STANDARD 17)
                    set(FF_AEC webrtc)
                endif()
            endif()
            if(FF_AEC STREQUAL "webrtc")
                add_executable(far-field-service src/far-field-service.c src/aec-webrtc.cpp)
                target_include_directories(far-field-service PRIVATE ${WAP_INCLUDE_DIRS})
                target_link_libraries(far-field-service PRIVATE ${WAP_LIBRARIES})
                target_link_directories(far-field-service PRIVATE ${WAP_LIBRARY_DIRS})
                message(STATUS "far-field-service: webrtc AEC3")
            else()
                add_executable(far-field-service src/far-field-service.c src/aec-null.c)
                message(STATUS "far-field-service: pass-through (no webrtc-audio-processing-1 / C++)")
            endif()
            target_include_directories(far-field-service PRIVATE proto src ${PULSE_INCLUDE_DIRS})
            target_link_libraries(far-field-service PRIVATE ${PULSE_LIBRARIES} pthread m)
        else()
            message(STATUS "libpulse-simple not found — far-field-service skipped")
        endif()
    endif()
endif()

# The planned in-process build (links libavformat/libavcodec/libavutil/libswscale/
# libswresample, demux+decode without spawning ffmpeg) will be gated here behind a
# WITH_LIBAV option once it lands — see README.

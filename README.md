# little-gemma-tools

Companion tools for [little-gemma](../little-gemma). **little-gemma's core depends
on nothing** — pure C/CUDA, no libraries, by design. Tools that *do* want a heavy
dependency (ffmpeg, etc.) live here instead, so the core stays clean. The only
coupling is the wire protocol in `proto/lg_media_proto.h`, a self-contained copy of
little-gemma's media frame format — a one-way dependency (tools → the protocol),
never the reverse.

## mmcat — multimodal cat

The successor to little-gemma's bundled `media_cat`. Where `media_cat` needs a
switch per modality (`-i` image, `-a` audio, `-f` frame), **mmcat just takes files
and figures out what each one is** — it probes with `ffprobe` and decodes with
`ffmpeg`, so it handles anything ffmpeg does (mp4, mkv, webm, jpg, png, wav, m4a…),
including an **mp4's video and soundtrack together** from one argument.

```
mmcat <socket> [flags] file... ["question"]
```

| input | becomes |
|-------|---------|
| still image | one image span, resized to the patch grid |
| video (mp4/…) | timestamped frames at `--fps` **and** its soundtrack |
| audio file | one audio span (mono 16 kHz) |
| any non-file arg | the question (the turn's text) |

Flags (modality is auto — these are policy, not switches):
- `-nv`, `--no-video` — skip frames
- `-na`, `--no-audio` — skip the soundtrack
- `--fps N` — frame sample rate (default **1**)
- `-n BUDGET` — vision tokens per frame (default 280; lower = faster prefill — for
  video you usually want a low budget, e.g. `-n 70`, since frames multiply)

### Example
```sh
# runner (in the little-gemma repo)
run-cuda-i8r -m model.gguf -mm mmproj-F16.gguf -s /tmp/lg.sock

# describe an mp4 with its soundtrack — one argument, auto video+audio
mmcat /tmp/lg.sock clip.mp4 "What happens in this video and what is said?"

# a photo
mmcat /tmp/lg.sock photo.jpg "What is this?"

# sample a long video sparsely, drop the soundtrack, small frames
mmcat /tmp/lg.sock --fps 0.5 -na -n 70 lecture.mp4 "Summarize the slides."
```

### How it works (and what's next)
mmcat sends little-gemma's typed media frames (`proto/lg_media_proto.h`): raw RGB
at patch-multiple dims for images, mono 16 kHz s16 for audio, text frames for the
`MM:SS` video timestamps — then the question as a plain line and a half-close. It
sends **raw decoded media**; the runner does the vision encoder / audio conformer
and wraps each span in the model's marker tokens.

- **v1 (this)** shells out to the `ffmpeg`/`ffprobe` CLI over pipes — no temp files,
  streaming (one frame of RAM at a time regardless of video length). Needs ffmpeg
  on `PATH`. Buildable with just a C compiler + sockets.
- **Planned:** an in-process build linking `libavformat`/`libavcodec`/`libavutil`/
  `libswscale`/`libswresample` — demux + decode + resample without spawning ffmpeg,
  interleaving video and audio by PTS in a single pass. Cleaner and faster; gated
  behind a `WITH_LIBAV` CMake option. (`pkg-config libav*` is available on Linux;
  on Windows you'd ship the ffmpeg DLLs.)

**Caveat — combined audio+video order.** mmcat emits frames-then-soundtrack for a
file with both. Whether Gemma best fuses a timestamped-frame sequence *with* an
audio span in one turn is still being pinned against measured model behavior; if a
combined turn reads incoherently, use `-na`/`-nv` to split into two turns.

## Build
```sh
cmake -S . -B build && cmake --build build --config Release
# -> build/mmcat (or build/Release/mmcat.exe on Windows)
```
Runtime: `ffmpeg` and `ffprobe` on `PATH`.

## License
MIT (see `LICENSE`), matching little-gemma.

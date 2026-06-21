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
- `--whisper-model FILE` — enable non-speech captions (or set `LG_WHISPER_MODEL`); see *Audio* below
- `--whisper-bin PATH` — the `whisper-cli` to run (default: on `PATH`; or `LG_WHISPER_BIN`)
- `-nw`, `--no-whisper` — disable captions even if a model is configured
- `--pre TEXT` — preamble text sent *before* the frames in the turn (e.g. metadata you want the
  model to see: format, fps, sample rate, file-vs-stream). mmcat just carries it — a skill/app
  builds the string. Overrides the auto video tag below.
- `--no-meta` — suppress the automatic video tag (below).

For **video**, mmcat auto-prepends a ~4-token ` downsampled video` tag before the frames, so the
model frames the input as a *downsampled video* (which it is — Gemma only ever sees sampled
frames) rather than disconnected stills. It's honest, not a trick, and cheap; suppress it with
`--no-meta` or replace it with your own `--pre`.

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

# caption non-speech audio so the model knows there's music (needs whisper.cpp)
export LG_WHISPER_MODEL=~/whisper.cpp/models/ggml-base.en.bin
export LG_WHISPER_BIN=~/whisper.cpp/build/bin/whisper-cli
mmcat /tmp/lg.sock clip.mp4 "Describe the scene and the background audio."

# hand the model input metadata up front (a skill would build this string)
mmcat /tmp/lg.sock --pre "[input] video, 2fps, 15s, from file" --fps 2 -n 66 clip.mp4 "What happens?"
```

### How it works (and what's next)
mmcat sends little-gemma's typed media frames (`proto/lg_media_proto.h`): raw RGB
at patch-multiple dims for images, mono 16 kHz s16 for audio, text frames for the
`MM:SS` video timestamps — then the question as a plain line and a half-close. It
sends **raw decoded media**; the runner does the vision encoder / audio conformer
and wraps each span in the model's marker tokens.

- **v1 (this)** shells out to the `ffmpeg`/`ffprobe` CLI over pipes — no temp files
  for media, streaming (one frame of RAM at a time regardless of video length).
  Needs ffmpeg on `PATH`; `whisper-cli` is optional (only for non-speech captions,
  and the one place a small temp wav is written). Buildable with just a C compiler
  + sockets.
- **Planned:** an in-process build linking `libavformat`/`libavcodec`/`libavutil`/
  `libswscale`/`libswresample` — demux + decode + resample without spawning ffmpeg,
  interleaving video and audio by PTS in a single pass. Cleaner and faster; gated
  behind a `WITH_LIBAV` CMake option. (`pkg-config libav*` is available on Linux;
  on Windows you'd ship the ffmpeg DLLs.)

### Audio: speech the model hears, music it doesn't

Gemma's audio tower is a **speech (ASR) encoder** — it transcribes spoken words
well, but it is *deaf to music and ambient sound*: fed those it confabulates (asked
to "describe the drums" over a solo-piano clip, it confidently invents heavy-metal
drums). So mmcat treats audio in a **hybrid** way once you give it a whisper model
(`--whisper-model` or `LG_WHISPER_MODEL`):

- **speech** → the raw audio goes to the model (its own ears), *plus* a transcript
  note for any music playing under it;
- **non-speech only** (music/ambient) → mmcat **drops the audio embedding** (it would
  only make the model hallucinate) and fuses whisper's non-speech tag onto the question
  as `Audio transcript of the provided clip: [music playing].`

Two details are load-bearing, both learned the hard way. The note is framed as a
**transcript** — the model trusts a transcript as provided ground truth and relays it;
a bare "(background audio: …)" gets dismissed by its "I'm a text AI, I can't hear"
prior. And it is **fused onto the question**, not sent as its own span — as a separate
span the model treats "can you hear?" as a capability question and denies it.

Without a whisper model mmcat sends the raw audio as before (the model still hears
speech; it just can't tell you there's music). whisper runs via `whisper-cli` (same
shell-out style as ffmpeg) and needs a small temp wav, since it can't read stdin.

**Prompting matters.** The model has a strong "I'm a text AI, I can't hear" reflex. A
yes/no *"can you hear anything?"* can make it deny audio outright — even speech it can
in fact transcribe. Ask it to **use** the audio instead — *"summarize what is said"*,
*"describe the background audio"* — and it engages. (The deepest fix is a runner `-sys`
system prompt declaring it can perceive provided audio; that's little-gemma's side.)

Combined video+audio in one turn (frames then soundtrack) is fine — the model fuses
the visuals with the speech; use `-na`/`-nv` only if you want to drop one stream.

## Build
```sh
cmake -S . -B build && cmake --build build --config Release
# -> build/mmcat (or build/Release/mmcat.exe on Windows)
```
Runtime: `ffmpeg` and `ffprobe` on `PATH`; optionally `whisper-cli` + a whisper.cpp
model for non-speech captions (point mmcat at them with `--whisper-model` /
`--whisper-bin` or `LG_WHISPER_MODEL` / `LG_WHISPER_BIN`).

## License
MIT (see `LICENSE`), matching little-gemma.

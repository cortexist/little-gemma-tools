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
| video (mp4/…) | timestamped frames at `--fps`, **and** its soundtrack as a whisper transcript (see *Video + audio together*) |
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
run-cuda-i8 -m model.gguf -mm mmproj-F16.gguf -s /tmp/lg.sock

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
sends **raw decoded media**; the runner does the vision encoding / audio projection
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

**Which Gemma 4 model can do audio at all?** There are two audio paths, and only one is
reliable:

| model | audio path | what it does |
|-------|-----------|--------------|
| **12B** | unified `gemma4ua` | **real speech ASR** — transcribes spoken words accurately; but **speech-only** (deaf to music/ambient — it confabulates on them). |
| **E4B** | legacy `gemma4a` conformer | **confabulates** — fluent output that tracks the *prompt*, not the audio; verbatim ASR fails. A model-capability limit (reproduced on llama.cpp mainline and the fork too — not a runner bug), so little-gemma's runner doesn't wire it up: E2B/E4B audio frames **error**. |
| **E2B** | legacy `gemma4a` conformer | same as E4B, weaker. |

So for in-model audio use the **12B**; for E2B/E4B the production path is Whisper for
speech-to-text. Everything below is about the 12B's tower.

The 12B's audio tower is a **speech (ASR) encoder** — it transcribes spoken words
well, but it is *deaf to music and ambient sound*: fed those it confabulates (asked
to "describe the drums" over a solo-piano clip, it confidently invents heavy-metal
drums). So mmcat treats audio in a **hybrid** way once you give it a whisper model
(`--whisper-model` or `LG_WHISPER_MODEL`):

- **speech, audio-only turn** → the raw audio goes to the model (its own ears), *plus*
  a transcript note for any music playing under it;
- **speech, video in the turn** → whisper's transcript of the words is fused onto the
  question and **no audio span is sent** — with frames in the context the model cannot
  hear (see the next section);
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

### Video + audio together: the model cannot hear once it sees

Measured on the 12B (greedy, every arrangement — soundtrack before, after or
interleaved with the frames, the model card's frames→text→audio order, a framing
tag, even *black* frames, even frames in an **earlier turn** of the session): once
image spans share the context, native audio comprehension is gone. The audio
confabulates into the visual narrative, or the model denies having ears outright
("I am a text-based AI…"). Audio-only turns hear fine. Vision survives audio;
audio does not survive vision — a capability boundary of the checkpoint, not a
template-ordering bug.

So mmcat's audio policy is **turn-scoped**: if *any* file in the turn carries
video, the soundtrack's speech travels as a whisper transcript fused onto the
question — never as an audio span. Without a whisper model configured, the
soundtrack of a video is **dropped with a loud warning**, because a poisoned span
helps nobody. Audio-only turns keep the model's own ears, unchanged. The same
boundary is why a voice+camera application should run **all** speech through
whisper (see voicecat below): one camera frame anywhere in the session silences
every voice turn after it.

### Enabling whisper (optional)

The captioning needs [whisper.cpp](https://github.com/ggerganov/whisper.cpp)'s
`whisper-cli` plus a model file — neither ships here (whisper is a dependency, kept out
like ffmpeg). One-time setup:

```sh
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j   # -> build/bin/whisper-cli
sh ./models/download-ggml-model.sh base.en                            # -> models/ggml-base.en.bin
```

Then point mmcat at both — env vars (handy in a launch script) or per-run flags:

```sh
export LG_WHISPER_BIN=~/whisper.cpp/build/bin/whisper-cli
export LG_WHISPER_MODEL=~/whisper.cpp/models/ggml-base.en.bin
mmcat /tmp/lg.sock clip.mp4 "Describe the scene and the background audio."
# or, no env: mmcat --whisper-bin … --whisper-model … /tmp/lg.sock clip.mp4 "…"
```

`base`/`base.en` is a good default. For audio-only turns mmcat uses whisper just to
tell speech from non-speech and to grab the non-speech tag — Gemma's own ears do the
transcription. For video+audio turns (and for voicecat) whisper's *words* are what the
model gets, so a larger model buys transcript quality at more latency. Use the
multilingual `base` (not `base.en`) if the audio may not be English. No model set →
audio-only turns send raw audio as before; `-nw` disables whisper for one run.

## voicecat — live voice turn-taking

mmcat is files-in-one-turn; **voicecat is an open microphone and an endless
conversation**. It segments speech with an energy VAD, and — this is the point —
**streams the transcript into the turn while you are still talking**. The runner's
turn text is appendable by design (text frames accumulate until the closing line),
so voicecat runs whisper over the growing utterance every `--commit-ms`, and words
that two consecutive passes agree on (LocalAgreement-2 — a confirmed prefix never
changes) are committed immediately. By end-of-speech only the unstable tail is
left: **the post-utterance cost is one whisper pass over the last chunk, not the
whole clip**.

```
voicecat <socket> [--mic FFMPEG_INPUT | --stdin-pcm] [flags]
```

- `--mic SPEC` — the ffmpeg capture input (default `-f alsa -i default` on Linux;
  e.g. `-f pulse -i default`, or `-f dshow -i audio=...` on Windows, where it is
  required)
- `--stdin-pcm` — mono 16 kHz s16 PCM on stdin instead of a mic; `--rt` paces it
  live (this is also the test harness — see below)
- `--whisper-model FILE` / `--whisper-bin PATH` — streaming-transcript mode
  (or `LG_WHISPER_MODEL` / `LG_WHISPER_BIN`)
- `--commit-ms N` — whisper cadence during an utterance (default 1000)
- `--hang-ms N` — trailing silence that ends an utterance (default 700)
- `--vad-level N` — RMS speech threshold, s16 units (default 400)
- `--barge-note TEXT` — see barge-in below (default `"(interrupting) "`)

With no whisper model, each utterance goes as **one native audio span** at
end-of-speech instead — the model's own ears, which per the section above work
*only while the session is vision-free*. Whisper mode is the one that composes
with a camera.

### Barge-in

The mic stays open while the reply streams. If you start talking over it,
voicecat sends the runner's one-byte `LG_BARGE` signal: the runner stops decoding
at the next token, closes the turn on the wire with `<turn|>`, and the session —
cut-off reply included, so the model knows what it didn't get to say — continues.
"I was interrupted" is remembered *here*, not in the runner: the next turn's text
opens with `--barge-note`. (Measured on the Orin: a detailed history of Rome
barged mid-sentence, then *"(interrupting) short version please"* → a one-sentence
summary, context intact.)

Turn-taking assumes the mic doesn't hear the speakers — a headset, or echo
cancellation upstream; an open-air mic will barge on its own TTS. Whisper must
run faster than real time (tiny/base on a GPU), or commits fall behind capture.

### Example, and testing without a mic

```sh
# runner (12B; voice+camera products want whisper mode)
run-cuda-i8 -m gemma-4-12b.gguf -mm mmproj-F16.gguf -s /tmp/lg.sock

# live conversation, streaming transcripts
export LG_WHISPER_BIN=~/whisper.cpp/build/bin/whisper-cli
export LG_WHISPER_MODEL=~/whisper.cpp/models/ggml-base.en.bin
voicecat /tmp/lg.sock

# no mic? feed PCM at real-time pace — the whole pipeline (VAD, commits,
# barge-in) runs exactly as live:
ffmpeg -i clip.wav -f s16le -ac 1 -ar 16000 - | voicecat /tmp/lg.sock --stdin-pcm --rt
```

A stub "whisper" (any script that prints text) plus `--whisper-bin` exercises the
streaming and barge paths deterministically with no whisper installed — that is
how voicecat was validated end-to-end.

## Build
```sh
cmake -S . -B build && cmake --build build --config Release
# -> build/mmcat, build/voicecat (build/Release/*.exe on Windows)
```
Runtime: `ffmpeg` and `ffprobe` on `PATH`; `whisper-cli` + a whisper.cpp model for
mmcat's captions/soundtrack transcripts and voicecat's streaming mode (point at
them with `--whisper-model` / `--whisper-bin` or `LG_WHISPER_MODEL` /
`LG_WHISPER_BIN`).

## License
MIT (see `LICENSE`), matching little-gemma.

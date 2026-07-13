# far-field-service — the audio authority

One process owns both directions of the sound card. That single sentence is
the whole design: echo cancellation only works where the playback reference
lives, barge-in can only cut sound that something owns, and a microphone
stream is only trustworthy when nothing else is fighting over the device.
The browser demo works precisely because Chrome is such a process; this is
the native equivalent, behind an AF_UNIX socket — no TCP, no localhost
listener, filesystem permissions as the ACL, the stance little-gemma has
kept from the start.

It is named for the field, not the filter: AEC is only the hardest of its
jobs today. With a multi-mic array (ReSpeaker XVF3800) the same tap stream
grows direction-of-arrival and beam metadata, which is what the world-model
layer upstream consumes — the likely future home of this tool is a
`little-gemma-world-model` repo; for now it lives in tools.

## Running it

```
far-field-service -s PATH [--source S] [--sink K] [--channels N]
                  [--use-channel K] [--pre-gain-db D] [--gain-db D] [--no-aec]
far-field-service --tap PATH [--raw]        clean (or raw) mic -> stdout
far-field-service --speak PATH [--rate N]   stdin PCM -> the speaker
```

`script/far-field.sh start|stop|status` is the lifecycle: the service is
persistent (conversations come and go; the card has one owner), `start` is
idempotent with ReSpeaker Lite defaults baked in, and `status` tails the
canceller's own statistics. Liveness is a pidfile — deliberately, because
the name is 17 characters, the kernel's `comm` is 15, and `pgrep -x` can
therefore never match it. A lesson paid for.

The voice pipeline composes without any voicecat changes
(`script/speakerphone.sh` is the packaged form):

```
far-field-service --tap /tmp/ff.sock \
  | voicecat /tmp/lg.sock --stdin-pcm ... \
      --mouth-synth "piper ... --output-mux --stream" \
      --mouth-play  "far-field-service --speak /tmp/ff.sock --rate 22050"
```

## The wire

House framing (`proto/lg_media_proto.h`): `{magic, type, u16 w, u16 h,
u32 len}` + payload. A client's first frame declares what it is:

| hello | meaning |
|---|---|
| `e` (ears) | a tap; `w=1` asks for the raw pre-cancel channel |
| `m` (mouth) | the speaker's owner; payload `rate=NNNNN` — one mouth at a time |

Then a mouth streams `P` frames (s16 mono PCM at its declared rate —
resampled here once, played, and fed to the canceller as reference), may
send `F` to flush, and `D` to drain: the service plays out what is queued,
waits the sink's tail, and acks one byte. Taps just read `P` frames of
16 kHz s16 mono.

**A mouth hanging up WITHOUT draining is the cut.** Everything it queued is
dropped and the sound stops within the sink's ~100 ms tail. This is not an
edge case, it is the barge-in contract: voicecat kills its speak client and
that kill IS the interruption. A finished mouth must send `D` first or lose
its tail — this bit us before it was a rule.

## The canceller seam (`src/aec.h`)

```c
struct aec *aec_new(int rate);
void aec_ref(struct aec *, const int16_t *, int);   /* about to be played */
void aec_mic(struct aec *, int16_t *, int);         /* cleaned in place */
void aec_stats(struct aec *, char *, int);          /* erl / erle / delay */
```

Two implementations, chosen at link time:

- `aec-webrtc.cpp` — webrtc AEC3 via the freedesktop `webrtc-audio-processing`
  **1.x** extraction (`modules/audio_processing` + minimal deps, NOT the
  WebRTC tree). The repo's only C++, quarantined in this one file.
- `aec-null.c` — pass-through: for hardware that cancels on-chip (the
  XVF3800 plan: run `--no-aec` first and let `aec_stats` and the test
  battery decide), and the fallback when the library is absent.

Build note that will save someone a day: Ubuntu jammy packages
webrtc-audio-processing **0.3.1, which predates AEC3** (it is also why
PipeWire 0.3.48's echo-cancel module underwhelms). Build 1.3 from source:
meson >= 0.63 (`pip3 install --user --ignore-installed meson`), and copy
`subprojects/abseil-cpp-*/absl` into the installed include dir — the meson
build vendors abseil but does not install its headers. Then configure with
`PKG_CONFIG_PATH=$HOME/.local/lib/<arch>/pkgconfig`.

## The duplex loop — every rule here was paid for

The card is the metronome: each 10 ms capture read drives exactly one
render block (queue or silence -> `aec_ref` -> sink) and then the matching
near block (channel pick -> pre-gain -> `aec_mic` -> post-gain -> taps).
The rules, each the survivor of a measured failure:

- **Feed the reference continuously, silence included.** AEC3 assumes both
  streams run in real time; a blocking queue wait between replies starved
  the reverse path and its delay estimator lost lock (converged on the
  startup hiss, then leaked the entire next utterance).
- **Never flush the sink stream.** A flush jumps the playback clock and the
  delay lock with it — every post-flush utterance leaked its onset. The cut
  clears only the queue; the sink's ~100 ms tail is the cut latency.
- **Keep an 80 ms cushion.** With `prebuf=0` and strict 1:1 writes the sink
  hovers at one block of buffer, and every >= 10 ms scheduling hiccup
  inserts silence: measured 130 near-silent 10 ms gaps per clip — severe
  audible warble. Pre-writing 8 silence blocks sets the hover depth; the
  gaps went to zero and the missing consonants came back.
- **Split the gain around the canceller.** A deaf channel (the Lite's raw
  mic sits ~30 dB low) needs boosting for VAD/ASR — but boosting before the
  canceller clips the echo peaks, and clipped echo is nonlinear, i.e.
  uncancelable (measured −22 dB leak at +28 dB pre-gain). `--pre-gain-db`
  stays inside clipping headroom to lift the echo into AEC3's working
  range; `--gain-db` lifts the cleaned signal after.
- **Prime at startup.** A soft fading hiss (0.8 s, low-passed) teaches the
  filter the echo path before the first reply. Without it: first reply
  leaks unconverged echo -> the client barges on the leak -> the flush
  starves the reference -> the filter never learns. A measured deadlock,
  and the hiss doubles as the ready chime.

## What the canceller actually achieves (ReSpeaker Lite, honest numbers)

On the Lite's v2.0.6 debug-firmware raw channel the *linear* filter learns
almost nothing — `aec_stats` shows the delay locked solid (~128 ms) with
ERLE 0–6 dB; the echo arrives ~30 dB deaf, at the edge of AEC3's design
envelope. What works is AEC3's **nonlinear output gate**: with no near-end
speech it holds the residual near the noise floor (clean tap −78 dB during
full-volume playback, whisper-blank) — the same mechanism that hides
Chrome's convergence. Gate confidence varies with conditions; treat any
single spectacular measurement with suspicion (an early −104 dB result was
a one-time outlier). The practical contract on this hardware: ghost-free
listening while the mouth speaks, barging needs a firm voice, and the
XVF3800's properly gain-staged mics are the expected real fix.

## Instrumentation

`aec_stats` prints one line per 2 s into the service log (`far-field.sh
status` shows the tail): `erl` (echo return loss), `erle` (what the linear
filter removes), `delay` (the estimator's lock). Delay stable + ERLE low =
signal-level problem, not alignment. Taps log why they exit; the service
logs every tap it drops and every mouth arrival/departure.

## Open items

- DoA / beam-index metadata frames on the tap stream (XVF3800; the
  world-model consumer).
- Multi-channel raw taps (`w` carries a channel count already).
- The XVF3800 decision battery: `--no-aec` vs AEC3 on its raw channels,
  judged by `aec_stats` + the echo/double-talk tests in
  little-gemma's `docs/voice-pipeline.md`.

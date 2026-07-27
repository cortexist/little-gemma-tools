# ReSpeaker Flex XVF3800 — two boards, one driver

The array is not a microphone. It is a DSP that publishes six channels, and
five of them will hurt you. Everything below was measured on 2026-07-26/27
across a circular-to-linear board swap; the swap is what exposed most of it,
because a hardware change is when inherited assumptions come due.

## The card id is the configuration

The USB product string and the ALSA card name are the same token, and it
encodes the whole firmware build:

    {C|L} {16K|48K} {2|6} Ch          C16K6Ch, L16K6Ch, c48k2ch, ...
     |      |        |
     |      |        +-- capture channels
     |      +----------- sample rate
     +------------------ geometry: Circular (squarecular) or Linear

Geometry is **read-only firmware state**, not something the board senses:

| | circular | linear |
|---|---|---|
| card / USB id | `C16K6Ch` / `2886:001e` | `L16K6Ch` / `2886:0022` |
| `AEC_MIC_ARRAY_TYPE` | 2 (squarecular) | 1 (linear) |
| `AEC_MIC_ARRAY_GEO` | (±22, ±22, 0) mm, 44 mm square | x = −50, −17, +17, +50 mm, y=z=0 |
| `AEC_NUM_MICS` | 4 | 4 |

Swapping the *array* while keeping the board therefore does nothing at all —
the beamformer keeps steering against the old geometry. Reflash the matching
image from `respeaker_flex/xmos_firmwares/usb/`. (A whole new board arrives
with its own correct firmware; check the card id before flashing anything.)

**The rule this repo now follows everywhere: match the PATTERN, never a
literal id.** Three separate places pinned literals and all three broke on the
swap, in three different ways:

| pinned | broke as | now |
|---|---|---|
| udev `idProduct=="001e"` | control needs root | match vendor `2886` |
| `far-field.sh` PulseAudio src/sink | service cannot find the card | `pactl` pattern + `FF_SRC`/`FF_SINK` |
| `far-field-service.c` libusb PID | tracker silently dies, blames udev | `xvf_open_any()`, vendor match |

`voicecat` matched the card id by pattern from the start and picked the new
board up with no edit at all. That is the whole argument.

## The channel map, measured

Six channels at 16 kHz S16_LE, `AEC_ASROUTONOFF=1` (beams, not raw mics),
`AEC_FIXEDBEAMSONOFF=0` (free-running):

| ch | what it is | speech SNR | during echo |
|----|-----------|-----------|-------------|
| **0** | **AEC + AGC beamformer output — the only one to feed ASR** | **+27.1 dB** | **−1.2 dB (immune)** |
| 1 | related processed output, ~55 ms later, corr 0.885 with ch0 | +16.8 dB | +14.7 dB |
| 2–5 | four mic-derived channels | +16.5…+17.0 dB | +21…+22 dB |

Two consequences worth stating plainly.

**Never average.** `ffmpeg -ac 1` mixes rather than selects. Echo in ch1–5 is
coherent so it sums, while ch0 is diluted to a sixth: **~17 dB worse echo
rejection and ~12.6 dB lower level**, for only ~3 dB of speech-SNR loss
(ch0 is ~28x hotter, so it still dominates the speech term). That asymmetry is
the signature — it degrades *echo* specifically, which is exactly the symptom
that makes someone add a software AEC. Use `pan=mono|c0=c0`.

**ch0 is already AGC'd to near full scale** — it peaks at −0.5 dBFS. Any
positive post-gain clips it. A `--gain-db 12` inherited from a quiet raw mic
put speech peaks ~11 dB over full scale, invisible in echo tests because echo
residual is far quieter than direct speech.

## The hardware AEC is sufficient, on ch0

With the software canceller disabled, cross-correlating the clean tap against
a played stimulus:

| | canceller on | `--no-aec` |
|---|---|---|
| reading **ch1** | 0.0106 (clean) | **0.4415 — echo present** |
| reading **ch0** | 0.0108 (clean) | **0.0102 — clean** |

The software AEC was only load-bearing because the service was reading ch1.
On ch0 the chip does the whole job, and `--no-aec` retires the canceller, its
convergence delay, and the priming hiss at startup.

## Direction of arrival is a cone, not a heading

Four microphones on one line are rotationally symmetric about that line. Every
direction on the same cone is indistinguishable — front, behind, above, below.
Measured twice, in two board orientations, on both trackers:

    CENTRE (in front)                    chip az = 87, 88, 91
    BEHIND (same spot, other side)       chip az = 90

The histogram correlator makes it explicit, emitting both solutions at once:

    09:49:41  src#10 NEW  az= -95
    09:49:41  src#11 NEW  az= +85      <- exactly 180 deg apart

So `az` carries **180 degrees of information**, and ~90 deg means "broadside",
which is both in front and behind. Any consumer must be written to that.
No orientation fixes it; orientation only chooses which physical pair
collapses. Keep the mic row horizontal and spanning where talkers sit — that
is the one axis the array can measure.

## Talker detection is sparse, and `--scene-gate` stays off

`AEC_SPENERGY_VALUES` is **bimodal** — exactly 0.000 between spikes, large
when it fires — so `--spe-thr 0` is a fine discriminator. It is also
instantaneous, so the poller now runs at 100 ms with a 0.6 s latch instead of
450 ms (`XVF_POLL_US` / `XVF_HOLD_SEC`).

That helped but was not the binding constraint: at 4.5x the sample rate there
were still only **3 spikes in ~32 s of continuous speech**, 6 talker reports
out of 94. The signal itself is sparse. Why is open — distance, beam steering,
or a gain the linear firmware sets differently.

Until that is understood, neither source is fit to gate on: the chip tracker is
accurate when it fires but fires rarely, and the correlator fires often but
emits mirror pairs a consumer would have to guess between. `--scene-gate` is
off, and that is the correct setting.

## Falsified along the way

- *"The software AEC compensates for a weak hardware AEC."* No — it compensated
  for reading the wrong channel.
- *"`--spe-thr 0` flags room noise."* No — spe is bimodal; the spread was spike
  amplitude, not a noise floor.
- *"The correlator cleanly separates left from right."* An artifact of catching
  one member of a mirror pair; it reports both.
- *"The tracker is undersampled."* Half right. Sampling was too slow, but the
  spike rate is the real limit.

## Settings that were inherited, not chosen

`--use-channel 1`, `--gain-db 12`, and the barge constants all arrived from the
**ReSpeaker Lite** defaults ("raw mic on capture channel 1, ~30 dB deaf") and
survived two hardware generations unexamined. Current values, each derived from
measurement on this board:

    far-field-service  --channels 6 --use-channel 0 --gain-db 0 --scene --no-aec
    voicecat           --vad-level 200 --barge-mult 7 --barge-onset 9 --barge-hang 7

`--barge-hang` exists because a strict consecutive-frame onset counter is the
wrong shape for speech: the /t/ closure in "wait a second" goes silent for
60–140 ms mid-phrase and zeroed the run, so that phrase could never interrupt
while "hold on" (continuously voiced) always could. Measured: "wait a second"
put MORE frames over threshold (22 vs 17) and was still rejected.

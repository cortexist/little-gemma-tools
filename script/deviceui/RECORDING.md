# Trio recording audio

The dashboard's **Room / Digital** switch chooses the audio for the next
recording. Choose before pressing Record; the choice is locked until capture
finishes. Arming waits for the session's models before opening any audio feeds.
The switch remembers its selection in this browser. Recording state from the
host takes precedence while a recording is active.

- **Room** records acoustic speech, overlap and room noise. With
  `recording_room_channels: [left, right]`, it selects two physical microphone
  channels (zero-based indices) as stereo. `recording_room_gain_db` applies a
  common gain, default 0 dB, limited to ±48 dB. Calibrate it against recorded
  peaks: unamplified raw mic channels can be much quieter than the DSP output.
  Gain belongs to a temporary recording source; physical microphone levels
  stay unchanged. Its PipeWire volume ceiling accommodates the configured
  gain (the usual +20 dB ceiling can silently limit amplification).
  Use a matched mic pair, not two
  differently processed DSP outputs. The selected capture profile must expose
  both channels. Raw microphones already hear local speech, so this mode adds
  no digital speaker overlay. Pair selection can be updated for new geometry.
  This is array stereo; a binaural head's acoustic cues require suitable
  microphone placement and head/ear construction.
  Without a configured pair, Room uses local speaker + processed microphone
  channel 0, labeled mono so it plays centrally through both output channels.
- **Digital** mixes the configured speaker output on every dashboard node,
  without microphones. It captures actual playback, including interruption,
  rather than Piper's potentially unaudible synthesis queue.

Both modes save MP4 video to `recording_directory`; trio filenames include
`room` or `digital`. Screen geometry and the existing record/start/stop barrier
are shared. The two-speaker classroom's configured recording mix is unchanged.

## Dependencies and transport

The dashboard host needs Wayland, wf-recorder and Pulse-compatible `pactl` and
`pacat`. Linux nodes normally use `parec`; nodes configured with
`recording_playback_tap: true` instead use a rebuilt `far-field-service` output
tap from their active voice session. This captures final 16 kHz mono blocks
after ducking/cancellation, before the system speaker volume. Other applications'
audio is excluded on that path. It avoids Pulse/ALSA monitor holes that can be
absent from physical playback. The tap never blocks the audio thread and closes
on backpressure instead of silently dropping PCM. Missing/stopped audio services
fail explicitly; there is no fallback to a microphone or a faulty monitor.
Windows nodes need
`PyAudioWPatch==0.2.12.8` installed in the node service's Python environment.
Its [WASAPI loopback support](https://github.com/s0d3s/PyAudioWPatch) captures the
exact configured speaker endpoint. A silent render stream keeps its clock
running through gaps between replies; no device defaults or microphone controls
are changed. Missing/ambiguous output devices fail explicitly.

`GET /api/recording/pcm` requires the existing node key and origin check. It
streams signed 16-bit little-endian PCM, with `X-PCM-Rate` and `X-PCM-Channels`
headers. A node permits one recording subscriber at a time. Capture is scoped
to that connection, has a bounded queue, and closes on disconnect or service
shutdown. No endpoint accepts an arbitrary device name from the caller.

The dashboard creates a separate null sink per digital recording and opens all
node feeds concurrently. Each feed has a fresh pacat stream with a 200 ms target
buffer and Pulse volume 58982 (90%, approximately -2.7 dB) for overlap headroom.
Linux capture uses the selected monitor's native rate or the output tap's 16 kHz,
and Pulse resamples each node's capture into the 48 kHz mix. Each feed supplies
200 ms of PCM before startup is acknowledged. Per-node verbose playback logs
are saved in a sibling `.audio` directory beside each digital MP4, including
underrun messages and negotiated buffer sizes. This is a
live LAN mix, not sample-synchronized multitrack recording. Feed failure stops
capture and reports an error; it never falls back to microphones or omits a
speaker silently. All three feeds must supply PCM before video is marked ready
and the session's opening is released.

The extra audio work exists only while recording and does not alter the
conversation, diarisation, ASR, model prompts, prefill or playback timing.

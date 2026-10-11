# Trio dashboard

`script/deviceui/dashboard.py` provides one monitor with independent panels for
Byers, Chey and Agent Smith. The browser reads one same-origin API; the hub
forwards only allowed operations to configured node endpoints. Node credentials
remain server-side. Its compact toolbar has Subject, Start, Stop and Record;
one shared Total turns control defaults to 12. Each speaker panel shows its
completed spoken-turn count, without an individual quota.

Each panel also has a compact timing summary. It shows session medians for
accepted transcript to first PCM and the estimated gap from the previous
speaker's PCM end. Expand it for latest/slowest values, sample counts, reported
ASR durations, and model/clause/TTS stages. Unsupported timings appear as a dash.
Room pipelines on both platforms emit first-PCM-ready telemetry. Legacy native
Linux sessions supply the first playback-pipe write as a proxy, marked `≈`.
Room input-to-PCM timing includes silent decisions, identity extraction and
waiting for a shared-budget permit; the expanded summary reports those stages. These are not hardware speaker-onset
measurements. Gap estimates use the nodes' scheduled PCM ends and clock offsets
estimated from the lowest-RTT status request in the session. The saved result
includes RTT uncertainty; audio buffering, embedded silence and clock drift can
still affect the estimate. The opening and same-speaker continuations have no
inter-speaker gap. Missing events or a detected clock jump suppress gap results
and produce a coverage notice.

Timing is collected in the host session loop, independently of the browser,
and checkpointed once a second under `runtime_dir/latency/<session-id>.json`.
`latest.json` restores the last measurements after a dashboard restart; an
interrupted capture is marked incomplete. Without `runtime_dir`, timing is
memory-only. Files contain per-turn timings, selected model paths, host labels,
clock samples and summary statistics, without transcripts or credentials. Keep
them private, with the deployment's configuration and source revisions when
making performance comparisons. Model loading is excluded from per-turn PCM
latency, but the first reply is included. Silent or unfinished replies do not
become zero-latency samples. Monitoring itself does not change speech behavior.

See [voice preparation and dependencies](trio-setup.md) before configuring nodes.

Example **private** configuration (replace keys and endpoints):

```json
{
  "key": "dashboard-access-key",
  "speaker_collection_id": "voice-trio",
  "nodes": [
    {"id": "teacher", "speaker_id": "catherine-byers", "url": "http://teacher-node:8098", "key": "byers-node-key"},
    {"id": "student", "speaker_id": "chey", "url": "http://student-node:8098", "key": "chey-node-key"},
    {"id": "guest", "speaker_id": "agent-smith", "url": "http://guest-node:8095", "key": "smith-node-key"}
  ]
}
```

```sh
python script/deviceui/dashboard.py --config /private/dashboard.json --host 0.0.0.0 --port 8097
```

Each node must serve the existing device API and report the expected collection
and speaker ID. Identity mismatch blocks controls. A failed endpoint affects its
own panel; Stop attempts every endpoint and reports individual failures.
Loading a model from this dashboard leaves listening off and sends only that
speaker's own prompt, without assigning a single peer or injecting the room
roster into its model context.

The shared page displays the device's model, transcripts, expression, phonemes,
playback timing, audio controls and current voice-identification status. It
keeps three columns above 580 CSS pixels, including a typical half-window
workspace. Details stay collapsed initially; smaller screens stack the panels.
The session coordinator runs on the dashboard host, so refreshing or closing
the browser does not reset counts or cancel the session. The shared limit accepts
1–48 spoken turns, including the opening. Silent observations and decisions never
consume that budget. Every participant keeps listening regardless of its own
spoken count. Completion waits for final playback and all listener observations
before stopping all pipelines and saving an armed recording. A node failure or
persistently malformed decision stops the session with an error.

Start validates every node before changing any of them. Each must advertise
`pipeline`, `capture`, and `room_session`. Missing capability keeps Start disabled.
The hub prepares every listener before the teacher opens. Subsequent permits
consume shared budget slots without exclusive floor arbitration; listeners can
respond concurrently.
Each listener independently hears the room, tracks its own acoustic voices,
learns names from evidence, and requests or declines the next turn. Known vocatives, returned questions and implicit addresses are interpreted by
each listener's model and decision head. There is no panel-order fallback. If every node
defers, all microphones remain open for another contribution. No transcripts,
voice labels or complete roster are relayed by the hub.

Each session resets learned identities and model context, retaining only the
configured prior acquaintances. See [room-sessions.md](room-sessions.md) for
CPU CAM++ configuration, spelling clarification, and current research limits.
The visible identity label uses the room's local track and accepted name;
unknown, uncertain, and learned names remain distinguishable.

Manual node loads continue to use their existing native Linux/Windows paths.
Legacy native turn limits still require a matching voicecat build and
`turn_limit_support`; those individual budgets are not the trio room policy.

Windows microphone configuration uses `capture_enabled`, `capture_hostapi`,
`capture_channels`, and `capture_rate` plus an owned `whisper_command`,
`whisper_health`, and `whisper_url`. Device selection requires the exact
configured endpoint name and API, without falling back to the system default.
The adapter uses processed channel 0 and preserves raw ASR spelling. Configure
`engine_client` with a freshly built little-gemma `socket_cat.exe`: its transport
must stream binary bytes before a newline. The legacy Windows engine `-c`
client cannot carry incremental input. Whisper must return timestamped VTT.

Capture continues while one ASR worker processes immutable audio snapshots.
Two agreeing hypotheses commit stable words as typed text frames into the open
model turn. Confirmed segment boundaries bound the retained ASR window; committed
words are never silently dropped. A pass with trailing silence finishes the
transcript, and the endpoint sends only the uncommitted suffix plus turn close.
ASR can finish during a coordinator hold and is reused unless speech resumes.
Prefixes and silent observations do not consume spoken turns. `input_commit` events expose their timing;
the engine's opt-in `LG_TRACE_PREFILL=1` logs actual open-turn prefill token IDs.

Before any text has committed, standalone input pauses discard pending audio and
in-flight results. Once a prefix is in the model cache, use **Stop** to discard
that open turn; Pause and unrelated manual prompts are rejected to prevent the
prefix leaking into the next turn. There is no model-cache rollback protocol.
`capture_diagnostics` optionally saves the last utterance under session state;
leave it disabled for normal use. ASR text, duration, and timing remain visible
in node events. Keep Whisper bound to loopback on a separate trio port.

Linux deployments can use `script/piper_named.py` as their Piper entry point.
The node passes `name_pronunciations` to that process and applies its mapping
only to spoken input; displayed transcripts and model history are preserved.
Set `env.PYTHONPATH` to the independent trio Piper source tree.

Windows keeps Piper loaded for the whole model session and streams complete
clauses as model bytes arrive. Its Piper checkout must support
`--output-mux --stream --mux-boundaries`: a `D` frame acknowledges each input
line after its last PCM frame. Turn completion waits for every submitted line
and for WASAPI playback to drain. Model-first-byte and first-PCM events make
generation and speech latency distinguishable. A node's optional
`default_profile` selects the initial model in the browser after a server restart.

Coordinated trio sessions add a spoken-answer policy of one or two short
sentences, at most 45 words, including when an older browser supplies its saved
prompt. This is a model instruction, not a hard output-token cap. The 90-second
stall guard watches incremental text, PCM, turn counters and the finite queued
playback tail; microphone noise alone does not keep a stalled session alive.

Recording uses the existing host recorder. Record arms capture; the coordinator
waits for model readiness and recording startup before the first prompt. Stop
finalizes the MP4, including on an automatic turn-limit stop. Configure a
separate `runtime_dir`, `recording_directory`, and `recording_prefix` for trio.
Supply an explicit `recording_audio_source` for a prepared mix, or `sink` and
`source` for local speaker-monitor plus processed room-microphone capture.
Room capture is not an isolated digital track per participant.

For a separate browser window, set `recording_window_title: "Voice dashboard"`.
For an embedded terminal browser, the host can instead pin
`recording_window_id` to its Sway container and `recording_crop` to fractional
`[x, y, width, height]`, such as `[0.5, 0, 0.5, 1]` for its right half. The
window must be visible; a missing/hidden target is rejected. Re-select the
container after replacing the terminal window and adjust the crop if the
split changes. These are private host settings, not browser-supplied commands.

A node can set `controls_only: true` while its independent trio pipeline is
being prepared. That exposes real device/audio status and volume/mute controls,
but rejects pipeline starts on the server and disables model/hello buttons in
the UI. Do not point an unfinished trio runtime at mutable classroom state.

The original one-node and two-node pages continue to use `server.py` and the
existing `peer` configuration. The new hub uses a separate config and port.

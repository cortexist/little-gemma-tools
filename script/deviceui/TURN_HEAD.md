# Trio decision head

## Resident C runtime

The live worker can use little-gemma's `decision_head` shared-library target.
Set `turn_head.backend` to `"c"`, `turn_head.library` to the absolute native
library path, and `turn_head.models` to a mapping from profile to
`{"path": "/path/decision-head.gguf", "sha256": "<file hash>"}`. Keep the
original `packages` mapping for reproducible NumPy benchmarks and explicit
rollback (`backend: "numpy"`, or omit backend). Mode remains independently
`off`, `shadow`, or `control`. There is no automatic fallback from C to NumPy.

The GGUF and library are loaded once per pipeline load, with a reusable native
workspace. Python still owns transport, scheduling, and control callbacks.
`ctypes.CDLL` releases the GIL for the CPU-only readout; no subprocess, CUDA
work, BLAS pool, or extra LLM request is added. Model hash, prompt format,
capture boundary, dimensions, layer, and answer-channel state are checked.
The library is freed by the worker after its last job, including on shutdown.

Status, per-decision records, and provenance include the actual runtime and
GGUF/library hashes. Original NPZ contract hashes remain training provenance.
The embedded historical `deployment_status` is not the current runtime mode.
Models, Prompt & Device Details shows the selected decision model and runtime;
an unloaded pipeline is explicitly marked `not loaded`.

Run `python -m unittest discover -s tests -p 'test_decision_head.py'` after
building the sibling engine target. `DECISION_HEAD_LIBRARY` and
`DECISION_HEAD_EXPORTER` override the default sibling paths for separate builds.

## Shadow observations

Set `turn_head.mode` to `shadow` in a private node configuration and provide
`turn_head.packages`, mapping model profile IDs to exported directories with
`head.npz` and `contract.json`. Only final-prompt speak/defer/wait heads are
supported. The model and head hashes must match. `save_hidden` defaults to true.
Omit the configuration or set mode `off` for the original transport/runtime.

The engine must support `LG_SNAPSHOT_HIDDEN=1`. RoomRuntime enables it only when
the shadow observer and binary frame decoder are installed. A snapshot arrives
before each reply; its immutable float32 bytes enter a bounded CPU-worker queue.
The worker uses the configured C or NumPy CPU readout; it never calls CUDA,
requests another generation, grants a budget permit, or cancels synthesis/audio.
Queue overload drops a diagnostic with an explicit event, never microphone data.

The live conversation prompt is different from the offline head's training
prompt. Results always carry `live_prompt_unvalidated`, and probabilities must
not be described as calibrated. Even with matching model/dimensions, this mode
measures runtime cost and exploratory disagreement, not validated live accuracy.
Interruption readouts require a different observation cutoff and are not run
on a start-turn vector. Existing acoustic speaker-ID/barge-in remains unchanged.

`turn_head_ready`, `turn_head_input`, `turn_head_decision`,
`turn_head_comparison` and errors are in the node session's `events.jsonl`.
The `turn-head/` subdirectory holds provenance, decisions and optional hidden
vectors. Connection/turn/response identifiers prevent cross-session ambiguity.
The dashboard's existing decision line says **Head shadow**; it is not a floor
grant. A disagreement with speech/[defer] is not a ground-truth error.

Timing distinguishes the engine's serialized snapshot copy, worker queue delay,
CPU inference and result readiness. Socket framing and event delivery also have
costs. Compare accepted-input-to-PCM and decoding with a matched baseline before
claiming a live latency improvement or no measurable overhead.

## Experimental head control

`turn_head.mode: "control"` and `text_decision_tags: true` opt into using the
saved head's argmax action. This requires an engine with the `LG_TURN_CONTROL=1`
same-turn prefix protocol; an old binary or missing head fails startup. No
probability threshold or authority priority is inferred from the head scores.

The immutable snapshot and conversational input remain at the original prefill
boundary. The CPU head returns a turn-scoped action before diagnostic disk I/O.
The engine prepares an empty answer channel when needed while the head runs,
then waits for its command before seeding `[speak]` or `[defer]` into the current
assistant turn. MTP continues the selected answer. This is a control dependency:
answer generation cannot run ahead of the decision. Engine logs separate the
remaining wait and prefix cost; do not claim fully parallel answer decoding or
zero overhead. There is no second request or user-prompt prefill.

`speak` permits content generation; the dashboard's existing total-turn budget
still governs playback. `defer` and `wait` produce no speech. Required name
metadata can still be generated privately; otherwise the engine closes the
silent response immediately. Wait is recorded distinctly and resolved only
by further acoustic input, not a timer or repeated generation. No unheard
candidate answer is added to silent-turn history. All listeners keep capturing.
Existing acoustic speaker-ID barge-in retains control over active playback.

An optional trio experiment, `turn_head.barge_hold.speak_below`, uses the
current reply's uncalibrated `speak` score as a local threshold. When new
speech begins during playback and the score is below the threshold, playback
holds at the next Piper word boundary. Quiet without a matched peer resumes
the buffered speech; confirmed peer identity still cuts it. The default is
off (`0`). This is an operational heuristic, not an interruption probability
or a cross-node comparison. Inspect `room_hold_request`,
`room_hold_release`, `room_barge`, and the audio service's hold/resume log
before making timing or quality claims. Windows gates WASAPI writes; Linux
requires the matching far-field-service `H`/`R` protocol.

`turn_head.barge_hold.grace_ms` (default `0`, range 0–2000) optionally keeps
playing real PCM before choosing the first word boundary at or after that
much additional audio. The trio PoC starts at `80`. A release before the
boundary cancels the pending stop without inserting silence, fading, skipping
or repeating samples. A later release resumes the retained queue as before.
Repeated hold/alignment updates do not restart the grace period. Confirmed
peer interruption still uses the existing cancellation path, bypassing this
unconfirmed-hold grace. No identity threshold or decision-head score changes.

The grace is measured in source samples: Linux anchors it at the mouth render
cursor; Windows anchors it after any already submitted WASAPI write. Device
buffering and the remainder of the word add to audible stopping time; this is
not an 80 ms acoustic-stop guarantee. Piper boundaries are alignment estimates,
not guaranteed quiet waveform points. Linux requires the rebuilt service's
`G` frame (uint32 grace in 16 kHz samples, followed by uint64 boundaries).
An empty boundary list may anchor a pending request before alignment arrives.
Its logs distinguish pending cancellation from an actual hold/resume, while
`room_hold_request.grace_ms` records the configured value. Set it to `0` to
restore next-boundary timing on the next session; config edits require an idle
node restart. This PoC does not establish an optimal grace or solve self-echo.

The wire tag is now `head_forced`, not an independent text-head prediction.
Timing/comparison records mark this distinction. Missing, stale or malformed
head commands fail closed; no fallback silently grants speaking. Current heads
remain experimental and live probabilities are uncalibrated. Set mode back to
`shadow` to restore independent text decisions on the next node start.

## Optional decision-tag timing

Set `turn_head.text_decision_tags` to `true` alongside `mode: "shadow"` to
request `[speak]` or `[defer]` **before** any private name metadata. `[speak]`
is stripped before synthesis. This changes the conversation prompt and can
change both head predictions and generated responses; it is a measurement
intervention, not a validated live policy. The default is false and preserves
the native prompt. Enabling tags in shadow mode does not grant head control.

`turn_text_decision` records the first complete decision tag at its closing
bracket, plus parser recognition time. Missing tags retain ordinary speech
behavior but are marked `implicit_speech`; they are not explicit-tag samples.
`turn_text_decision_outcome` records completion, missing/late tags, protocol
errors and cancellation. Join outcomes before reporting valid-tag statistics.
Early observations alone do not establish a valid completed response.

`turn_decision_timing` joins head readiness with text receipt using one process's
monotonic clock. Its origin is receipt of the complete hidden-state frame,
before decoding that frame. `text_after_frame_s` ends at the transport read
containing the closing bracket; `parser_after_receipt_s` exposes subsequent
queue/parser delay. `text_minus_head_s` is positive when the head was ready
first. Opening turns are flagged separately. Text, hidden state and several
MTP tokens may arrive in one read: `same_transport_read` preserves that limit
instead of inventing per-token times. This measures receiver availability,
not precise GPU token generation. Wall-clock engine-prefill intervals are
supplementary and must not be mixed across hosts. No extra LLM request is made.

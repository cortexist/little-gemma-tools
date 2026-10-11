# Room sessions — v1.0 GGUF

The trio dashboard uses one shared spoken-turn limit (default 12), including
the teacher's opening. Each listener independently chooses `speak`, `defer` or
`wait` using its same-turn decision head. The host permits requests while budget
slots remain; it does not rank speakers or impose an exclusive speaking floor.
Responses can overlap. If everybody defers, the room keeps listening.

The Python room path shares incremental ASR and clause-streamed Piper across
platforms. Stable ASR commits enter an open model turn while capture continues.
Decision metadata is appended at the endpoint. Final-prompt hidden features
feed the C classifier before generation; an allowed speak decision continues
that same model turn. Final audio and listener observations finish before
budget completion shuts down the session.

Node config `room_identity` supplies a hash-pinned CPU CAM++ model and WAV priors.
Only the listener's acquaintances from `trio-classroom.json` are enrolled. The
node environment needs numpy and sherpa-onnx. Each session resets model context,
anonymous tracks, learned names and pending spelling requests. Name proposals
must reference acoustic evidence. Spelling can correct a learned name only after
a request to the same voice and a literal letter sequence; prior acquaintances
cannot be overwritten. Raw ASR remains unchanged.

CAM++ runs in a separate CPU process, concurrently with capture, Whisper and
input prefill. It retains the classroom cadence: first window at 500 ms, then
every 250 ms, two consecutive score/margin-qualified matches, at most two
seconds per recognition window, and expiry after 350 ms quiet. Separate bounded
3/6/8-second windows prepare enrollment embeddings in parallel; short phonetic
prefixes cannot create new tracks. New anonymous tracks are finalized before
name learning. The live label is distinct from the last completed turn.
An utterance must have current single-speaker attribution before its turn
decision is generated. Missing attribution or confirmed speaker changes stop
the session rather than routing an anonymous transcript as if identity were known.

Silent metadata also consumes model context. At completed-response boundaries,
the node periodically reconnects and prefills bounded listener-local conversation
and identity memory. This prefix stays open for the next incoming speech or
floor grant; restoration generates no acknowledgement or spoken turn. It never
renews while an incremental speech input turn is open.

This is a research prototype. Acoustic thresholds and LLM decisions can be wrong.
The shared budget does not prevent simultaneous playback or reliably detect
human overlap. Invalid head contracts fail closed; runtime failures remain in the event log.
Silent decisions add model work before spoken generation. Existing manual node
pipelines and the preserved two-node classroom are separate from this path.

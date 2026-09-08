# Device voice monitor and teacher–student demo

`script/deviceui/server.py` provides a monitor for the physical voice pipeline on each device. The browser does not capture or play audio. A single dashboard can display a local teacher and a remote student side by side; each device also has its own page.

The UI shows pipeline phase, loaded model/head, configured Whisper and Piper voices, owned process status, rolling transcripts, expression and gesture events, phoneme-driven mouth shapes, and actual Pulse/PipeWire speaker volume and mute. The **Say hello** button provides a sound check after loading a pipeline. Changing a model or system prompt takes effect on the next load/session.

## Run one agent per device

Build tools normally and install Flask in an environment of your choice:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 -m venv /tmp/voice-ui-env
/tmp/voice-ui-env/bin/pip install -r script/deviceui/requirements.txt
```

Copy `script/deviceui/config.example.json` to a private local configuration file. Fill in exact paths and audio names from `pactl -f json list sources` and `pactl -f json list sinks`. Keep machine-specific paths, control keys and runtime logs outside the public repository. Generate a different key for each node, for example with `python3 -c 'import secrets; print(secrets.token_hex(24))'`, and protect configuration files with mode 0600.

- `profiles` is an allowlist of model paths and optional MTP heads/environment. Browser requests select profile names; they cannot submit executable paths or shell commands.
- `role` is `teacher` or `student`, used for the single-device page's defaults.
- `ff_socket` attaches to an existing far-field-service, which the UI does not stop. Omit it to let the UI start its own service. For a temporary input profile switch, additionally set `card` to the exact Pulse card name and `capture_profile` to the desired profile. The prior profile is restored on stop. Volume is never reset on start/stop.
- `whisper_command` starts an owned resident server. Omit it to attach to an existing server at `whisper_health`/`whisper_url`. Voicecat needs whisper.cpp-compatible timestamped VTT, not a JSON-only transcription endpoint. The ASR/model labels are configuration, while owned process liveness and HTTP readiness are checked separately.
- `piper` is an argv list; `voice` must have its config and streaming encoder/decoder companions. To expose actual phoneme durations, regenerate split halves with the Piper fork's current `python -m piper.split VOICE.onnx`. Use a separate copy/output directory when preserving existing exports. No schedule means the UI explicitly reports that lip sync is unavailable.
- `peer` is needed only on the dashboard coordinating both devices. It contains the other agent's fixed URL and key. Both agents are controlled under the same OS user that owns their audio session.

```sh
/tmp/voice-ui-env/bin/python script/deviceui/server.py \
  --config /private/device.json --host 0.0.0.0 --port 8095
```

The default bind is localhost. Binding to the LAN allows the other node and browser to connect. This is a trusted-LAN demo, not an Internet-facing service: the page supplies its device control key to the browser. APIs require that key and reject cross-origin browser requests. Use SSH forwarding or an authenticated reverse proxy for access outside the trusted LAN. Do not expose this Flask development server publicly.

The app starts/stops only its own inference and voice processes. It does not discover or take over arbitrary existing LLM sessions. Logs and the exact system prompt are written beneath the private `runtime_dir` in a separate directory for each load. Unexpected child exits produce an error state and cleanup; the pre-existing audio service remains running.

## Physical conversation

Load a model and edit its system prompt in each pane, or let **Start conversation** load both selected profiles. Choose a topic, a positive time limit and an input-turn limit from 1 to 48 (default 48, including the opening teacher prompt). Sessions end at the first limit reached. Unlimited sessions are not supported.

The teacher receives a text seed naming the topic. All subsequent cross-device exchanges are acoustic: speaker → other device's microphone → Whisper → model → Piper. The coordinator manages lifecycle and counts turns; it does not forward replies as text. Both microphones remain enabled throughout the conversation, including during local playback. Each device uses its normal AEC, speech detection and barge-in handling. Listening pauses only for explicit controls, session completion. During coordinated conversations, the controller renews a short endpoint hold while the peer has an unfinished reply or queued playback. Capture, ASR and incremental prefill continue; only silence-based turn closure waits. After estimated playback drains, a 200 ms acoustic tail is allowed. The 1500 ms lease expires if the controller disappears. This is a two-device coordination aid, not a general human endpoint detector, and genuine overlap still uses the existing barge-in behavior.

A turn limit waits for the final synthesis-complete marker and estimated queued playback to drain. A time limit or **Stop both** cuts the pipelines. A 90-second stall or device failure ends the session with a visible error.

The 48-input-turn cap is a conservative demo policy, not a measured context-window capacity. At the cap, final playback drains and both pipelines stop. There is no automatic reload, history summary or continuation seed. Longer conversation history management is deferred; initial model loading is excluded from the session timer.

## Metadata and synchronization

Expressions automatically return to neutral after the reply finishes synthesis and estimated queued playback drains; a generation-complete event alone does not end the expression. An explicit neutral expression can replace it earlier. Nod and shake run three 650 ms animation cycles (1.95 seconds) then return to quiet. Repeated gesture events restart the sequence; quiet cancels it. Stop and cut clear both. These dashboard defaults do not require model-generated closing tags. Gesture timing starts at metadata receipt, not phoneme-aligned placement within the spoken turn.


The optional voicecat flags are:

```text
--monitor-sock PATH     send best-effort JSON datagrams to a local AF_UNIX socket
--control-sock PATH     bind a local datagram control socket (must not exist)
--start-muted           drain capture without admitting microphone speech initially
```

Put these sockets in a private directory. Control messages are UTF-8 `listen 0`, `listen 1`, `endpoint-hold MS` (0–5000 ms, renewable silence-endpoint deferral), or `prompt TEXT` (one turn, up to roughly 3.5 KB through the HTTP API). A prompt is rejected while an utterance, reply or playback is in progress. It still enters the engine through the normal turn-close protocol, with voicecat tracking the expected reply.

Telemetry contains `kind`, Unix-epoch `at`, and `text`. Types include `input`, `input_done`, `reply`, `reply_done`, `clause`, `config`, `expression`, `gesture`, `alignment`, `play`, `synth_done`, `cut`, `listening`, and `rejected`. `input` is an incremental committed piece; `input_done` carries the remaining tail, not a second full transcript. The agent reconstructs the whole turn. Monitor sends are nonblocking and may drop events; a missing UI cannot block audio. The dashboard clears stale schedules when it detects a gap.

- Piper `A` schedules carry phoneme durations. `alignment.text` starts with `PCM-byte-offset<TAB>sample-rate<NEWLINE>`, followed by the original schedule.
- `play.text` is `start-epoch<TAB>PCM-byte-offset<TAB>byte-count<TAB>sample-rate`, based on bytes handed to the mouth player and its estimated audible horizon. Byte counters reset on a cut.
- `synth_done` carries the cumulative synthesized PCM byte count at a protocol turn end. It is driven by a silent Piper metadata sentinel, not a formatting newline or an inference-completion guess.
- `[happy]`, `[sad]`, `[angry]`, `[neutral]` reach expression telemetry and, with `--mood-route`, Piper. `[nod]`, `[shake]`, `[quiet]` reach gesture telemetry and are not spoken. Use these tags in system prompts, as in the page defaults. This voicecat path does not inherit every legacy browser-demo control syntax.

Lip sync is an estimate of device playback, not an acoustic measurement. The browser estimates its clock offset from each agent's status response, then applies a display-only **Visual offset** (default 100 ms) for downstream audio delay. Adjust it by observation. Expression/gesture tags are displayed on receipt rather than with sample-accurate actuator scheduling. Actual hardware actuator output is outside this UI. Browser disconnect/reconnect does not stop a session; **Stop both**, the timer, limits and server shutdown do.

## Validation

```sh
VOICECAT_TEST_BIN="$PWD/build/voicecat" /tmp/voice-ui-env/bin/python \
  -m unittest discover -s tests -p test_deviceui.py
```

The tests cover API key/origin boundaries, POST-only mutation, volume units, bounded event history, page rendering without browser audio, and a real voicecat run against fake engine/synth/player endpoints using local sockets. The latter exercises seeded turns, phoneme schedules, expression, gesture, playback and synthesis completion without touching audio hardware. Physical multi-turn, volume readback and browser checks belong in the protected research repository, not in public result dumps.


## Microphone controls

Microphone volume appears beneath speaker volume and uses the desktop capture-volume scale (0–100%). **Mute microphone** sets the system source mute and the configured ALSA capture switches; status distinguishes **Hardware capture muted**, **System capture muted**, and **Hardware capture enabled**. This works even when the inference pipeline is stopped. The UI uses a single microphone mute control. Explicit unmute also enables listening on a ready pipeline; loading from the UI enables listening but preserves hardware mute. The internal `listen` API remains available to the coordinator.

For supported arrays, configure `capture_controls` with the ALSA simple capture switch names (the tested XVF3800 arrays expose `Headset,0` and `Headset,1`). The source's ALSA card is resolved from its properties, including when its capture profile changes. Hardware mute is unavailable if no controls are configured, and control/readback errors are shown rather than claiming success.

An explicit mute is retained in `runtime_dir/microphone-muted` across agent restarts and reapplied before owned capture starts after a profile change. Only explicit unmute clears it. If an external audio tool changes capture switches while this mute is active, the agent reapplies mute on its next polling cycle. A runtime directory under `/tmp` may not survive reboot. This is a software-controlled USB hardware capture mute, not an electrical microphone power disconnect or a physical privacy switch; the array remains powered. Model loading and conversation start do not unmute capture automatically.

Control requests display a waiting spinner and disable duplicate actions until the request completes. Pipeline loading keeps a loading indicator until ready.

For speaker and microphone volume/mute changes, the waiting indicator remains until device status confirms the requested value. A missing confirmation times out with an error after 15 seconds.

`playback_controls` can pin hidden ALSA playback gains that USB resets may
silently restore to firmware defaults while the desktop volume remains unchanged.
Each entry has an ALSA control `name` and integer raw `value`; the dashboard
checks and restores it during audio polling without changing the mute switch.

Streaming Piper audio is not peak-normalized because playback begins before the
whole sentence exists. Configure measured corrections in `voice_gains_db`, keyed
by the exact voice model path. The correction is passed to Piper's streaming
volume control, is limited to ±12 dB, and does not add sentence buffering.

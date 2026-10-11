# v1.0 GGUF trio classroom setup

Use matching v1.0 tags of [little-gemma](https://github.com/cortexist/little-gemma/tree/v1.0)
and [little-gemma-tools](https://github.com/cortexist/little-gemma-tools/tree/v1.0).
The source includes the classroom runtime and dashboard, not a complete model
or deployment bundle. Supply the models, compatible external services and
private device configuration below before starting a session.

## Prepare three voices

Download [Agent Voice](https://huggingface.co/cortexist/agent-voice) for the guest.
Keep these published files together:

- `en_US-agent-voice.onnx`
- `en_US-agent-voice.onnx.json`
- `en_US-agent-voice.enc.onnx`
- `en_US-agent-voice.dec.onnx`

**Choose two additional Piper/VITS voices: one for the teacher and one for the
student. Chey's original voice weights are private, closed-weight and will not
be published.** The Chey character label in this source does not include a voice
license or download. Check the terms for each voice you choose.

For each replacement voice, prepare its original `.onnx`, matching `.onnx.json`
(with phoneme IDs and the correct phonemizer/language), phoneme-duration output,
and split `.enc.onnx`/`.dec.onnx` graphs. The JSON alone does not provide phoneme
timing. Use a Piper build with alignment and streaming support; the Agent Voice
model card links the [streaming implementation](https://github.com/OHF-Voice/piper1-gpl/pull/302).
The classroom additionally needs `--output-mux --stream --mux-boundaries`, including
per-line completion frames, and phoneme timing for the animated face.
A standard Piper installation without these extensions is insufficient.

In that compatible Piper environment, working on copies of the chosen models:

```sh
python -m pip install onnx
python -m piper.patch_voice_with_alignment /voices/teacher.onnx
python -m piper.patch_voice_with_alignment /voices/student.onnx
python -m piper.split /voices/teacher.onnx /voices/student.onnx
```

The alignment command edits the copied graph in place to expose duration output.
Skip that step if the model already exposes it. If automatic detection reports
multiple duration tensors, inspect the graph and use `--tensor-name` for the
correct duration tensor. Do not ignore alignment failures: splitting can succeed
without a phoneme schedule. Verify phoneme/timing events and streaming PCM with
each selected voice before using the room. Keep the matching JSON and both split
graphs beside the original model, with the same filename stem. Agent Voice
already supplies split graphs; no re-export is needed for that voice.

Copy [speakers.public.example.json](../script/deviceui/speakers.public.example.json)
to private configuration and replace `teacher.onnx` and `student.onnx` with the
actual model basenames. Set `speaker_collections` to this JSON path and
`speaker_collection_id` to `voice-trio` in both the dashboard and each node.
Set each node's `voice` to its full model path. The runtime resolves speaker
identity by exact basename: all three must match the collection, and each voice
must have a distinct basename. Keep the speaker IDs aligned with the scene's
acquaintance graph in `trio-classroom.json`; filenames and voices can change
without changing character IDs. The older `speakers.json` records development
voices and is not a list of supplied weights.

Generate fresh, clean WAV reference recordings from the three selected voices
for CAM++ enrollment. Configure only each listener's prior acquaintances and its
own self-reference; do not reuse enrollment recordings from the private Chey
voice or inject the full roster into every listener's prompt.

## Engine and external dependencies

Build the engine's `run-cuda-i8`, `decision_head`, `decision-head` and `socket_cat`
targets. On either Orin use `-DLG_CUDA_ARCH=87` at CMake configuration. Current
hardware names are **cortex1 = Orin Nano 8 GB Super** and **cortex0 = Orin NX**.
Build the tools with CMake, including `far-field-service` on Linux (PulseAudio
headers; WebRTC audio processing enables AEC). Install Python dependencies from
`script/deviceui/requirements.txt` and `requirements-room.txt` in your environment.

Supply these additional artifacts and services:

| Component | Requirement |
| --- | --- |
| Gemma | Compatible E4B or 12B QAT GGUF, and a matched MTP assistant GGUF if enabled. Model terms remain separate. |
| Decision head | [Published E4B/12B heads](https://huggingface.co/cortexist/gemma4-decision-head), matched to the exact backbone hash and final-prompt feature contract. See the model card's evaluation limits. |
| ASR and sound tags | Whisper weights, a compatible whisper.cpp build with the encoder feature tap, and a matching exported Whisper-AT ONNX classifier. The [integration patch and exporter](../integrations/whisper-at/README.md) are included; upstream builds/weights are separate. See [the adapter contract](acoustic-input.md). |
| Identity | CPU CAM++ ONNX model with SHA-256, sherpa-onnx, and WAV priors generated from your selected voices. |
| Audio | Configured microphone/speaker endpoints, Linux far-field service or Windows audio dependencies, and the compatible streaming Piper build above. |
| Recording (optional) | ffmpeg and an explicitly configured capture source/window/audio mix. |

The published heads use the `v2_sound_event` prompt lane. That lane name and the
head's v2.1 training revision do not mean the EXL3/v2 engine is part of this
release. Configure `turn_head.mode = "control"`, `text_decision_tags = true`,
`prompt_lane = "v2_sound_event"` and the matching `acoustic_input.mode =
"whisper_at_v2"`. The acoustic adapter requires the native feature-tap protocol
and pinned library/classifier hashes; substituting an unmodified Whisper server
is not sufficient. Missing per-request sound metadata has a trained fallback,
but does not remove these setup requirements. See
[TURN_HEAD.md](../script/deviceui/TURN_HEAD.md) for native head configuration.

Adapt [the device configuration](../script/deviceui/config.collections.example.json)
and [dashboard example](trio-dashboard.md) in private files. The device example
is a starting schema, not a complete trio deployment. Give nodes separate runtime
directories, ports, device keys, voice paths and model profiles. Check model/head
compatibility, ASR, identity references, phoneme events and first PCM per node
before starting all three. Local memory-service integration is optional.

Active mode runs the LLM/decision/TTS loop; Passive mode loads listening and
identity components without generation. See [room behavior](room-sessions.md)
for shared turn budgets and overlap limitations. Published throughput figures
are not promises of a particular end-to-end voice latency on your hardware.

# Classroom sound-event input format (v2)

Included in the v1.0 GGUF classroom. Prepare the external services using the
[Whisper integration patch and exporter](../integrations/whisper-at/README.md).

`turn_head.prompt_lane = "v2_sound_event"` selects the frozen v2 decision input
format and requires `acoustic_input.mode = "whisper_at_v2"`. The role prompt is
preserved; the shared metadata policy is appended at engine load and renewal.
Use the matching E4B/12B native head. Its backbone hash, hidden width/layer,
capture boundary and input format are checked before enabling control.

The teacher's explicit UI `room-open` grant starts the class with speak. It is a
user command, outside the listener decision targets; the dataset stages this
opening with `forced_opening=true`. The raw head prediction is still logged,
alongside `selection=explicit_room_open` and the applied action. Every heard
speech or standalone sound decision retains argmax control. An absent speaker
on ROOM_SOUND is not an opening grant.

`whisper_sound_server.py --config NODE_CONFIG` owns a private loopback native
Whisper server with the opt-in encoder tap. It serializes requests, deletes the
previous feature file, runs ASR, reads that request's features and deletes them
again. It verifies the library and ONNX head hashes. The native child is bound
to the adapter's lifetime on Linux. The original Whisper server/library remains
available for rollback. CPU tagging uses four threads and the released ten-second
pooling semantics; only real audio windows contribute logits.

The capture thread continues capturing during ASR. Partial passes still stream
immutable text without running the classifier. Settled/final passes request a
JSON envelope containing VTT, a structured observation and separate ASR/head
timings. Rolling ASR-window offsets align metadata to the original utterance.
Reusing a settled pass retains its observed span; it does not label later silence
that was never encoded. This deployment does not change ASR's prefix commitment
or implement transcript replacement/ASR_REVISION.

After admitted words, before ROOM_INPUT, a turn contains at most two canonical
AudioSet labels and raw logits rounded to one decimal:

```
 SOUND_EVENT {"t":[0,4],"c":[["Speech",0.5],["Speech synthesizer",0.4]]}
```

The original 527 labels are in `audioset_label.csv` (Whisper-AT/AudioSet inventory).
Labels are uncertain observations, never speaker identity or a speech veto. A
tagging error preserves the VTT transcript and records `sound_error`; speech
can use the trained missing-metadata path. A wordless event with valid metadata
uses ROOM_SOUND without a name or assigned voice. It does not advance the speech
epoch or conversational partner, bind a name, or answer a pending question.
VAD/minimum-duration admission and existing CAM++ speech attribution still apply.
Quiet background sounds below capture admission are not continuously tagged.

The room event stream records `asr.sound_timings`, aligned sound observations,
`room_sound`, head inputs and decisions. Those measurements exclude microphone
endpointing, Gemma, TTS and playback unless a separate end-to-end probe records
them. Offline head accuracy is not a live conversation completion guarantee.

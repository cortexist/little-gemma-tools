#!/bin/sh
# speakerphone.sh — the open-air voice loop on one full-duplex card:
# mic and speaker on the same box, no headset, barge-in included.
#
#     speakerphone.sh [SOCKET]            (default /tmp/lg.sock)
#
# voicedemo.py's hardware sibling: the browser mic becomes a real one and
# the AEC that a headset (or the browser's echoCancellation) used to provide
# comes from far-field-service (see docs/far-field-service.md) — the ears
# are its clean tap, the mouth is its --speak client, and what the speaker
# says is subtracted from what the mic hears, so the turn-taker can keep
# the mic open while the reply plays.
#
# Pieces are found through the environment (defaults in parentheses):
#     LG_BIN      run-cuda-i8 (server started only when SOCKET is absent)
#     LG_MODEL    the gguf              LG_MTP  draft head (optional)
#     LG_SYS      system prompt        (docs/voice-sys.txt beside LG_BIN)
#     LG_VOICE    piper voice .onnx — .enc/.dec split beside it for --stream
#     PIPER       piper executable     (piper)
#     LG_WHISPER_BIN / LG_WHISPER_MODEL   read by voicecat itself
# voicecat and clausecat are taken from this repo's build/ (or PATH).
set -u
sock="${1:-/tmp/lg.sock}"
here="$(cd "$(dirname "$0")" && pwd)"
tools="$here/../build"
vc="$tools/voicecat";  [ -x "$vc" ] || vc=voicecat
cc="$tools/clausecat"; [ -x "$cc" ] || cc=clausecat
piper="${PIPER:-piper}"

[ -n "${LG_VOICE:-}" ] || { echo "speakerphone: set LG_VOICE to a piper voice .onnx" >&2; exit 1; }
# the voice's PCM rate rides in its sidecar json
rate=$(grep -o '"sample_rate": *[0-9]*' "$LG_VOICE.json" 2>/dev/null | grep -o '[0-9]*$')
rate="${rate:-22050}"

# far-field-service is the persistent audio authority — started here if
# absent, but NOT owned: ctrl-c ends the conversation, the service keeps the
# card. `far-field.sh stop` is the explicit off switch.
ffsock=/tmp/ff.sock
sh "$here/far-field.sh" start || exit 1

# one cleanup for everything this run started (a later trap would replace
# an earlier one — never stack them); an external server/socket is left alone
srv=""; wpid=""
cleanup() {
    kill $srv $wpid 2>/dev/null
    [ -n "$srv" ] && rm -f "$sock"
}
trap cleanup INT TERM EXIT

# a socket FILE is not a server: a killed server leaves the file behind,
# and connecting to it hangs the ears — reap it unless something listens
if [ -S "$sock" ] && ! ss -xl 2>/dev/null | grep -qF "$sock"; then
    echo "speakerphone: removing stale socket $sock" >&2
    rm -f "$sock"
fi

if [ ! -S "$sock" ]; then
    [ -n "${LG_BIN:-}" ] && [ -n "${LG_MODEL:-}" ] || {
        echo "speakerphone: no server at $sock — set LG_BIN and LG_MODEL to start one" >&2; exit 1; }
    sys="${LG_SYS:-$(dirname "$LG_BIN")/../docs/voice-sys.txt}"
    "$LG_BIN" -m "$LG_MODEL" ${LG_MTP:+-mtp "$LG_MTP"} -sys "$sys" -s "$sock" &
    srv=$!
    while [ ! -S "$sock" ]; do
        kill -0 $srv 2>/dev/null || { echo "speakerphone: server died" >&2; exit 1; }
        sleep 0.3
    done
fi

# a resident whisper-server makes each ASR pass pay inference only (~0.36s
# vs 0.55s for a whisper-cli spawn) — start one if nobody is listening yet
wurl="http://127.0.0.1:8642/inference"
if ! curl -s -o /dev/null --max-time 2 "$wurl" 2>/dev/null; then
    wsrv="$(dirname "${LG_WHISPER_BIN:-whisper-cli}")/whisper-server"
    [ -x "$wsrv" ] && [ -n "${LG_WHISPER_MODEL:-}" ] || {
        echo "speakerphone: no whisper-server at $wurl and none startable" >&2; exit 1; }
    "$wsrv" -m "$LG_WHISPER_MODEL" --host 127.0.0.1 --port 8642 >/dev/null 2>&1 &
    wpid=$!
    until curl -s -o /dev/null --max-time 2 "$wurl" 2>/dev/null; do
        kill -0 $wpid 2>/dev/null || { echo "speakerphone: whisper-server died" >&2; exit 1; }
        sleep 0.3
    done
fi

echo "speakerphone: talk when ready (ctrl-c to stop)" >&2
# Ears: far-field-service's clean tap (AEC3'd, gain-lifted) pipes straight
# into --stdin-pcm. Mouth: voicecat owns piper (warm, mux-framed, exact barge
# discard) and its player is the --speak client — a barge KILLS it, the
# service flushes on the hangup, sound stops now; a clean close drains
# politely. --hush-tail is ON: with real cancellation, speech over the
# reply's tail is the user, not the echo.
# --clock 0: E2B parrots the "[18:52]" turn-clock into speech (E4B+ uses it
# silently as taught); keep time awareness off below the finetune boundary.
# --barge-mult 8 / onset 5: AEC3's output gate holds the TTS residual near
# the noise floor (measured −78 dB), so the in-reply bar only needs to clear
# noise, not echo — a firm word interrupts. (An earlier 4×/8 bar, set before
# the gate proved itself, was unreachable on this deaf channel: barge never
# fired and the reply just played out.)
# --duck-sock: two-stage barge — speech over the reply first DUCKS it (it
# keeps talking, 12 dB down); the hard cut waits for words to materialize,
# and a door slam or cough swells the reply back instead of killing it.
"$tools/far-field-service" --tap "$ffsock" \
  | "$vc" "$sock" --stdin-pcm \
      --vad-level 200 --hang-ms 500 --clock 0 --hush-tail \
      --barge-mult 8 --barge-onset 15 --duck-sock "$ffsock" \
      --whisper-url "$wurl" --commit-ms 1100 \
      --mouth-synth "$piper -m $LG_VOICE --output-mux --stream" \
      --mouth-play  "$tools/far-field-service --speak $ffsock --rate $rate"

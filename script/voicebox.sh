#!/bin/sh
# voicebox.sh — one command for the XVF3800 far-field voice loop (speakerphone).
#
#   voicebox.sh start | stop | status | logs [session]
#
# Owns the whole stack under tmux, so it survives the ssh session that starts it:
#   vb-lg    little-gemma serve (E2B QAT + MTP + voice-sys)  -> /tmp/lg.sock   (persistent)
#   vb-loop  speakerphone.sh: far-field-service (XVF3800, --no-aec) + voicecat loop
# whisper-server (:8642) and far-field-service are shared; reused if up, torn
# down by `stop`. Edit the paths block for a different model or voice.
set -u
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# ---- paths ----
LG="$HOME/repos/cortexist/little-gemma/build/run-cuda-i8"
MD="$HOME/repos/cortexist/llama.cpp/.scratch/gemma-4-e2b"
MODEL="$MD/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
MTP="$MD/mtp-gemma-4-E2B-it.gguf"
SYS="$HOME/repos/cortexist/little-gemma/docs/voice-sys.txt"
SPK="$HOME/repos/cortexist/lgt-duplex/script/speakerphone.sh"
FFSH="$HOME/repos/cortexist/lgt-duplex/script/far-field.sh"
VOICE="$HOME/repos/piper/local/model/oz_girl_v6-thresh18500-step18500.onnx"
PIPER="$HOME/repos/piper/.venv/bin/piper"
WBIN="$HOME/repos/whisper.cpp/build/bin/whisper-cli"
WMODEL="$HOME/repos/whisper.cpp/models/ggml-base.en.bin"
LGSOCK=/tmp/lg.sock
FFSOCK=/tmp/ff.sock
WURL=http://127.0.0.1:8642/inference

up() { [ -S "$1" ] && ss -xl 2>/dev/null | grep -qF "$1"; }

case "${1:-status}" in
start)
    # VOLUME lives in the ALSA mixer PCM,1 (mono output) — NOT pinned here, so
    # your tuning survives a restart (persisted via `sudo alsactl store`):
    #     amixer -c 0 sset PCM,1 <0-60>   (higher = louder; 60 = 0 dB max)
    #     sudo alsactl store
    # The hardware AEC only cancels ~16-20 dB, so near max the TTS echo residual
    # can climb over voicecat's VAD/barge and the loop self-barges — back PCM,1
    # off if that returns. The PulseAudio sink (far-field.sh vol=) is a minor
    # stage on top and is left fully open (100%).

    # 1) little-gemma serve — persistent, so restarting the loop won't reload the model
    if up "$LGSOCK"; then
        echo "lg     : reusing $LGSOCK"
    else
        rm -f "$LGSOCK"
        tmux new-session -d -s vb-lg "$LG -m $MODEL -mtp $MTP -sys $SYS -s $LGSOCK >/tmp/vb-lg.log 2>&1"
        i=0; while [ $i -lt 120 ]; do up "$LGSOCK" && break; sleep 0.5; i=$((i+1)); done
        up "$LGSOCK" && echo "lg     : up (E2B QAT + MTP)" || { echo "lg     : FAILED — tail /tmp/vb-lg.log"; exit 1; }
    fi
    # 2) the loop — speakerphone.sh brings up far-field (XVF3800) + whisper + voicecat
    if tmux has-session -t vb-loop 2>/dev/null; then
        echo "loop   : already running"
    else
        tmux new-session -d -s vb-loop \
            "env LG_VOICE=$VOICE PIPER=$PIPER LG_WHISPER_BIN=$WBIN LG_WHISPER_MODEL=$WMODEL sh $SPK $LGSOCK >/tmp/vb-loop.log 2>&1"
        sleep 3
        if tmux has-session -t vb-loop 2>/dev/null && up "$FFSOCK"; then
            echo "loop   : up — talk to the XVF3800  ('voicebox.sh logs' to watch)"
        else
            echo "loop   : FAILED — tail /tmp/vb-loop.log:"; tail -6 /tmp/vb-loop.log 2>/dev/null; exit 1
        fi
    fi
    ;;
stop)
    tmux kill-session -t vb-loop 2>/dev/null && echo "loop   : stopped" || echo "loop   : -"
    sh "$FFSH" stop 2>/dev/null | sed 's/^/far-fld: /'
    # reap any stray far-field-service tap/speak clients the loop left behind
    # (comm truncates to 15 chars, so pgrep/pkill -x can't match — use ps -C)
    left=$(ps -C far-field-servi -o pid= 2>/dev/null)
    [ -n "$left" ] && { kill $left 2>/dev/null; sleep 1; kill -9 $left 2>/dev/null; echo "far-fld: reaped stray clients"; }
    tmux kill-session -t vb-lg 2>/dev/null && { rm -f "$LGSOCK"; echo "lg     : stopped"; } || echo "lg     : - (external, left alone)"
    echo "whisper: left running (shared; pkill whisper-server to stop)"
    ;;
status)
    printf "lg.sock  %s\n" "$(up "$LGSOCK" && echo listening || echo down)"
    printf "ff.sock  %s\n" "$(up "$FFSOCK" && echo listening || echo down)"
    printf "whisper  %s\n" "$(curl -s -o /dev/null --max-time 2 "$WURL" && echo up || echo down)"
    for s in vb-lg vb-loop; do
        printf "%-8s %s\n" "$s" "$(tmux has-session -t "$s" 2>/dev/null && echo running || echo -)"
    done
    ;;
logs)
    tmux attach -t "${2:-vb-loop}"
    ;;
*)
    echo "usage: voicebox.sh start | stop | status | logs [vb-loop|vb-lg]" >&2
    exit 1
    ;;
esac

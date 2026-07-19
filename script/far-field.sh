#!/bin/sh
# far-field.sh — lifecycle for the audio authority. The service owns the
# sound card and outlives any one conversation (speakerphone.sh starts it if
# needed and leaves it running); this is the explicit handle.
#
#     far-field.sh start [EXTRA ARGS...]   idempotent; Lite defaults below
#     far-field.sh stop
#     far-field.sh status                  liveness + the canceller's own stats
#
# Liveness is a PIDFILE, deliberately: the name is 17 chars and the kernel's
# comm is 15, so pgrep/pkill -x can NEVER match it — a lesson paid for.
#
# Defaults target the ReSpeaker Lite on the v2.0.6 debug firmware (raw mic on
# capture channel 1, ~30 dB deaf -> post-cancel gain for VAD/ASR downstream).
# For the XVF3800: override --source/--sink/--channels/--use-channel, and try
# --no-aec first — its hardware AEC may retire the software canceller (the
# aec.h seam's whole point).
sock=/tmp/ff.sock
log=/tmp/far-field-service.log
pidf=/tmp/far-field-service.pid
here="$(cd "$(dirname "$0")" && pwd)"
bin="$here/../build/far-field-service"
[ -x "$bin" ] || bin=far-field-service
src=alsa_input.usb-Seeed_Studio_reSpeaker_Flex_XVF3800_C16K6Ch_100005504261800275-00.multichannel-input
sink=alsa_output.usb-Seeed_Studio_reSpeaker_Flex_XVF3800_C16K6Ch_100005504261800275-00.analog-stereo
vol=100%                              # pinned BEFORE launch: the sink volume is
                                     # part of the echo path the priming hiss
                                     # teaches — changing it mid-session forces
                                     # the canceller to re-converge

alive() {
    [ -f "$pidf" ] && kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null
}

case "${1:-status}" in
start)
    shift
    if alive; then
        echo "far-field: already running (socket $sock, pid $(cat "$pidf"))"
        exit 0
    fi
    rm -f "$sock"
    pactl set-sink-volume "$sink" "$vol" 2>/dev/null
    pactl set-source-volume "$src" 100% 2>/dev/null
    nohup "$bin" -s "$sock" \
        --source "$src" --sink "$sink" \
        --channels 6 --use-channel 1 --gain-db 12 --no-aec \
        "$@" >"$log" 2>&1 &
    echo $! > "$pidf"
    for i in 1 2 3 4 5 6 7 8 9 10; do
        [ -S "$sock" ] && break
        alive || { echo "far-field: died on start — $log:"; tail -3 "$log"; rm -f "$pidf"; exit 1; }
        sleep 0.3
    done
    echo "far-field: up on $sock (pid $(cat "$pidf"))"
    ;;
stop)
    if alive; then
        kill "$(cat "$pidf")" 2>/dev/null
        echo "far-field: stopped"
    else
        echo "far-field: not running"
    fi
    rm -f "$pidf" "$sock"
    ;;
status)
    if alive; then
        echo "far-field: running on $sock (pid $(cat "$pidf"))"
        grep -a erl "$log" 2>/dev/null | tail -3
    else
        echo "far-field: not running"
    fi
    ;;
*)
    echo "usage: far-field.sh start [ARGS...] | stop | status" >&2
    exit 1
    ;;
esac

#!/bin/sh
# respeaker-aec.sh — echo-cancelled speakerphone on a ReSpeaker Lite (or any
# full-duplex card) via PipeWire's webrtc echo-cancel module.
#
#     respeaker-aec.sh [up|down|status] [DEVICE_MATCH]
#
# The Lite's onboard XMOS AEC does not cancel its own USB playback (measured
# on fw v2.0.5; the Seeed forum reports the same through v2.0.7, and the two
# capture channels are bit-identical — no raw/processed split either). So the
# board serves as a plain 16 kHz full-duplex sound card and the cancelling
# runs on the host, where webrtc AEC3 gets the exact playback as reference —
# same USB device both ways means one clock, no drift to fight. Two virtual
# nodes appear:
#
#     ears : ec_src   voicecat <sock> --mic "-f pulse -i ec_src"
#     mouth: ec_sink  ... | paplay --device=ec_sink --raw --format=s16le \
#                              --channels=1 --rate=22050
#
# Anything played to ec_sink is subtracted from what ec_src hears: the mic
# stays open under TTS, which is what barge-in needs from an open-air rig.
# DEVICE_MATCH picks the card by node-name substring (default ReSpeaker) —
# the same wiring works for any speaker/mic pair on one card.
cmd="${1:-up}"
match="${2:-ReSpeaker}"

src=$(pactl list short sources | awk -v m="$match" '$2 ~ m && $2 !~ /\.monitor$/ && $2 !~ /^ec_/ {print $2; exit}')
sink=$(pactl list short sinks | awk -v m="$match" '$2 ~ m && $2 !~ /^ec_/ {print $2; exit}')

drop() {
    pactl list short modules | awk '/module-echo-cancel/{print $1}' \
        | while read -r i; do pactl unload-module "$i"; done
}

case "$cmd" in
up)
    if [ -z "$src" ] || [ -z "$sink" ]; then
        echo "respeaker-aec: no source/sink matching '$match'" >&2; exit 1
    fi
    drop
    # channels=2: webrtc cancels each capture channel independently. On the
    # Lite's v2.0.6 debug firmware ch1 is the RAW mic — the only channel a
    # canceller can actually work with (ch0 rides the XMOS NS/AGC, which
    # decorrelates the echo; measured ~1 dB there vs ~13 dB on raw). On a
    # plain stereo card the channels match and picking ch1 is harmless.
    pactl load-module module-echo-cancel \
        source_master="$src" sink_master="$sink" \
        source_name=ec_src sink_name=ec_sink rate=16000 channels=2 >/dev/null
    # the module's virtual source is born at 33% — a near-silent mic that
    # looks exactly like perfect echo cancellation — and its saved volumes
    # land asynchronously after load; settle first, then pin everything to 0 dB
    sleep 1
    pactl set-source-volume ec_src 100%
    pactl set-sink-volume   ec_sink 100%
    pactl set-source-volume "$src" 100%
    pactl set-sink-volume   "$sink" 100%
    echo "ec_src / ec_sink up on $match"
    echo "  ears : --mic \"-f pulse -i ec_src -af channelmap=map=1\""
    echo "  mouth: paplay --device=ec_sink --raw --format=s16le --channels=1 --rate=22050"
    ;;
down)
    drop
    echo "echo-cancel unloaded"
    ;;
status)
    pactl list short sources | grep -E 'ec_src|'"$match" || true
    pactl list short sinks   | grep -E 'ec_sink|'"$match" || true
    for n in ec_src "$src"; do
        [ -n "$n" ] && printf '%-12s %s\n' "$n" "$(pactl get-source-volume "$n" 2>/dev/null | head -1)"
    done
    for n in ec_sink "$sink"; do
        [ -n "$n" ] && printf '%-12s %s\n' "$n" "$(pactl get-sink-volume "$n" 2>/dev/null | head -1)"
    done
    ;;
*)
    echo "usage: respeaker-aec.sh [up|down|status] [DEVICE_MATCH]" >&2; exit 1
    ;;
esac

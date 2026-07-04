#!/bin/sh
# Progressive fake ASR for voicecat testing/showcases: prompt-aware (the words
# already trimmed away tell it where in the fixed utterance the window starts),
# emits timestamped ~2.5s segments proportional to the window it is handed, and
# sleeps like a fast whisper pass (~0.12x realtime of the WINDOW). Use with:
#   voicecat <sock> --stdin-pcm --realtime --whisper-model dummy --whisper-bin test/stub-whisper.sh
# and any >=10s PCM clip with speech-level energy. No ears, no external code.
WAV=""; PROMPT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -f) WAV="$2"; shift;;
    --prompt) PROMPT="$2"; shift;;
  esac
  shift
done
BYTES=$(stat -c %s "$WAV" 2>/dev/null || echo 44)
DUR=$(awk "BEGIN{printf \"%.2f\", ($BYTES - 44) / 32000.0}")
sleep $(awk "BEGIN{printf \"%.2f\", 0.12 * $DUR}")
NP=$(echo "$PROMPT" | wc -w)
TEXT="Please compare the three great ancient capitals of the old world and tell me which single one you would most want to visit, explaining the reasons for your choice in two or three sentences."
awk -v np="$NP" -v dur="$DUR" -v txt="$TEXT" "BEGIN{
  n = split(txt, W, \" \");
  take = int(2.4 * dur); end = np + take; if (end > n) end = n;
  segs = int(dur / 2.5); if (segs < 1) segs = 1;
  per = (end - np) / segs;
  for (s = 0; s < segs; s++) {
    t0 = s * 2.5; t1 = (s + 1) * 2.5; if (t1 > dur) t1 = dur;
    printf \"[00:00:%06.3f --> 00:00:%06.3f]  \", t0, t1;
    for (i = np + int(s * per) + 1; i <= np + int((s + 1) * per) && i <= end; i++) printf \"%s \", W[i];
    print \"\";
  }
}"
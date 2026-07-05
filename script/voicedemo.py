#!/usr/bin/env python3
# voicedemo — a browser push-to-talk demo of the little-gemma voice pipeline:
# mic -> whisper -> little-gemma serve -> clause split -> piper --stream -> ear,
# with optional phoneme timings in the UI (piper fork branch `phoneme-stream`).
#
#   python3 script/voicedemo.py --sock /tmp/lg.sock \
#       --client  ~/repos/cortexist/little-gemma/build/run-cuda-i8 \
#       --piper   ~/repos/piper/.venv/bin/piper \
#       --voice   ~/voices/en_US-kristin-medium.onnx \
#       --whisper-bin ~/repos/whisper.cpp/build/bin/whisper-cli \
#       --whisper-model ~/repos/whisper.cpp/models/ggml-base.en.bin \
#       [--phonemes] [--http] [--port 8443]
#
# Serves over ad-hoc HTTPS (self-signed, generated at start) so a browser on
# another machine grants microphone access; --http for localhost-only runs.
# One conversation, one user: the lg socket connection lives as long as the
# app, so multi-turn context works. The browser receives one streamed reply
# per turn, framed exactly like piper's mux ([kind][u32 len][payload]):
# 'T' transcript, 'R' reply clause, and piper's 'C'/'A'/'P' relayed through —
# the page is the demux. Requires flask + pyopenssl; ffmpeg on PATH.
import argparse
import os
import queue
import re
import struct
import subprocess
import tempfile
import threading
import time

from flask import Flask, Response, render_template_string, request

# ---- the reply framing (piper.mux's TLV, re-stated: 10 lines beat a dep) ----
def frame(kind, payload):
    return kind + struct.pack("<I", len(payload)) + payload


# ---- clause splitting (the policy of clausecat / bench/clause_pipe.py) -------
CLAUSE = re.compile(r"[,;:.!?](?=[\s\*\)\"'\]])")
TAG = re.compile(r"<[^<>]{1,24}>")
CHANNEL = re.compile(r"<\|channel>.*?(<channel\|>|$)", re.S)


def speakable(raw):
    return TAG.sub("", CHANNEL.sub("", raw))


# ---- the three companions -----------------------------------------------------
class Pipeline:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()          # one turn at a time
        self.lg = None
        self._lg_spawn()
        piper_out = ["--output-mux"] if args.phonemes else ["--output-raw"]
        self.piper = subprocess.Popen(
            [args.piper, "-m", args.voice] + piper_out + ["--stream"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        self.sink = None                      # the active turn's frame queue
        self.last_pcm = 0.0
        threading.Thread(target=self._piper_reader, daemon=True).start()

    def _lg_spawn(self):
        """(Re)connect the conversation: one `run -c` client per lg session.
        The server closes the session when the context fills, and the client
        exits with it — respawning starts a FRESH conversation."""
        self.lg = subprocess.Popen([self.args.client, "-c", self.args.sock],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        time.sleep(0.3)
        if self.lg.poll() is not None:
            raise RuntimeError(
                "lg client exited immediately — is little-gemma serving on %s?"
                % self.args.sock)

    def _emit(self, kind, payload):
        self.last_pcm = time.monotonic()
        sink = self.sink
        if sink is not None:
            sink.put(frame(kind, payload))

    def _piper_reader(self):
        """Relay piper's output as frames: mux frames pass through, raw PCM
        is wrapped as 'P' so the browser sees one format either way."""
        out = self.piper.stdout
        if self.args.phonemes:
            while True:
                head = out.read(5)
                if len(head) < 5:
                    return
                kind, length = head[:1], struct.unpack("<I", head[1:])[0]
                payload = out.read(length)
                self._emit(kind, payload)
        else:
            while True:
                data = out.read1(65536) if hasattr(out, "read1") else out.read(4096)
                if not data:
                    return
                self._emit(b"P", data)

    def transcribe(self, blob):
        """Browser audio (webm/ogg) -> 16 kHz wav -> whisper-cli -> text."""
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.webm")
            wav = os.path.join(td, "in.wav")
            with open(src, "wb") as f:
                f.write(blob)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", src, "-ar", "16000", "-ac", "1", wav], check=True)
            res = subprocess.run([self.args.whisper_bin, "-m", self.args.whisper_model,
                                  "-f", wav, "-nt"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        text = re.sub(r"\[[^\]]*\]", "", res.stdout)         # tolerate timestamped output
        return re.sub(r"\s+", " ", text).strip()

    def converse(self, blob):
        """One turn: yields framed transcript, reply clauses, and audio."""
        with self.lock:
            text = self.transcribe(blob)
            yield frame(b"T", text.encode("utf-8"))
            if not text:
                return

            q = queue.Queue()
            self.sink = q
            if not self.args.phonemes:                       # mux mode sends its own 'C'
                yield frame(b"C", b"rate=%d\nwidth=2\nchannels=1\n" % self.args.rate)

            reply_done = threading.Event()

            def speak(clause):
                clause = " ".join(clause.split())            # one piper line per clause
                if clause:
                    q.put(frame(b"R", clause.encode("utf-8")))
                    self.piper.stdin.write(clause.encode("utf-8") + b"\n")
                    self.piper.stdin.flush()

            def reply_to_clauses():
                # The lg client lives one conversation: the server closes the
                # session when the context fills, and the client exits with
                # it. Respawn (a fresh conversation) instead of dying.
                try:
                    if self.lg.poll() is not None:
                        raise BrokenPipeError
                    self.lg.stdin.write(text.encode("utf-8") + b"\n")
                    self.lg.stdin.flush()
                except (BrokenPipeError, OSError):
                    speak("The voice session restarted. Please ask again.")
                    self._lg_spawn()
                    reply_done.set()
                    return

                raw, spoken = b"", 0
                while b"<turn|>" not in raw:                 # replies may contain newlines;
                    b = self.lg.stdout.read(1)               # only the marker ends the turn
                    if not b:
                        break
                    raw += b
                    sp = speakable(raw.decode("utf-8", errors="replace"))
                    m = None
                    for m in CLAUSE.finditer(sp, spoken):
                        pass
                    if m is not None and m.end() > spoken:
                        clause, spoken = sp[spoken:m.end()], m.end()
                        speak(clause)
                self.lg.stdout.read(1)                       # the client's trailing newline
                speak(speakable(raw.decode("utf-8", errors="replace"))[spoken:])
                reply_done.set()

            threading.Thread(target=reply_to_clauses, daemon=True).start()
            self.last_pcm = time.monotonic()
            heard = False                                    # any PCM this turn yet?
            while True:
                try:
                    item = q.get(timeout=0.1)
                    heard = heard or item[:1] == b"P"
                    yield item
                except queue.Empty:
                    if not reply_done.is_set():
                        continue
                    # After the reply, wait 0.6s of TTS quiet — but give the
                    # FIRST audio a longer grace (piper's first-ever synthesis
                    # pays espeak init and can exceed the quiet window).
                    idle = time.monotonic() - self.last_pcm
                    if idle > (0.6 if heard else 5.0):
                        break
            self.sink = None
            yield frame(b"E", b"")


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>little-gemma voice</title>
<style>
  body { background:#101216; color:#d6d9de; font:16px/1.5 system-ui, sans-serif;
         max-width:40rem; margin:2rem auto; padding:0 1rem; }
  h1   { font-size:1.1rem; font-weight:600; color:#9aa3af; }
  #talk{ font:inherit; padding:.8rem 2.2rem; border-radius:.5rem; border:1px solid #3a3f47;
         background:#1a1e24; color:inherit; cursor:pointer; }
  #talk.rec { background:#5b1f1f; border-color:#a33; }
  .lbl { color:#6b7280; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em;
         margin:1.2rem 0 .2rem; }
  #you, #reply { min-height:1.5rem; }
  #ph  { color:#7fa1d4; font-family:ui-monospace, monospace; word-wrap:break-word; }
</style>
<h1>little-gemma — push to talk</h1>
<button id="talk">start talking</button> <span id="status"></span>
<div class="lbl">you said</div><div id="you"></div>
<div class="lbl">reply</div><div id="reply"></div>
<div class="lbl" id="phlbl" hidden>phonemes</div><div id="ph"></div>
<script>
let rec = null, chunks = [], ctx = null, cursor = 0, rate = 22050, sched = null;
const $ = id => document.getElementById(id);

$('talk').onclick = async () => {
  if (rec && rec.state === 'recording') { rec.stop(); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  if (!ctx) ctx = new AudioContext();
  chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = e => chunks.push(e.data);
  rec.onstop = () => { stream.getTracks().forEach(t => t.stop()); send(new Blob(chunks)); };
  rec.start();
  $('talk').textContent = 'stop & send'; $('talk').classList.add('rec');
};

async function send(blob) {
  $('talk').textContent = 'start talking'; $('talk').classList.remove('rec');
  $('you').textContent = '…'; $('reply').textContent = ''; $('ph').textContent = '';
  $('status').textContent = 'thinking';
  const res = await fetch('converse', { method: 'POST', body: blob });
  const reader = res.body.getReader();
  let buf = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const merged = new Uint8Array(buf.length + value.length);
    merged.set(buf); merged.set(value, buf.length); buf = merged;
    for (;;) {
      if (buf.length < 5) break;
      const len = new DataView(buf.buffer, buf.byteOffset + 1, 4).getUint32(0, true);
      if (buf.length < 5 + len) break;
      onframe(String.fromCharCode(buf[0]), buf.slice(5, 5 + len));
      buf = buf.slice(5 + len);
    }
  }
  $('status').textContent = '';
}

function onframe(kind, payload) {
  const text = () => new TextDecoder().decode(payload);
  if (kind === 'T') $('you').textContent = text();
  else if (kind === 'R') $('reply').textContent += (($('reply').textContent && ' ') || '') + text();
  else if (kind === 'C') { const m = /rate=(\\d+)/.exec(text()); if (m) rate = +m[1]; }
  else if (kind === 'A') { $('phlbl').hidden = false; sched = text(); }
  else if (kind === 'P') playPCM(payload);
}

function playPCM(bytes) {
  const n = bytes.length >> 1;
  if (!n) return;
  const i16 = new Int16Array(bytes.buffer, bytes.byteOffset, n);
  const buf = ctx.createBuffer(1, n, rate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < n; i++) ch[i] = i16[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf; src.connect(ctx.destination);
  cursor = Math.max(cursor, ctx.currentTime + 0.05);
  src.start(cursor);
  if (sched) {                       // anchor this sentence's phonemes to its audio
    const t0 = cursor;
    for (const line of sched.split('\\n')) {
      const [start, dur, ph] = line.split('\\t');
      if (ph === undefined) continue;
      const at = (t0 - ctx.currentTime + start / 1000) * 1000;
      setTimeout(() => { $('ph').textContent += ph + ' '; }, Math.max(0, at));
    }
    sched = null;
  }
  cursor += buf.duration;
}
</script>"""

app = Flask(__name__)
pipe = None


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/converse", methods=["POST"])
def converse():
    return Response(pipe.converse(request.get_data()),
                    mimetype="application/octet-stream")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sock", required=True, help="little-gemma serve socket")
    p.add_argument("--client", required=True,
                   help="path to a little-gemma run binary (used as `run -c <sock>`)")
    p.add_argument("--piper", required=True, help="path to the piper CLI")
    p.add_argument("--voice", required=True,
                   help="piper voice .onnx (split with `python3 -m piper.split` first)")
    p.add_argument("--whisper-bin", default=os.environ.get("LG_WHISPER_BIN", "whisper-cli"))
    p.add_argument("--whisper-model", default=os.environ.get("LG_WHISPER_MODEL"), required=False)
    p.add_argument("--phonemes", action="store_true",
                   help="show phoneme timings (needs the piper fork's `phoneme-stream` "
                   "branch and an alignment-enabled voice — see README)")
    p.add_argument("--rate", type=int, default=22050, help="voice sample rate")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--http", action="store_true",
                   help="plain http (localhost only — browsers gate the mic elsewhere)")
    args = p.parse_args()
    if not args.whisper_model:
        p.error("--whisper-model (or LG_WHISPER_MODEL) is required")

    global pipe
    pipe = Pipeline(args)
    app.run(host="0.0.0.0", port=args.port,
            ssl_context=None if args.http else "adhoc", threaded=True)


if __name__ == "__main__":
    main()

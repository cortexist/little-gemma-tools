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
import codecs
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

# The control spans this demo lets through to piper (clausecat's
# --allow-control-token, hardcoded to the voice switch), and how a speaker
# value is pulled out for the browser when no mux carries it.
ALLOW_CONTROL = "<|tool_call>call:set_voice{*}<tool_call|>"
SET_VOICE = re.compile(r"set_voice\s*\{(.*)\}")


def speakable(raw):
    return TAG.sub("", CHANNEL.sub("", raw))


def glob_match(s, p):
    """'*' = any run; the same iterative matcher as clausecat."""
    si = pi = 0
    star_p = star_s = -1
    while si < len(s):
        if pi < len(p) and p[pi] == "*":
            pi += 1
            star_p, star_s = pi, si
        elif pi < len(p) and p[pi] == s[si]:
            pi += 1
            si += 1
        elif star_p >= 0:
            star_s += 1
            pi, si = star_p, star_s
        else:
            return False
    while pi < len(p) and p[pi] == "*":
        pi += 1
    return pi == len(p)


# ---- visemes (--phonemes): the say-app experiment's Preston Blair set --------
# script/lipsync/*.svg are the 12 mouth shapes; the IPA map below is ported
# from the piper fork's say-app branch. espeak-ng also emits stress marks,
# length marks and boundary tokens — those map to nothing and the mouth keeps
# its previous shape.
LIPSYNC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lipsync")
REST_VISEME = "b_m_p"                     # closed mouth, start and end
PHONEME_TO_VISEME = {
    "a": "a_e_i", "æ": "a_e_i", "ʌ": "a_e_i", "ɐ": "a_e_i", "ɑ": "a_e_i",
    "ɛ": "a_e_i", "ɜ": "a_e_i", "ə": "a_e_i", "e": "a_e_i",
    "i": "ee", "ɪ": "ee",
    "o": "o", "ɔ": "o", "ɒ": "o",
    "u": "u", "ʊ": "u",
    "b": "b_m_p", "m": "b_m_p", "p": "b_m_p",
    "f": "f_v", "v": "f_v",
    "θ": "th", "ð": "th",
    "l": "l", "ɫ": "l",
    "r": "r", "ɹ": "r", "ɾ": "r", "ʁ": "r",
    "w": "q_w", "ʍ": "q_w",
    "ʃ": "ch_sh_j", "ʒ": "ch_sh_j", "tʃ": "ch_sh_j", "dʒ": "ch_sh_j",
    "c": "c_d_g_k_n_s_t_x_y_z", "d": "c_d_g_k_n_s_t_x_y_z",
    "g": "c_d_g_k_n_s_t_x_y_z", "k": "c_d_g_k_n_s_t_x_y_z",
    "n": "c_d_g_k_n_s_t_x_y_z", "s": "c_d_g_k_n_s_t_x_y_z",
    "t": "c_d_g_k_n_s_t_x_y_z", "x": "c_d_g_k_n_s_t_x_y_z",
    "j": "c_d_g_k_n_s_t_x_y_z", "z": "c_d_g_k_n_s_t_x_y_z",
    "h": "c_d_g_k_n_s_t_x_y_z", "ŋ": "c_d_g_k_n_s_t_x_y_z",
    "ʔ": "c_d_g_k_n_s_t_x_y_z",
}


def load_visemes():
    """lipsync/*.svg as inline markup (stroke: currentColor — the shapes wear
    the page's ink); an empty dict degrades to labels only."""
    svgs = {}
    try:
        for name in sorted(os.listdir(LIPSYNC_DIR)):
            if name.endswith(".svg"):
                with open(os.path.join(LIPSYNC_DIR, name), encoding="utf-8") as f:
                    svgs[name[:-4]] = f.read().strip()
    except OSError:
        pass
    return svgs


# ---- the three companions -----------------------------------------------------
class Pipeline:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()          # one turn at a time
        self.lg = None
        self._lg_spawn()
        self.piper = None
        self.piper_cfg = None                 # piper's startup 'C' frame payload
        self.sink = None                      # the active turn's frame queue
        self.last_pcm = 0.0
        self._piper_spawn()

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

    def _piper_spawn(self):
        piper_out = ["--output-mux"] if self.args.phonemes else ["--output-raw"]
        self.piper = subprocess.Popen(
            [self.args.piper, "-m", self.args.voice] + piper_out + ["--stream"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        if self.piper.poll() is not None:
            raise RuntimeError(
                "piper exited immediately — does it support %s --stream? "
                "(run `%s --help`; the voice halves come from `python3 -m piper.split`)"
                % (piper_out[0], self.args.piper))
        threading.Thread(target=self._piper_reader, daemon=True).start()

    def _emit(self, kind, payload):
        self.last_pcm = time.monotonic()
        if kind == b"C":
            self.piper_cfg = payload          # remember for every later turn
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
            if self.piper.poll() is not None:                # a dead mouth respawns
                self._piper_spawn()
            if self.piper_cfg is not None:                   # piper sent 'C' once at startup
                yield frame(b"C", self.piper_cfg)
            else:
                yield frame(b"C", b"rate=%d\nwidth=2\nchannels=1\n" % self.args.rate)

            reply_done = threading.Event()

            def speak(clause):
                clause = " ".join(clause.split())            # one piper line per clause
                if clause:
                    q.put(frame(b"R", clause.encode("utf-8")))
                    try:
                        self.piper.stdin.write(clause.encode("utf-8") + b"\n")
                        self.piper.stdin.flush()
                    except (BrokenPipeError, OSError):       # text still flows, audio doesn't
                        pass

            def reply_to_clauses():
                # The lg client lives one conversation: the server closes the
                # session when the context fills, and the client exits with
                # it. Respawn (a fresh conversation) instead of dying — and
                # reply_done ALWAYS fires, whatever this thread hits, or the
                # response generator would spin forever holding the turn lock.
                try:
                    try:
                        if self.lg.poll() is not None:
                            raise BrokenPipeError
                        self.lg.stdin.write(text.encode("utf-8") + b"\n")
                        self.lg.stdin.flush()
                    except (BrokenPipeError, OSError):
                        speak("The voice session restarted. Please ask again.")
                        self._lg_spawn()
                        return

                    def control(span_text):
                        """An allowed control span goes to piper as its own
                        line (it switches the speaker); dropped whole
                        otherwise. Without a mux to carry piper's M frame
                        (raw mode), the app reports the speaker itself."""
                        if not glob_match(span_text, ALLOW_CONTROL):
                            return
                        flush()                              # hold its place among clauses
                        try:
                            self.piper.stdin.write(span_text.encode("utf-8") + b"\n")
                            self.piper.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                        if not self.args.phonemes:
                            m2 = SET_VOICE.search(TAG.sub("", span_text))
                            if m2:
                                value = m2.group(1).split(":", 1)[-1].strip().strip("\"'")
                                q.put(frame(b"M", value.encode("utf-8")))

                    # clausecat's state machine, ported (see bench/clause_pipe.py):
                    # thought spans dropped, allowed control spans passed through,
                    # clauses flushed at punctuation-plus-space boundaries.
                    clause = [""]

                    def flush():
                        out = clause[0].strip()
                        clause[0] = ""
                        if out:
                            speak(out)

                    def add(c):
                        if clause[0] and clause[0][-1] in ",;:.!?" and c in " \t\n\r\f\v*)\"']":
                            flush()
                        clause[0] += c

                    tag, thought, cn = "", False, 0
                    span, tcall, tcn = "", False, 0
                    CLOSE, TCLOSE = "<channel|>", "<tool_call|>"
                    utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    raw = b""
                    while b"<turn|>" not in raw:             # replies may contain newlines;
                        b = self.lg.stdout.read(1)           # only the marker ends the turn
                        if not b:
                            break
                        raw += b
                        for c in utf8.decode(b):
                            if thought:
                                cn = cn + 1 if c == CLOSE[cn] else (1 if c == CLOSE[0] else 0)
                                if cn == len(CLOSE):
                                    thought, cn = False, 0
                                continue
                            if tcall:
                                span += c
                                tcn = tcn + 1 if c == TCLOSE[tcn] else (1 if c == TCLOSE[0] else 0)
                                if tcn == len(TCLOSE):
                                    control(span)
                                    span, tcall, tcn = "", False, 0
                                continue
                            if tag:
                                if c == ">":
                                    if tag == "<|channel":
                                        thought = True
                                    elif tag == "<|tool_call":
                                        span, tcall, tcn = "<|tool_call>", True, 0
                                    tag = ""
                                    continue
                                if c != "<" and len(tag) < 25:
                                    tag += c
                                    continue
                                for t in tag:
                                    add(t)
                                tag = ""
                            if c == "<":
                                tag = "<"
                                continue
                            add(c)
                    self.lg.stdout.read(1)                   # the client's trailing newline
                    flush()
                finally:
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
  #emorow { display:flex; align-items:center; gap:1rem; margin-top:.4rem; }
  #emo, #mouth { width:80px; height:80px; border-radius:.5rem; background:#181c22;
                 color:#d6d9de; flex:none; }
  #emo svg, #mouth svg { width:100%; height:100%; }
  #mouth[hidden] { display:none; }      /* stays flex-hidden until phonemes flow */
  #vis, #emoname { color:#6b7280; font-family:ui-monospace, monospace; }
</style>
<h1>little-gemma — voice</h1>
<button id="talk">enable microphone</button> <span id="status"></span>
<div class="lbl">you said</div><div id="you"></div>
<div class="lbl">reply</div><div id="reply"></div>
<div class="lbl">emotion</div>
<div id="emorow"><span id="emo"></span><span id="emoname"></span><span id="mouth" hidden></span><span id="vis"></span></div>
<div class="lbl" id="phlbl" hidden>phonemes</div>
<div id="ph"></div>
<script>
// Viseme mouth (--phonemes): the say-app branch's Preston Blair shapes,
// driven by the same schedule that paces the phoneme ticker.
const VISEMES = {{ visemes|tojson }};
const PH2VIS = {{ ph2vis|tojson }};
const REST = {{ rest|tojson }};
function visemeOf(p) {
  if (!p) return null;
  return PH2VIS[p] || PH2VIS[p[0]] || null;   // strip stress/length marks
}
function setMouth(v) {
  if (!v || !VISEMES[v]) return;
  document.getElementById('mouth').innerHTML = VISEMES[v];
  document.getElementById('vis').textContent = v;
}

// Emotion indicator (M frames): the same one-stroke language as the mouth.
// A set_voice value naming a known emotion gets its face; anything the
// mapping does not know (a bare speaker id, a voice name) reads as neutral —
// this row shows the EXPRESSION channel, not speaker identity.
function emoSvg(inner) {
  return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" fill="none"'
       + ' stroke="currentColor" stroke-width="2.2" stroke-linecap="round"'
       + ' stroke-linejoin="round">' + inner + '</svg>';
}
// No head outline: just eyes and a mouth, drawn at the same scale as the
// lipsync shapes so the two tiles read as one face split in half.
const EMO_EYES = '<path d="M12 10.5 L12 13.5"/><path d="M28 10.5 L28 13.5"/>';
const EMOTIONS = {
  happy:   emoSvg(EMO_EYES + '<path d="M8 23 C13 30 27 30 32 23"/>'),
  sad:     emoSvg(EMO_EYES + '<path d="M8 29 C13 22.5 27 22.5 32 29"/>'),
  neutral: emoSvg(EMO_EYES + '<path d="M8 24.5 C13 27 27 27 32 24.5"/>'),
  angry:   emoSvg(EMO_EYES + '<path d="M8 8.5 L15 11.5"/><path d="M32 8.5 L25 11.5"/>'
                  + '<path d="M9 28 C14 24 26 24 31 28"/>'),
};
function setEmotion(name) {
  const known = EMOTIONS[name] !== undefined;
  document.getElementById('emo').innerHTML = EMOTIONS[known ? name : 'neutral'];
  document.getElementById('emoname').textContent = known ? name : 'neutral';
}
window.addEventListener('DOMContentLoaded', () => setEmotion('neutral'));
</script>
<script>
// Hands-free turn taking: one click arms the mic; an energy VAD (voicecat's
// parameters: onset above threshold, HANG_MS of silence closes the turn)
// starts and stops the capture. While the mic is idle the recorder restarts
// every PREROLL_MS so the utterance keeps up to that much lead-in — the
// first syllable is never clipped. Listening gates off while the reply
// audio plays (no echo cancellation: an open-air mic would hear the voice).
const VAD_THRESH = 0.02, HANG_MS = 700, ONSET_N = 3, PREROLL_MS = 300;
let rec = null, chunks = [], ctx = null, cursor = 0, rate = 22050, sched = null;
let armed = false, talking = false, silentSince = 0, onsetRun = 0, busy = false;
const $ = id => document.getElementById(id);

$('talk').onclick = async () => {
  if (armed) { location.reload(); return; }              // disarm = reset
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  if (!ctx) ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser();
  an.fftSize = 1024;
  src.connect(an);
  const buf = new Float32Array(an.fftSize);
  const restart = () => {
    chunks = [];
    rec = new MediaRecorder(stream);
    rec.ondataavailable = e => chunks.push(e.data);
    rec.start();
  };
  restart();
  armed = true;
  $('talk').textContent = 'disarm mic';
  $('status').textContent = 'listening';
  let lastRestart = performance.now();
  setInterval(() => {
    if (!armed || busy) return;
    an.getFloatTimeDomainData(buf);
    let rms = 0;
    for (let i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / buf.length);
    const speaking = ctx.currentTime < cursor + 0.3;     // reply still playing
    const now = performance.now();
    if (!talking) {
      if (!speaking && rms > VAD_THRESH && ++onsetRun >= ONSET_N) {
        talking = true; silentSince = 0;
        $('talk').classList.add('rec'); $('status').textContent = 'recording';
      } else {
        if (rms <= VAD_THRESH) onsetRun = 0;
        if (now - lastRestart > PREROLL_MS) { rec.stop(); rec.onstop = restart; lastRestart = now; }
      }
    } else if (rms > VAD_THRESH) {
      silentSince = 0;
    } else if (!silentSince) {
      silentSince = now;
    } else if (now - silentSince > HANG_MS) {            // end of utterance
      talking = false; onsetRun = 0; busy = true;
      $('talk').classList.remove('rec');
      rec.onstop = () => { send(new Blob(chunks)); restart(); lastRestart = performance.now(); };
      rec.stop();
    }
  }, 50);
};

async function send(blob) {
  try { await send_(blob); } finally {      // never leave the VAD stuck on `busy`
    busy = false;
    $('status').textContent = armed ? 'listening' : '';
  }
}

async function send_(blob) {
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
}

function onframe(kind, payload) {
  const text = () => new TextDecoder().decode(payload);
  if (kind === 'T') $('you').textContent = text();
  else if (kind === 'R') $('reply').textContent += (($('reply').textContent && ' ') || '') + text();
  else if (kind === 'C') { const m = /rate=(\\d+)/.exec(text()); if (m) rate = +m[1]; }
  else if (kind === 'A') {
    $('phlbl').hidden = false;
    if (Object.keys(VISEMES).length) { $('mouth').hidden = false; setMouth(REST); }
    sched = text();
  }
  else if (kind === 'M') setEmotion(text());
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
    let end = 0;
    for (const line of sched.split('\\n')) {
      const [start, dur, ph] = line.split('\\t');
      if (ph === undefined) continue;
      const at = (t0 - ctx.currentTime + start / 1000) * 1000;
      end = Math.max(end, at + dur * 1);
      const v = visemeOf(ph);
      setTimeout(() => { $('ph').textContent += ph + ' '; setMouth(v); }, Math.max(0, at));
    }
    setTimeout(() => setMouth(REST), end + 60);   // sentence over: mouth closes
    sched = null;
  }
  cursor += buf.duration;
}
</script>"""

app = Flask(__name__)
pipe = None


@app.route("/")
def index():
    return render_template_string(PAGE, visemes=load_visemes(),
                                  ph2vis=PHONEME_TO_VISEME, rest=REST_VISEME)


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

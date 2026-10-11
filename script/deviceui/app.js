"use strict";
const config = window.VOICE_CONFIG;

// Theme only; face colors belong to speakers in the selected collection.
const appearanceReady = fetch(new URL("characters.json", document.currentScript.src), {
  cache: "no-cache",
}).then(async (response) => {
  if (!response.ok) throw new Error("Character colors could not be loaded");
  const colors = await response.json();
  const values = [colors.default_color, colors.happy_cheek_color, colors.unknown_color];
  if (!values.every((value) => typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)))
    throw new Error("Character colors must be six-digit hex values");
  return colors;
}).catch((error) => {
  console.error(error);
  return null; // Keep the face usable if the optional appearance file is unavailable.
});

const $ = (id) => document.getElementById(id);
function displayPath(value) {
  if (!value) return "—";
  const text = String(value);
  const path = /^[a-z]:[\\/]/i.test(text) ? text.replaceAll("\\", "/") : text;
  return path.replace(/^(?:\/home\/[^/]+|[a-z]:\/Users\/[^/]+)(?=\/|$)/i, "~");
}
// Assigning even identical text replaces its text node and clears selection.
function setText(element, value) {
  const text = value == null ? "" : String(value);
  if (element.textContent !== text) element.textContent = text;
}
function updateSoundTags(tags, status, state, now) {
  const seen = new Set();
  const categories = (state.last_sound_event?.candidates || []).filter(candidate => {
    if (typeof candidate.label !== "string" || !candidate.label.trim() || seen.has(candidate.label)) return false;
    seen.add(candidate.label);
    return true;
  }).map(candidate => candidate.label + " (score " + (Number.isFinite(candidate.logit)
    ? (candidate.logit >= 0 ? "+" : "") + candidate.logit.toFixed(2) : "unavailable") + ")");
  setText(tags, state.last_sound_error ? "Sound tags unavailable." : categories.length
    ? categories.join(" · ") : state.last_sound_at ? "No sound tags for last input." : "No sound tags yet.");
  tags.title = state.last_sound_error || "Raw uncalibrated logits: higher scores mean stronger model support. These are not confidence percentages. The top two labels are shown without a minimum score cutoff.";
  const activity = !state.ready ? state.phase === "loading" ? "loading" : state.error ? "pipeline error" : "stopped"
    : !state.listening ? "paused" : state.capture_active ? "hearing; waiting for tags" : "listening";
  const seconds = Number.isFinite(state.last_sound_at) ? Math.max(0, Math.floor(now - state.last_sound_at)) : null;
  const age = seconds === null ? "No update yet" : "Updated " + (seconds < 60 ? seconds + "s"
    : Math.floor(seconds / 60) + "m " + seconds % 60 + "s") + " ago";
  const span = state.last_sound_event?.span_seconds;
  const window = span?.length === 2 && span.every(Number.isFinite) && span[1] > span[0]
    ? " · " + (span[1] - span[0]).toFixed(1) + "s audio" : "";
  setText(status, "Uncalibrated scores · " + age + window + " · " + activity);
}
const modelDetailRows = new WeakMap();
function updateModelDetails(container, details) {
  let rows = modelDetailRows.get(container);
  if (!rows) {
    rows = new Map();
    modelDetailRows.set(container, rows);
  }
  // Labels are fixed; keep each row while other values change during polling.
  for (const [label, value] of details) {
    let dd = rows.get(label);
    if (!dd) {
      const dt = document.createElement("dt");
      dd = document.createElement("dd");
      setText(dt, label);
      container.append(dt, dd);
      rows.set(label, dd);
    }
    setText(dd, displayPath(value));
  }
  const visible = new Set(details.map(([label]) => label));
  for (const [label, dd] of rows) {
    dd.hidden = dd.previousElementSibling.hidden = !visible.has(label);
  }
}
async function api(path, body, method) {
  const response = await fetch("/api/" + path, {
    method: method || (body ? "POST" : "GET"),
    headers: { "X-Voice-Key": config.key, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(path.endsWith('/stop') ? 35000 : 12000),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
const common =
  "You are in a spoken conversation with another assistant. Keep each turn to one or two short sentences. Plain speech only. Do not use markdown or tool calls. You may use [happy], [sad], [angry], [neutral], [nod] or [shake] sparingly; these tags control the face and are not spoken. Respond to what the other speaker actually said. Do not invent their next reply. When the other speaker wants to end the conversation or take a break, acknowledge it with one brief farewell. Do not ask another question or reopen the topic. If you have already exchanged farewells, remain silent and end your turn without speaking. ";
const briefFirst = " Begin with a brief, meaningful clause that directly addresses the question, then elaborate if needed. Aim for 3 to 8 words in the opening clause when that is enough to be useful, and end that clause with natural punctuation. Avoid generic acknowledgements and filler words. Do not sacrifice accuracy or force a short opening when it would be misleading. The guidance about ending a conversation still takes priority.";
const panels = {};
class Panel {
  constructor(side, role, node = null) {
    this.role = role;
    this.node = node;
    this.connected = false;
    this.clockOffset = 0;
    this.side = side;
    this.cursor = 0;
    this.timers = [];
    this.schedules = [];
    this.segments = [];
    this.clauses = [];
    this.gestureTurn = 0;
    this.explicitGesture = false;
    this.autoGestureUsed = false;
    this.autoGestureUntil = 0;
    this.lines = [];
    this.ready = false;
    this.el = $("device-template").content.firstElementChild.cloneNode(true);
    this.el.dataset.node = side;
    $("devices").append(this.el);
    this.q = (cls) => this.el.querySelector("." + cls);
    const soundTags = this.q("heard-sound-tags");
    if (soundTags) {
      const details = document.createElement("div");
      details.className = "heard-sound-details";
      soundTags.before(details);
      details.append(soundTags);
      const status = document.createElement("small");
      status.className = "heard-sound-status";
      details.append(status);
    }
    this.captureStatus = this.q("mic-state").textContent;
    setText(this.q("role"), role.toUpperCase());
    if (node) {
      setText(this.q("name"), node.speaker.address_name || node.speaker.display_name);
      this.el.style.setProperty("--face-color", node.speaker.face_color);
      setText(this.q("host-platform"), side);
    }
    this.q("auto-gestures").checked = role.toLowerCase() === "student";
    this.q("auto-gestures").onchange = () => {
      this.autoGestureUntil = 0;
      this.q("portrait").classList.remove("auto-nod", "auto-shake");
    };
    this.q("prompt").value =
      common +
      (role.toLowerCase() === "teacher"
        ? "You are the teacher. While the conversation is continuing, explain the topic accurately, then ask your student one clear question. End your turn immediately after that question and wait for the real student. Never answer your own question or continue as the student. Encourage useful corrections. Your input comes from speech recognition and may contain errors. Use the conversation topic to interpret likely sound-alike words. Treat incidental transcription annotations such as [Music], (gentle music), and [BLANK_AUDIO] as non-speech; do not comment on them or invent an event from them. When the intended meaning is clear, respond naturally without criticizing the wording. When ambiguity affects the answer, ask a brief clarification instead of assuming the student is wrong. Do not invent an answer for a missing or unintelligible reply."
        : "You are the student. While the conversation is continuing, answer your teacher’s question, then ask one brief follow-up question about the topic. Admit uncertainty. Use your teacher’s address name politely when appropriate, such as before or after a question, without repeating it every turn. If you are unsure who is speaking, avoid guessing a name.") + briefFirst;
    if (side === "local" && config.default_prompt) this.q("prompt").value = config.default_prompt;
    if (node) this.q("prompt").value = node.default_prompt;
    this.setupAudioPopups();
    this.mouth("b_m_p");
    this.q("volume").oninput = () =>
      setText(this.q("volume-value"), this.q("volume").value + "%");
    this.q("volume").onchange = () =>
      this.act("volume", { percent: +this.q("volume").value });
    this.q("mute").onclick = () => this.act("volume", { mute: !this.muted });
    this.q("mic-volume").oninput = () =>
      setText(this.q("mic-volume-value"), this.q("mic-volume").value + "%");
    this.q("mic-volume").onchange = () =>
      this.act("microphone", { percent: +this.q("mic-volume").value });
    this.q("mic-mute").onclick = () =>
      this.act("microphone", { mute: !this.hardwareMuted });
    this.q("load").onclick = () => this.act("start", this.settings());
    this.q("hello").onclick = () =>
      this.act("prompt", {
        text: "Say hello in one short sentence so I can check your speaker volume.",
        reaction: "happy",
      });
    this.q("stop").onclick = () => this.act("stop", {});
    this.disableControls();
    this.poll();
  }
  disableControls() {
    this.el.querySelectorAll("button, input[type=range]").forEach(el => { el.disabled = true; });
  }
  setupAudioPopups() {
    const close = (restoreFocus = false) => {
      for (const name of ["speaker", "microphone"]) {
        const popup = this.q(name + "-popup"), button = this.q(name + "-toggle");
        if (!popup.hidden && restoreFocus) button.focus();
        popup.hidden = true;
        button.setAttribute("aria-expanded", "false");
      }
    };
    for (const name of ["speaker", "microphone"]) {
      const popup = this.q(name + "-popup"), button = this.q(name + "-toggle");
      popup.id = this.side + "-" + name + "-volume";
      button.setAttribute("aria-controls", popup.id);
      button.onclick = () => {
        const open = popup.hidden;
        close();
        popup.hidden = !open;
        button.setAttribute("aria-expanded", String(open));
        if (open) popup.querySelector("input").focus();
      };
    }
    document.addEventListener("pointerdown", (event) => {
      if (!this.q("audio-toolbar").contains(event.target) ||
          this.q("device-actions").contains(event.target)) close();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") { close(true); event.preventDefault(); }
    });
    this.q("audio-toolbar").addEventListener("focusout", (event) => {
      if (event.relatedTarget && (!this.q("audio-toolbar").contains(event.relatedTarget) ||
          this.q("device-actions").contains(event.relatedTarget))) close();
    });
  }
  settings() {
    if (this.node) return {
      profile: this.q("model").value, system_prompt: this.q("prompt").value, listen: false,
    };
    const peer = panels[this.side === "local" ? "peer" : "local"]?.speaker;
    return {
      peer_speaker_id: peer?.speaker_id || null,
      peer_collection_id: peer?.collection_id || null,
      profile: this.q("model").value,
      system_prompt: this.q("prompt").value,
      listen: this.side !== "local" || config.capture !== false,
    };
  }
  async act(action, body) {
    if (this.pendingAction) return;
    const labels = { volume: "Updating speaker…", microphone: "Updating microphone…",
      start: "Loading pipeline…", stop: "Stopping device…", prompt: "Sending prompt…" };
    this.pendingAction = labels[action] || "Working…";
    this.operationError = "";
    this.showBusy();
    try {
      await api(this.side + "/" + action, body);
      if (action === "volume" || action === "microphone") {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => {
            this.confirmation = null;
            reject(new Error("Device has not confirmed the requested audio state. Check the device and retry."));
          }, 15000);
          this.confirmation = { action, body, resolve: () => { clearTimeout(timer); resolve(); } };
        });
      }
    } catch (e) {
      this.operationError = e.message;
      setText(this.q("error"), e.message);
    } finally {
      this.pendingAction = "";
      this.showBusy();
    }
  }
  showBusy() {
    const text = this.pendingAction || (this.loading ? "Loading pipeline…" : "");
    const status = this.q("mic-state");
    setText(status, text || this.captureStatus);
    status.classList.toggle("busy", !!text);
    this.el.setAttribute("aria-busy", String(!!text));
    // Disable duplicate requests immediately, even before the next poll.
    if (this.pendingAction) {
      this.el.querySelectorAll("button:not(.speaker-toggle):not(.microphone-toggle), input[type=range]").forEach(el => { el.disabled = true; });
    }
  }
  muteIcon(selector, muted, device) {
    const button = this.q(selector);
    const label = (muted ? "Unmute " : "Mute ") + device;
    button.title = label;
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", String(!!muted));

  }
  mouth(name) {
    this.q("mouth").innerHTML =
      config.visemes[name] || config.visemes.b_m_p || "—";
  }
  clear() {
    for (const t of this.timers) clearTimeout(t);
    this.timers = [];
    this.schedules = [];
    this.segments = [];
    this.clauses = [];
    this.gestureTurn++;
    this.explicitGesture = false;
    this.autoGestureUsed = false;
    this.autoGestureUntil = 0;
    this.q("portrait").classList.remove("auto-nod", "auto-shake");
    this.mouth("b_m_p");
    setText(this.q("phoneme"), "—");
  }
  viseme(ph) {
    if (config.phonemes[ph]) return config.phonemes[ph];
    for (const c of ph) if (config.phonemes[c]) return config.phonemes[c];
    return null;
  }
  align() {
    const pending = [];
    for (const schedule of this.schedules) {
      const segment = this.segments.find(
        (s) =>
          schedule.offset >= s.offset && schedule.offset < s.offset + s.count,
      );
      if (!segment) {
        pending.push(schedule);
        continue;
      }
      const base =
        segment.start * 1000 +
        this.clockOffset +
        ((schedule.offset - segment.offset) * 500) / segment.rate +
        +this.q("offset").value;
      let end = base;
      const cue = speechGesture(schedule.text || "", schedule.rows);
      if (cue && base + cue.start >= Date.now()) {
        const turn = schedule.turn;
        this.timers.push(setTimeout(() => {
          if (turn !== this.gestureTurn || this.explicitGesture || this.autoGestureUsed ||
              !this.q("auto-gestures").checked) return;
          this.autoGestureUsed = true;
          this.autoGestureUntil = Date.now() + 325;
          const face = this.q("portrait");
          face.classList.remove("nod", "shake", "auto-nod", "auto-shake");
          face.style.animationDelay = "0s";
          void face.offsetWidth;
          face.classList.add("auto-" + cue.kind);
          setText(this.q("gesture"), cue.kind + " (automatic)");
          this.event({ kind: "auto_gesture", text: cue.kind + " on “" + cue.word + "”",
            at: Date.now() / 1000 });
          this.timers.push(setTimeout(() => {
            face.classList.remove("auto-nod", "auto-shake");
            this.autoGestureUntil = 0;
          }, 325));
        }, base + cue.start - Date.now()));
      }
      for (const row of schedule.rows) {
        const [start, dur, ph] = row.split("\t");
        if (ph === undefined) continue;
        const at = base + +start;
        end = Math.max(end, at + +dur);
        const shape = this.viseme(ph);
        if (at + +dur < Date.now()) continue;
        this.timers.push(
          setTimeout(
            () => {
              setText(this.q("phoneme"), ph);
              if (shape) this.mouth(shape);
            },
            Math.max(0, at - Date.now()),
          ),
        );
      }
      this.timers.push(
        setTimeout(() => this.mouth("b_m_p"), Math.max(0, end - Date.now())),
      );
    }
    this.schedules = pending.slice(-30);
    this.segments = this.segments.slice(-250);
    // Expired timer handles do not need to accumulate during infinite sessions.
    if (this.timers.length > 5000) this.timers = this.timers.slice(-2000);
  }
  event(e) {
    if (e.kind === "reset" || e.kind === "cut" || e.kind === "stopped")
      this.clear();
    if (e.kind === "input_done") {
      this.gestureTurn++;
      this.explicitGesture = false;
      this.autoGestureUsed = false;
      this.clauses = [];
    }
    if (e.kind === "gesture" && ["nod", "shake"].includes(e.text)) {
      this.explicitGesture = true;
      this.autoGestureUntil = 0;
      this.q("portrait").classList.remove("auto-nod", "auto-shake");
    }
    if (e.kind === "clause") this.clauses.push(e.text);
    if (e.kind === "alignment") {
      const lines = e.text.split("\n");
      const [offset, rate] = lines.shift().split("\t");
      this.schedules.push({ offset: +offset, rate: +rate, rows: lines,
        text: this.clauses.shift() || "", turn: this.gestureTurn });
      this.align();
    }
    if (e.kind === "play") {
      const [start, offset, count, rate] = e.text.split("\t").map(Number);
      this.segments.push({ start, offset, count, rate });
      this.align();
    }
    if (
      [
        "expression",
        "gesture",
        "auto_gesture",
        "clause",
        "input_done",
        "error",
        "rejected",
      ].includes(e.kind)
    ) {
      this.lines.push(
        new Date(e.at * 1000).toLocaleTimeString() +
          " " +
          e.kind +
          ": " +
          e.text,
      );
      this.lines = this.lines.slice(-35);
      setText(this.q("events"), this.lines.join("\n"));
    }
  }
  async poll() {
    try {
      const appearance = await appearanceReady;
      const requestedAt = Date.now();
      let data = await api(this.side + "/status?after=" + this.cursor);
      if ((this.instance && data.instance_id && this.instance !== data.instance_id) || data.cursor < this.cursor) {
        this.clear();
        this.cursor = 0;
        data = await api(this.side + "/status?after=0");
      }
      this.instance = data.instance_id;
      const s = data.state;
      this.connected = true;
      this.el.classList.remove("disconnected");
      this.speaker = data.speaker;
      if (appearance) {
        this.el.style.setProperty("--face-color", data.speaker?.face_color || appearance.unknown_color);
        this.q("portrait").style.setProperty("--cheek-color", appearance.happy_cheek_color);
      }
      this.clockOffset =
        (requestedAt + Date.now()) / 2 - data.server_time * 1000;
      if (
        this.cursor &&
        data.events.length &&
        data.events[0].id > this.cursor + 1
      )
        this.clear();
      if (!this.q("model").options.length) {
        for (const [key, p] of Object.entries(data.profiles)) {
          const o = new Option(p.label, key);
          this.q("model").add(o);
        }
        const preferred = s.profile || data.default_profile || (this.role.toLowerCase() === "teacher" ? "12b" : "e4b");
        if (data.profiles[preferred]) this.q("model").value = preferred;
        this.q("offset").value = data.sync_offset_ms;
      }
      setText(this.q("name"), (this.node ? data.speaker?.address_name : data.speaker?.display_name) || s.name);
      setText(this.q("host-platform"), data.host
        ? data.host.hostname + " · " + data.host.platform : "Host information unavailable");
      const pipeline = data.capabilities?.pipeline !== false;
      const passive = this.node && window.trioPassive || s.pipeline_mode === "passive" && (s.ready || s.phase === "loading");
      const loaded = this.q("loaded");
      const whisper = data.asr ? String(data.asr).replaceAll("\\", "/").split("/").pop()
        .replace(/^ggml-/, "").replace(/\.bin$/i, "") : "not configured";
      const whisperStatus = s.ready ? "loaded" : s.phase === "loading" ? "loading…" : "not loaded";
      if (passive) {
        setText(loaded, "Whisper " + whisper + " · " + whisperStatus);
        loaded.title = data.asr ? displayPath(data.asr) : "Whisper model not configured";
      } else {
        setText(loaded, !pipeline ? "Voice pipeline not configured" : s.ready ? (data.profiles[s.profile]?.label || s.profile) + " loaded" : s.phase === "loading" ? "Loading model…" : "No model loaded");
        loaded.title = "";
      }
      const note = this.q("node-note");
      const noCapture = data.capabilities?.capture === false || s.microphone?.capture_enabled === false;
      if (note) { // A running node may still have the previous HTML template cached.
        note.hidden = !this.node || (pipeline && !noCapture);
        setText(note, !pipeline ? "Audio controls ready. Trio model loading is not available yet." : "Playback available. Microphone conversation is not enabled yet.");
      }
      setText(this.q("phase"), s.phase);
      this.q("phase").dataset.phase = s.phase;
      this.ready = s.ready;
      this.state = s;
      this.loading = s.phase === "loading";
      this.q("stop").disabled = s.phase === "stopped";
      this.q("speaker-toggle").disabled = false;
      this.q("microphone-toggle").disabled = false;
      this.q("volume").disabled = false;
      this.q("mute").disabled = false;
      this.q("hello").disabled =
        !pipeline || !s.ready || s.finishing || ["speaking", "thinking", "hearing"].includes(s.phase);
      this.q("load").disabled = !pipeline || !Object.keys(data.profiles).length || !["stopped", "error"].includes(s.phase);
      if (this.node && (window.trioSessionActive || window.trioPassive || s.pipeline_mode === "passive" && s.ready)) {
        this.q("load").disabled = this.q("hello").disabled = true;
      }
      if (s.volume) {
        this.muted = s.volume.mute;
        this.muteIcon("mute", this.muted, "speaker");
        this.q("speaker-toggle").classList.toggle("is-muted", !!this.muted);
        this.q("speaker-toggle").title = "Speaker volume: " + s.volume.percent + "%" + (this.muted ? " (muted)" : "");
        if (document.activeElement !== this.q("volume")) {
          this.q("volume").value = s.volume.percent;
          setText(this.q("volume-value"), s.volume.percent + "%");
        }
      }
      const mic = s.microphone;
      this.q("mic-volume").disabled = !mic || !!s.audio_error;
      this.q("mic-mute").disabled = !mic || mic.hardware_mute === null || !!s.audio_error;
      if (mic) {
        this.hardwareMuted = mic.hardware_mute;
        this.muteIcon("mic-mute", mic.hardware_mute, "microphone");
        this.q("microphone-toggle").classList.toggle("is-muted", !!(mic.hardware_mute || mic.mute));
        this.q("microphone-toggle").title = "Microphone volume: " + mic.percent + "%";
        this.q("mic-mute").setAttribute("aria-pressed", String(!!mic.hardware_mute));
        this.captureStatus = s.audio_error ? "Microphone state unavailable" :
          mic.capture_enabled === false ? (mic.mute ? "Windows microphone muted · playback test" : "Windows microphone enabled · playback test") :
          mic.hardware_mute ? "Hardware capture muted" : mic.mute ? "System capture muted" :
          mic.hardware_mute === null ? "Hardware capture mute unavailable" : "Hardware capture enabled";
        if (document.activeElement !== this.q("mic-volume")) {
          this.q("mic-volume").value = mic.percent;
          setText(this.q("mic-volume-value"), mic.percent + "%");
        }
      }
      else this.captureStatus = "Microphone state unavailable";
      const identity = s.speaker_identity || { status: "disabled" };
      const previous = identity.last_peer?.display_name;
      const confirmed = identity.status === "identified" && !!identity.speaker_id;
      const faceSpeaker = confirmed ? identity : identity.last_peer;
      this.q("heard-face").style.setProperty("--heard-color", faceSpeaker?.face_color || appearance?.unknown_color || "#d9dde1");
      this.q("heard-face").style.opacity = !confirmed && faceSpeaker ? "0.45" : "1";
      this.q("heard-face").setAttribute("aria-label", confirmed ? identity.display_name :
        previous ? "Unconfirmed speaker; last identified: " + previous : "Unknown speaker");
      const labels = { disabled: "Voice identification disabled.", waiting: "Listening for a voice…",
        identifying: "Identifying voice…", uncertain: "Voice uncertain.", quiet: "Listening…",
        other_voice: "Another voice; peer not identified.", audio_gap: "Audio gap; identifying again…",
        unavailable: "Voice identification unavailable.", paused: "Voice tracking paused.", stopped: "Voice tracking stopped." };
      let heardIdentityText = confirmed && identity.display_name
        ? identity.display_name + " · voice estimate"
        : (labels[identity.status] || "Voice uncertain.") + (previous ? " Last identified: " + previous + "." : "");
      if (s.room?.identity) {
        const local = s.room.identity, turn = local.turns.at(-1);
        const track = local.tracks.find(t => t.track_id === turn?.track_id);
        const live = s.ready ? s.room.live_identity : null;
        const fresh = live?.status === "identified" && live.epoch === s.room.epoch &&
          (Date.now() - this.clockOffset) / 1000 - live.at < 1;
        heardIdentityText = fresh
          ? (live.name || "Unknown voice") + " · " + live.track_id + " · voice estimate"
          : live?.status === "identifying" ? "Identifying voice…"
          : live?.status === "unknown" ? "Unknown voice · tracking…"
          : (s.ready ? "Listening…" : "Stopped.") + (track
            ? " Last: " + (track.name || "Unknown voice") + " · " + track.track_id
            : turn ? " Voice uncertain." : "");
        this.q("heard-face").style.opacity = fresh ? "1" : "0.45";
        this.q("heard-face").setAttribute("aria-label", fresh ? live.name || "Unknown voice" : "No current voice identified");
      }
      setText(this.q("heard-identity"), heardIdentityText);
      const soundTags = this.q("heard-sound-tags");
      if (soundTags) {
        updateSoundTags(soundTags, this.q("heard-sound-status"), s, (Date.now() - this.clockOffset) / 1000);
      }
      setText(this.q("heard"), s.input || s.last_input || "Waiting for speech.");
      setText(this.q("said"), s.reply
          .replace(/<\|channel>.*?<channel\|>/gs, "")
          .replace(/<\|tool_call>.*?(?:<tool_call\|>|\n)/gs, "")
          .replace(/<[^>]*>/g, "")
          .replace(/\[(?:happy|sad|angry|neutral|nod|shake|quiet)\]/g, "") ||
        "No reply yet.");
      setText(this.q("expression"), s.expression);
      this.q("portrait").dataset.mood = s.expression;
      if (Date.now() >= this.autoGestureUntil) setText(this.q("gesture"), s.gesture);
      const face = this.q("portrait");
      const gestureKey = s.gesture + ":" + (s.gesture_until || 0);
      if (gestureKey !== this.gestureKey) {
        this.gestureKey = gestureKey;
        face.classList.remove("nod", "shake");
        if (["nod", "shake"].includes(s.gesture)) {
          void face.offsetWidth;
          face.style.animationDelay =
            -Math.max(0, (Date.now() - this.clockOffset) / 1000 - (s.gesture_until - 1.95)) + "s";
          face.classList.add(s.gesture);
        }
      }
      setText(this.q("alignment"), s.alignment
        ? "Phonemes · estimated playback sync"
        : "No phoneme schedule received");
      const configuredHead = data.decision_heads?.[s.ready ? s.profile : this.q("model").value];
      const headState = s.ready ? s.room?.turn_head : null;
      const headRuntime = headState?.runtime;
      const headEnabled = configuredHead && configuredHead.mode !== "off";
      const headBackend = headRuntime?.backend || configuredHead?.backend;
      const headStatus = headState?.status === "error" ? "error"
        : headRuntime ? "loaded" : s.phase === "loading" ? "loading" : "not loaded";
      const details = passive ? [
        ["Whisper", data.asr],
        ["Whisper status", whisperStatus],
        ["Microphone", data.source],
        ["Speaker", data.sink],
        ["Processes", Object.entries(s.components)
          .map(([k, v]) => k + ": " + v.pid + (v.alive ? " running" : " exited")).join(" · ")],
      ] : [
        [
          "LLM",
          s.ready ? data.profiles[s.profile]?.model : "Not loaded / loading",
        ],
        ["MTP", s.ready ? data.profiles[s.profile]?.head : "—"],
        ["Decision head", headEnabled ? headRuntime?.model || configuredHead.model || "Not configured" : "Disabled"],
        ["Decision runtime", headEnabled
          ? [headBackend === "c" ? "C" : headBackend === "numpy" ? "NumPy" : headBackend,
             "CPU", headState?.mode || configuredHead.mode, headStatus].join(" · ") : "—"],
        ["Whisper", data.asr],
        ["Piper", data.voice],
        ["Speaker", data.sink],
        ["Microphone", data.source],
        [
          "Processes",
          Object.entries(s.components)
            .map(
              ([k, v]) => k + ": " + v.pid + (v.alive ? " running" : " exited"),
            )
            .join(" · "),
        ],
      ];
      setText(this.q("device-details").querySelector("summary"), passive
        ? "MODELS & DEVICE DETAILS · Whisper " + whisper
        : "MODELS, PROMPT & DEVICE DETAILS");
      updateModelDetails(this.q("models"), details);
      setText(this.q("error"), this.operationError || s.error || s.audio_error || "");
      if (s.audio_error || !s.volume) {
        this.q("volume").disabled = true;
        this.q("mute").disabled = true;
      }
      const confirmation = this.confirmation;
      if (confirmation && !s.audio_error) {
        const observed = confirmation.action === "volume" ? s.volume : s.microphone;
        const wanted = confirmation.body;
        const matches = observed &&
          (!("percent" in wanted) || observed.percent === Math.round(wanted.percent)) &&
          (!("mute" in wanted) || (confirmation.action === "volume"
            ? observed.mute === wanted.mute
            : observed.hardware_mute === wanted.mute && observed.mute === wanted.mute));
        if (matches) {
          this.confirmation = null;
          confirmation.resolve();
        }
      }
      this.showBusy();
      for (const e of data.events) this.event(e);
      this.cursor = data.cursor;
    } catch (e) {
      this.connected = false;
      this.ready = false;
      this.loading = false;
      this.clear();
      this.disableControls();
      this.el.classList.add("disconnected");
      setText(this.q("phase"), "Disconnected");
      this.q("phase").dataset.phase = "error";
      setText(this.q("loaded"), "Device status unavailable");
      this.q("loaded").title = "";
      setText(this.q("heard-identity"), "Voice tracking unavailable while disconnected.");
      this.q("heard-face").style.opacity = "0.35";
      if (this.q("heard-sound-tags")) setText(this.q("heard-sound-tags"), "Sound tags unavailable while disconnected.");
      if (this.q("heard-sound-status")) setText(this.q("heard-sound-status"), "Disconnected · no new sound results");
      setText(this.q("expression"), "unknown");
      this.q("portrait").dataset.mood = "neutral";
      setText(this.q("gesture"), "—");
      this.captureStatus = "Hardware capture unavailable";
      this.showBusy();
      setText(this.q("error"), e.message);
    }
    if (config.nodes) showNodes();
    setTimeout(() => this.poll(), this.connected ? 250 : 1500);
  }
}
let sessionPending = "";
function sessionBusy() {
  const busy = !!sessionPending;
  $("begin").disabled = busy || !config.peer || !!sessionActive;
  $("end").disabled = busy;
  $("session-state").classList.toggle("busy", busy || sessionLoading);
  if (busy) setText($("session-state"), sessionPending);
}
let sessionActive = false, sessionLoading = false;
let recordingState = { available: false, phase: "idle" }, recordingPending = false;
let recordingError = "", recordingMutation = 0;
function showRecording() {
  const r = recordingState, button = $("record"), status = $("recording-state");
  button.hidden = !r.available;
  button.disabled = recordingPending || r.phase === "stopping";
  setText(button, ({ armed: "Cancel recording", starting: "Cancel recording",
    recording: "Stop recording", stopping: "Saving recording…" })[r.phase] || "Record");
  button.setAttribute("aria-pressed", String(["armed", "starting", "recording", "stopping"].includes(r.phase)));
  button.classList.toggle("recording", r.phase === "recording");
  status.hidden = !r.available || (!recordingPending && r.phase === "idle" && !r.file);
  status.classList.toggle("busy", recordingPending || ["armed", "starting", "stopping"].includes(r.phase));
  setText(status, recordingPending ? "Updating recording…" :
    r.phase === "armed" ? "Armed · waiting for models to load" :
    r.phase === "starting" ? "Starting recording…" :
    r.phase === "recording" ? "Recording · " + r.file :
    r.phase === "stopping" ? "Saving recording…" :
    r.phase === "error" ? "Recording failed" : r.file ? "Saved · " + r.file : "");
  setText($("recording-error"), recordingError || r.error || "");
}
$("record").onclick = async () => {
  if (recordingPending) return;
  const stop = ["armed", "starting", "recording"].includes(recordingState.phase);
  recordingPending = true; recordingError = ""; recordingMutation++; showRecording();
  try { recordingState = await api("recording", stop ? undefined : {}, stop ? "DELETE" : "POST"); }
  catch (e) { recordingError = e.message; }
  finally { recordingPending = false; recordingMutation++; showRecording(); }
};
if (config.nodes) {
  for (const node of config.nodes) panels[node.id] = new Panel(node.id, node.speaker.role, node);
} else {
  panels.local = new Panel("local", config.role || "teacher");
  if (config.peer) panels.peer = new Panel("peer", (config.role || "teacher").toLowerCase() === "teacher" ? "Student" : "Teacher");
}
function showNodes() {
  const values = Object.values(panels);
  if ($("nodes-state")) setText($("nodes-state"), values.filter(p => p.connected).length + " / " + config.nodes.length + " connected");
}
if ($("stop-all")) $("stop-all").onclick = async () => {
  const button = $("stop-all");
  if (button.disabled) return;
  button.disabled = true;
  setText(button, "Stopping…");
  setText($("nodes-error"), "");
  const ids = Object.keys(panels);
  try {
    const results = await Promise.allSettled(ids.map(id => api(id + "/stop", {})));
    setText($("nodes-error"), results.flatMap((r, i) => r.status === "rejected" ? [ids[i] + ": " + r.reason.message] : []).join("\n"));
  } finally {
    button.disabled = false;
    setText(button, "Stop all");
  }
};
$("begin").onclick = async () => {
  if (sessionPending) return;
  sessionPending = "Starting conversation…"; sessionBusy();
  try {
    await api("conversation", {
      topic: $("topic").value,
      seconds: +$("duration").value,
      turns: +$("turns").value,
      local: panels.local.settings(),
      peer: panels.peer.settings(),
    });
  } catch (e) {
    setText($("session-error"), e.message);
  } finally {
    sessionPending = ""; sessionBusy();
  }
};
$("end").onclick = async () => {
  if (sessionPending) return;
  sessionPending = "Stopping both devices…"; sessionBusy();
  try {
    await api("conversation", undefined, "DELETE");
    recordingState = await api("recording", undefined, "DELETE"); showRecording();
    await Promise.all(
      Object.keys(panels).map((side) => api(side + "/stop", {})),
    );
  } catch (e) {
    setText($("session-error"), e.message);
  } finally {
    sessionPending = ""; sessionBusy();
  }
};
async function sessionPoll() {
  try {
    const recordingVersion = recordingMutation;
    const s = await api("conversation");
    if (!recordingPending && recordingVersion === recordingMutation && s.recording) {
      recordingState = s.recording; showRecording();
    }
    sessionActive = s.active;
    sessionLoading = s.phase === "loading";
    setText($("session-state"), s.phase);
    sessionBusy();
    setText($("session-error"), s.error);
    setText($("session-detail"), s.active
      ? s.turns +
        " input turns · " +
        (s.started
          ? Math.round(Date.now() / 1000 - s.started) + " seconds"
          : "loading models")
      : "Teacher begins. Each device hears the other through its microphone.");
  } catch (e) {
    setText($("session-error"), e.message);
  }
  setTimeout(sessionPoll, 500);
}
if (!config.nodes) sessionPoll();

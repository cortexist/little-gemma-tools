"use strict";
const config = window.VOICE_CONFIG;
const $ = (id) => document.getElementById(id);
async function api(path, body, method) {
  const response = await fetch("/api/" + path, {
    method: method || (body ? "POST" : "GET"),
    headers: { "X-Voice-Key": config.key, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
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
  constructor(side, role) {
    this.role = role;
    this.clockOffset = 0;
    this.side = side;
    this.cursor = 0;
    this.timers = [];
    this.schedules = [];
    this.segments = [];
    this.lines = [];
    this.ready = false;
    this.el = $("device-template").content.firstElementChild.cloneNode(true);
    $("devices").append(this.el);
    this.q = (cls) => this.el.querySelector("." + cls);
    this.q("role").textContent = role.toUpperCase();
    this.q("prompt").value =
      common +
      (role.toLowerCase() === "teacher"
        ? "You are the teacher. While the conversation is continuing, explain the topic accurately, then ask your student one clear question. Encourage useful corrections."
        : "You are the student. While the conversation is continuing, answer your teacher’s question, then ask one brief follow-up question about the topic. Admit uncertainty.") + briefFirst;
    this.mouth("b_m_p");
    this.q("volume").oninput = () =>
      (this.q("volume-value").textContent = this.q("volume").value + "%");
    this.q("volume").onchange = () =>
      this.act("volume", { percent: +this.q("volume").value });
    this.q("mute").onclick = () => this.act("volume", { mute: !this.muted });
    this.q("mic-volume").oninput = () =>
      (this.q("mic-volume-value").textContent = this.q("mic-volume").value + "%");
    this.q("mic-volume").onchange = () =>
      this.act("microphone", { percent: +this.q("mic-volume").value });
    this.q("mic-mute").onclick = () =>
      this.act("microphone", { mute: !this.hardwareMuted });
    this.q("load").onclick = () => this.act("start", this.settings());
    this.q("hello").onclick = () =>
      this.act("prompt", {
        text: "Say hello in one short sentence so I can check your speaker volume.",
      });
    this.q("stop").onclick = () => this.act("stop", {});
    this.poll();
  }
  settings() {
    return {
      profile: this.q("model").value,
      system_prompt: this.q("prompt").value,
      listen: true,
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
      this.q("error").textContent = e.message;
    } finally {
      this.pendingAction = "";
      this.showBusy();
    }
  }
  showBusy() {
    const text = this.pendingAction || (this.loading ? "Loading pipeline…" : "");
    const status = this.q("operation");
    status.hidden = !text;
    status.textContent = text;
    status.classList.toggle("busy", !!text);
    this.el.setAttribute("aria-busy", String(!!text));
    // Disable duplicate requests immediately, even before the next poll.
    if (this.pendingAction) {
      this.el.querySelectorAll("button, input[type=range]").forEach(el => { el.disabled = true; });
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
    this.mouth("b_m_p");
    this.q("phoneme").textContent = "—";
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
              this.q("phoneme").textContent = ph;
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
    if (e.kind === "alignment") {
      const lines = e.text.split("\n");
      const [offset, rate] = lines.shift().split("\t");
      this.schedules.push({ offset: +offset, rate: +rate, rows: lines });
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
      this.q("events").textContent = this.lines.join("\n");
    }
  }
  async poll() {
    try {
      const requestedAt = Date.now();
      const data = await api(this.side + "/status?after=" + this.cursor);
      const s = data.state;
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
        const preferred = this.role.toLowerCase() === "teacher" ? "12b" : "e2b";
        if (data.profiles[preferred]) this.q("model").value = preferred;
        this.q("offset").value = data.sync_offset_ms;
      }
      this.q("name").textContent = s.name;
      this.q("loaded").textContent = s.ready ? (data.profiles[s.profile]?.label || s.profile) + " loaded" : "No model loaded / loading";
      this.q("phase").textContent = s.phase;
      this.q("phase").dataset.phase = s.phase;
      this.ready = s.ready;
      this.loading = s.phase === "loading";
      this.q("stop").disabled = false;
      this.q("volume").disabled = false;
      this.q("mute").disabled = false;
      this.q("hello").disabled =
        !s.ready || ["speaking", "thinking", "hearing"].includes(s.phase);
      this.q("load").disabled = !["stopped", "error"].includes(s.phase);
      if (s.volume) {
        this.muted = s.volume.mute;
        this.muteIcon("mute", this.muted, "speaker");
        if (document.activeElement !== this.q("volume")) {
          this.q("volume").value = s.volume.percent;
          this.q("volume-value").textContent = s.volume.percent + "%";
        }
      }
      const mic = s.microphone;
      this.q("mic-volume").disabled = !mic || !!s.audio_error;
      this.q("mic-mute").disabled = !mic || mic.hardware_mute === null || !!s.audio_error;
      if (mic) {
        this.hardwareMuted = mic.hardware_mute;
        this.muteIcon("mic-mute", mic.hardware_mute, "microphone");
        this.q("mic-mute").setAttribute("aria-pressed", String(!!mic.hardware_mute));
        this.q("mic-state").textContent = s.audio_error ? "Microphone state unavailable" :
          mic.hardware_mute ? "Hardware capture muted" : mic.mute ? "System capture muted" :
          mic.hardware_mute === null ? "Hardware capture mute unavailable" : "Hardware capture enabled";
        if (document.activeElement !== this.q("mic-volume")) {
          this.q("mic-volume").value = mic.percent;
          this.q("mic-volume-value").textContent = mic.percent + "%";
        }
      }
      this.q("heard").textContent =
        s.input || s.last_input || "Waiting for speech.";
      this.q("said").textContent =
        s.reply
          .replace(/<\|channel>.*?<channel\|>/gs, "")
          .replace(/<\|tool_call>.*?(?:<tool_call\|>|\n)/gs, "")
          .replace(/<[^>]*>/g, "")
          .replace(/\[(?:happy|sad|angry|neutral|nod|shake|quiet)\]/g, "") ||
        "No reply yet.";
      this.q("expression").textContent = s.expression;
      this.q("portrait").dataset.mood = s.expression;
      this.q("gesture").textContent = s.gesture;
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
      this.q("alignment").textContent = s.alignment
        ? "Phonemes · estimated playback sync"
        : "No phoneme schedule received";
      const details = [
        [
          "LLM",
          s.ready ? data.profiles[s.profile]?.model : "Not loaded / loading",
        ],
        ["MTP", s.ready ? data.profiles[s.profile]?.head : "—"],
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
      this.q("models").replaceChildren(
        ...details.flatMap(([k, v]) => {
          const dt = document.createElement("dt"),
            dd = document.createElement("dd");
          dt.textContent = k;
          dd.textContent = v ? String(v).replace(/^\/home\/[^/]+\//, "~/") : "—";
          return [dt, dd];
        }),
      );
      this.q("error").textContent = this.operationError || s.error || s.audio_error || "";
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
      this.q("phase").textContent = "Disconnected";
      this.q("error").textContent = e.message;
    }
    setTimeout(() => this.poll(), 150);
  }
}
let sessionPending = "";
function sessionBusy() {
  const busy = !!sessionPending;
  $("begin").disabled = busy || !config.peer || !!sessionActive;
  $("end").disabled = busy;
  $("session-state").classList.toggle("busy", busy || sessionLoading);
  if (busy) $("session-state").textContent = sessionPending;
}
let sessionActive = false, sessionLoading = false;
panels.local = new Panel("local", config.role || "teacher");
if (config.peer) panels.peer = new Panel("peer", "Student");
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
    $("session-error").textContent = e.message;
  } finally {
    sessionPending = ""; sessionBusy();
  }
};
$("end").onclick = async () => {
  if (sessionPending) return;
  sessionPending = "Stopping both devices…"; sessionBusy();
  try {
    await api("conversation", undefined, "DELETE");
    await Promise.all(
      Object.keys(panels).map((side) => api(side + "/stop", {})),
    );
  } catch (e) {
    $("session-error").textContent = e.message;
  } finally {
    sessionPending = ""; sessionBusy();
  }
};
async function sessionPoll() {
  try {
    const s = await api("conversation");
    sessionActive = s.active;
    sessionLoading = s.phase === "loading";
    $("session-state").textContent = s.phase;
    sessionBusy();
    $("session-error").textContent = s.error;
    $("session-detail").textContent = s.active
      ? s.turns +
        " input turns · " +
        (s.started
          ? Math.round(Date.now() / 1000 - s.started) + " seconds"
          : "loading models")
      : "Teacher begins. Each device hears the other through its microphone.";
  } catch (e) {
    $("session-error").textContent = e.message;
  }
  setTimeout(sessionPoll, 500);
}
sessionPoll();

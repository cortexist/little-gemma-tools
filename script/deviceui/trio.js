"use strict";
// Counts and automatic shutdown belong to the host, never this browser tab.
(() => {
  const topic = $("trio-topic"), begin = $("trio-begin"), end = $("trio-end"), record = $("trio-record");
  let state = { active: false, phase: "idle", turns: {}, blockers: ["Checking nodes…"] };
  let rec = { available: false, phase: "idle" }, pending = "", recordPending = false;
  let error = "", pollError = "", recordError = "", mutation = 0, recordMutation = 0, online = false;
  let restoredSession = null;
  const storageKey = "voice-trio-session:" + config.nodes.map(n => n.id).join(",");
  let draft = {};
  try { draft = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) {}
  if (typeof draft.topic === "string") {
    // Earlier classroom subjects often kept a custom suffix after this old prefix.
    const migrated = draft.topic.replace(/^(LLMs|Language models), agents, harnesses and IDEs\b/i,
      (_, models) => models + ", agents, and harnesses");
    topic.value = migrated;
    if (migrated !== draft.topic) {
      try { localStorage.setItem(storageKey, JSON.stringify({ ...draft, topic: migrated })); } catch (_) {}
    }
  }
  const audioMode = $("trio-audio-mode");
  let selectedAudio = draft.audio_mode === "digital" ? "digital" : "room";
  audioMode.onclick = () => {
    if (audioMode.disabled) return;
    selectedAudio = selectedAudio === "digital" ? "room" : "digital";
    save();
  };
  const pipelineMode = $("trio-pipeline-mode");
  let selectedMode = draft.mode === "passive" ? "passive" : "active";
  pipelineMode.onclick = () => {
    if (pipelineMode.disabled) return;
    selectedMode = selectedMode === "passive" ? "active" : "passive";
    save();
  };
  const total = $("trio-total");
  total.value = Number.isInteger(draft.total_turns) && draft.total_turns >= 1 && draft.total_turns <= 48 ? draft.total_turns : 12;
  total.oninput = save;
  topic.oninput = save;
  function save() {
    try {
      localStorage.setItem(storageKey, JSON.stringify({ topic: topic.value,
        total_turns: +total.value, audio_mode: selectedAudio, mode: selectedMode }));
    } catch (_) {}
    show();
  }
  const seconds = value => Number.isFinite(value) ? value.toFixed(2) + "s" : "—";
  function timing(panel, node) {
    const measurement = state.latency || {}, data = measurement.nodes?.[node.id];
    const summary = panel.q("latency-summary"), detail = panel.q("latency-detail");
    if (!data) {
      setText(summary, "Timing · waiting for a session");
      setText(detail, "Session medians appear here. PCM starts at accepted transcript; gap starts at the previous speaker’s estimated PCM end. " +
        (measurement.warnings || []).join("; "));
      return;
    }
    const approximate = data.pcm_kinds?.includes("pipe-write"), prior = measurement.id !== state.id;
    setText(summary, (measurement.reason === "error" ? "Partial · " : prior ? "Last · " : "Median · ") + "PCM" + (approximate ? "≈" : " ") + seconds(data.pcm_s?.median) +
      " · gap≈" + seconds(data.gap_s?.median));
    const lines = [
      (prior ? "Last saved session" : "This session") + " · " + (measurement.profiles?.[node.id] || "") +
        " · " + (data.pcm_s?.n || 0) + "/" + data.observed_turns + " turns with PCM",
      "PCM: accepted transcript → " + (approximate ? "first playback-pipe write (proxy)" : "first PCM ready") +
        ". Latest " + seconds(data.pcm_s?.last) + "; slowest " + seconds(data.pcm_s?.maximum) + ".",
      "Gap: previous speaker’s estimated PCM end → first PCM; includes endpointing and ASR. Latest " +
        seconds(data.gap_s?.last) + "; slowest " + seconds(data.gap_s?.maximum) + "; n=" + (data.gap_s?.n || 0) + ". Not measured acoustic silence.",
      "ASR median " + seconds(data.asr_s?.median) + (data.asr_s ? "; n=" + data.asr_s.n + "." : " (duration not reported by this node)."),
      "Stage medians: input → model token " + seconds(data.model_s?.median) + "; input → clause " + seconds(data.clause_s?.median) +
        "; clause → PCM " + seconds(data.tts_s?.median) + ".",
      ...(data.response_start_s ? [
        "One response: input → first speech text " + seconds(data.response_start_s.median) +
          "; text → permitted PCM " + seconds(data.budget_wait_s?.median) + ". No separate turn-decision request.",
        "CAM++ wait after accepted transcript " + seconds(data.identity_wait_s?.median) + "."
      ] : [
        "Room decision " + seconds(data.decision_s?.median) + "; waiting for the floor " + seconds(data.floor_wait_s?.median) + ".",
        "Within room decision: CAM++ wait " + seconds(data.identity_wait_s?.median) + "; name learning " + seconds(data.name_s?.median) +
          "; turn selection " + seconds(data.floor_decision_s?.median) + "."
      ]),
      "Capture: " + (measurement.active ? "running" : measurement.reason || "interrupted") + ".",
      measurement.source === "replayed" ? "Imported from the recorded session’s event logs." : "Timing is collected by the host while the class runs.",
      measurement.storage === "memory-only" ? "No runtime directory: measurements are not saved." : "Session measurements saved on the dashboard host.",
      ...(measurement.warnings || [])
    ];
    setText(detail, lines.join("\n"));
    summary.title = lines.slice(0, 4).join("\n");
  }
  function show() {
    window.trioSessionActive = state.active;
    const passive = selectedMode === "passive";
    window.trioPassive = passive;
    document.body.classList.toggle("passive", passive);
    pipelineMode.disabled = !online || !!pending || state.active;
    pipelineMode.setAttribute("aria-checked", String(!passive));
    setText($("trio-pipeline-label"), passive ? "Passive" : "Active");
    const valid = topic.value.trim() && total.checkValidity();
    begin.disabled = !online || !!pending || state.active || !valid || !!state.blockers.length;
    begin.title = state.blockers.join("; ");
    end.disabled = pending === "Stopping…" || state.phase === "stopping";
    topic.disabled = state.active || !!pending;
    total.disabled = passive || state.active || !!pending;
    setText($("trio-phase"), pending || ({ idle: "Idle", loading: "Loading…", running: "Running", waiting: "Listening",
      stopping: "Stopping…", stopped: "Stopped", error: "Stopped" })[state.phase] || state.phase);
    $("trio-phase").classList.toggle("busy", !!pending || ["loading", "stopping"].includes(state.phase));
    setText($("trio-detail"), passive ? "Whisper and CAM++ listening · no LLM responses · Stop to finish." : state.reason === "limit" ? "Total turns fulfilled." :
      state.active ? (state.completed || 0) + " / " + state.total_turns + " spoken turns · " +
        (state.phase === "waiting" ? "Listening for more speech." : "Speakers answer as needed.") : "One shared limit; the opening counts.");
    setText($("trio-blockers"), state.active ? "" : state.blockers.join(" · "));
    setText($("trio-error"), pollError || error || state.error || "");
    for (const node of config.nodes) {
      const p = panels[node.id];
      timing(p, node);
      setText(p.q("turn-progress"), state.turns[node.id]?.completed || 0);
      const room = p.state?.room;
      const head = room?.turn_head;
      const headMode = ["shadow", "control"].includes(head?.mode);
      const headLabel = head?.mode === "control" ? "Head control" : "Head shadow";
      setText(p.q("room-decision"), headMode ?
        (head.error ? headLabel + " · error" : headLabel + " · " + (head.action || "ready")) :
        room?.decision?.error ? "Decision unavailable" :
        room?.decision ? room.decision.action + " · " + room.decision.reason : "");
      p.q("room-decision").title = head?.mode === "control" ?
        "The head selects speak, defer or wait. Experimental live policy; scores are not calibrated confidence." : head?.mode === "shadow" ?
        "Observation only; current speech behavior is unchanged. Live prompt differs from head training." : "";
      p.q("turn-budget").title = "Completed spoken turns. All speakers keep listening; only the shared total ends the session.";
      if (state.active) {
        p.q("load").disabled = true;
        p.q("hello").disabled = true;
      }
    }
    const recording = ["armed", "starting", "recording", "stopping"].includes(rec.phase);
    if (recording && rec.audio_mode) selectedAudio = rec.audio_mode;
    audioMode.disabled = !online || !rec.available || recordPending || recording;
    audioMode.setAttribute("aria-checked", String(selectedAudio === "digital"));
    setText($("trio-audio-label"), selectedAudio === "digital" ? "Digital" : "Room");
    record.disabled = !online || !rec.available || recordPending || rec.phase === "stopping";
    setText(record, ({ armed: "Cancel rec", starting: "Cancel rec", recording: "Stop rec", stopping: "Saving…" })[rec.phase] || "Record");
    record.setAttribute("aria-pressed", String(recording));
    record.classList.toggle("recording", rec.phase === "recording");
    record.title = rec.available ? "Arm recording before Start; capture begins when the models are ready." : "Recording is not configured on the dashboard host";
    const rs = $("trio-recording-state");
    rs.hidden = !recordPending && rec.phase === "idle" && !rec.file;
    const modeLabel = rec.audio_mode === "digital" ? "Digital" : "Room";
    setText(rs, recordPending ? "Updating recording…" : rec.phase === "armed" ? modeLabel + " recording armed · waiting for Start and model readiness" :
      rec.phase === "starting" ? "Starting recording…" : rec.phase === "stopping" ? "Saving recording…" :
      rec.phase === "recording" ? modeLabel + " recording · " + rec.file : rec.file ? "Saved · " + rec.file : "");
    setText($("trio-recording-error"), recordError || rec.error || "");
  }
  function accept(next) {
    state = next;
    if (next.active && next.id !== restoredSession) {
      selectedMode = next.mode === "passive" ? "passive" : "active";
      topic.value = next.topic;
      total.value = next.total_turns;
      restoredSession = next.id;
    }
    if (!recordPending && next.recording) rec = next.recording;
  }
  begin.onclick = async () => {
    if (begin.disabled) return;
    save(); error = ""; pending = "Starting…"; mutation++; show();
    try {
      accept(await api("conversation", { mode: selectedMode, topic: topic.value, total_turns: +total.value, nodes: Object.fromEntries(config.nodes.map(n =>
        [n.id, panels[n.id].settings()])) }));
    } catch (e) { error = e.message; }
    finally { mutation++; pending = ""; show(); }
  };
  end.onclick = async () => {
    if (end.disabled) return;
    pending = "Stopping…"; error = ""; mutation++; show();
    try { accept(await api("conversation", undefined, "DELETE")); }
    catch (e) { error = e.message; }
    finally { mutation++; pending = ""; show(); }
  };
  record.onclick = async () => {
    if (record.disabled) return;
    recordPending = true; recordError = ""; recordMutation++; show();
    try { rec = await api("recording", ["armed", "starting", "recording"].includes(rec.phase) ? undefined : { audio_mode: selectedAudio },
      ["armed", "starting", "recording"].includes(rec.phase) ? "DELETE" : "POST"); }
    catch (e) { recordError = e.message; }
    finally { recordPending = false; recordMutation++; show(); }
  };
  async function poll() {
    const version = mutation, recVersion = recordMutation;
    try {
      const next = await api("conversation");
      if (version === mutation && !pending) {
        if (recVersion !== recordMutation) delete next.recording;
        accept(next);
        online = true;
        pollError = "";
      }
    } catch (e) { online = false; pollError = e.message; }
    show();
    setTimeout(poll, 750);
  }
  show(); poll();
})();

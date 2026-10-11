"""Opt-in room pipeline: incremental listening, silent decisions, granted speech.

All semantic input comes from this node's microphone. The hub sends floor epochs,
never another node's transcript, voice label, or model reply. Classic voicecat
and manual Windows pipelines retain their existing behavior.
"""
import json
import os
from pathlib import Path
import queue
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from reply_stream import ReplyStream
from room_policy import ROOM_POLICY, aliases, decision_prompt, parse_decision, name_prompt, local_context, needs_name
from session_replay import AcousticEncoder
from session_scenario import ClassroomScenario
from session_speakers import SpeakerSession
from speech_names import whisper_name_hint
from room_identity import ParallelIdentity, LiveUtterance
from windows_audio import WindowsDeviceMixin, read_exact, select_output
from windows_capture import CapturePCM, select_input


class SocketClient:
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.stdin = self.socket.makefile('wb', buffering=0)
        self.stdout = self.socket.makefile('rb')

    def close(self):
        try: self.socket.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.stdin.close()
        self.stdout.close()
        self.socket.close()


class LinuxOutput:
    """Persistent far-field playback pipe with an explicit estimated drain."""
    def __init__(self, runtime, rate):
        self.runtime, self.rate = runtime, rate
        self.latency = runtime.cfg.get('sync_offset_ms', 100)/1000

    def start(self): pass
    def close(self): pass
    def abort(self): pass  # Owned service is terminated during cancellation.

    def write(self, stereo):
        import numpy as np
        mono = np.frombuffer(stereo, '<i2')[::2].tobytes()
        self.runtime.player.stdin.write(mono)
        self.runtime.player.stdin.flush()
        return False

    def stop(self):
        while time.time() < self.runtime.state['audible_until']:
            if self.runtime.cancel.wait(.02): break


class RoomRuntime:
    policy = ROOM_POLICY
    persistent_player = True
    capture = WindowsDeviceMixin.capture
    commit_input = WindowsDeviceMixin.commit_input
    read_synthesis = WindowsDeviceMixin.read_synthesis
    synthesis_config = WindowsDeviceMixin.synthesis_config
    stream_reply = WindowsDeviceMixin.stream_reply
    spoken_turn = WindowsDeviceMixin.turn
    terminate_owned = WindowsDeviceMixin.terminate_owned

    def __init__(self, device, settings):
        self.device = device
        self.cfg, self.state = device.cfg, device.state
        self.lock, self.cancel, self.procs = device.lock, device.cancel, device.procs
        self.session = settings['session']
        self.passive = settings.get('mode', 'active') == 'passive'
        from recording_input import InputTap
        self.asr_input = InputTap(self.session)
        self.topic = settings['topic']
        self.epoch = 0
        self.granted = set()
        self.capture_epoch = 0
        self.endpoint_until = 0
        self.input_prefix = ''
        self.turn_active = False
        self.prompts = queue.Queue(maxsize=1)
        self.audio = bytearray()
        self.live = None
        self.identity_worker = None
        self.heard = 0
        self.last_observation = None
        self.last_spoken_epoch = 0
        self.last_spoken_text = ''
        self.context_turns = 0
        self.conversation = []
        self.output = None
        self.client = None
        self.room_phase = 'loading'
        self.asr_name_hint = ''
        self.memory_trial = None
        memory = self.cfg.get('room_memory', {})
        self.context_renew_turns = memory.get('renew_after', 8)
        if (type(self.context_renew_turns) is not int or self.context_renew_turns not in (4, 8)
                or (self.context_renew_turns == 4 and memory.get('mode') != 'clm_trial')):
            raise ValueError('Four-turn renewal is limited to the explicit memory trial')
        if not self.passive and memory.get('mode', 'off') != 'off':
            from room_memory import RoomMemory
            self.memory_trial = RoomMemory(memory, self.session, self.speaker['speaker_id'],
                                           self.cancel, self.emit)

    def remember(self, entry):
        self.conversation.append(entry)
        self.conversation = self.conversation[-8:]
        if self.memory_trial:
            self.memory_trial.observe(entry, local_context(self.identities))

    def __getattr__(self, name):
        return getattr(self.device, name)

    def prepare_identity(self):
        pass

    def refresh_asr_names(self):
        """Offer only names this listener already knows; never grant identity by ASR."""
        names = [self.speaker['display_name']]
        names += [track['name'] for track in self.identities.tracks.values() if track['name']]
        names += list(self.identities.pending_names)
        hint = whisper_name_hint(names)
        with self.lock:
            self.asr_name_hint = hint
        self.emit('room_asr_hint', hint, names=names)

    def prepare_engine_observer(self, profile, env):
        pass

    def close_engine_observer(self):
        pass

    def reply_decoder(self):
        return None

    def publish(self, **extra):
        with self.lock:
            room = dict(self.state.get('room', {}))
            room.update(session=self.session, epoch=self.epoch, phase=self.room_phase,
                        heard_count=self.heard, spoken_count=self.state['synth_count'])
            room.update(extra)
            self.state['room'] = room
            if 'live_identity' in extra:
                live = extra['live_identity']
                self.state['speaker_identity'] = dict(live, mode='room',
                    speaker_id=live.get('track_id'), display_name=live.get('name'),
                    attribution='estimate', overlap_checked=False)
                self.state['speaker_identity_at'] = time.time()

    def reset_capture_audio(self):
        self.audio = bytearray()
        with self.lock:
            if self.live and not self.live.closed: self.live.cancelled = True
            self.live = None

    def record_input_block(self, captured, pcm, eligible):
        self.asr_input.feed(captured, pcm, eligible)

    def capture_audio(self, utterance, pcm):
        if utterance.active:
            if self.live is None:
                token = self.capture_epoch
                def publish(identity):
                    if token != self.capture_epoch or self.cancel.is_set(): return
                    old = self.state.get('room', {}).get('live_identity', {})
                    identity.update(at=time.time(), epoch=self.epoch)
                    self.publish(live_identity=identity)
                    if (identity['status'], identity.get('track_id')) != (old.get('status'), old.get('track_id')):
                        self.emit('room_identity_live', identity.get('name') or '', identity=identity)
                if self.identity_worker is None: raise RuntimeError('Parallel identification is unavailable')
                self.live = LiveUtterance(self.identity_worker, self.identities, self.lock,
                                          publish, self.cfg.get('vad_level', 200))
            self.live.feed(pcm, self.state['capture_rms'])
            self.audio.extend(pcm if self.audio else utterance.audio)
            # Bound identity work independently of the ASR rolling window.
            if len(self.audio) > 32000*8: del self.audio[32000*8:]

    def admit(self, text, tail, asr=None):
        if self.turn_active or self.input_prefix+tail != text:
            raise ValueError('Invalid room input boundary')
        if self.live is None: raise ValueError('Current utterance has no diarisation')
        self.live.seal()
        self.turn_active = True
        self.capture_epoch += 1
        self.input_prefix = ''
        self.room_phase = 'deciding'
        self.state.update(last_input=text, input='', phase='thinking')
        self.prompts.put_nowait(('observe', text, tail, dict(asr or {}, admitted_at=time.time()), bytes(self.audio), self.epoch, self.live))
        self.publish()

    def control(self, action, data):
        with self.lock:
            if data.get('session') != self.session:
                raise ValueError('Stale room session')
            epoch = data.get('epoch')
            if type(epoch) is not int or epoch < 1:
                raise ValueError('Invalid floor epoch')
            if action == 'room-idle':
                if epoch != self.epoch or self.turn_active:
                    raise ValueError('Room observation is unfinished')
                if not self.state['listening']:
                    self.capture_epoch += 1
                    self.state.update(listening=True, phase='listening')
                self.room_phase = 'listening'
                self.publish()
            elif action == 'room-prepare':
                if epoch == self.epoch: return {'ok': True}
                if (epoch != self.epoch+1 or self.turn_active or self.input_prefix
                        or self.room_phase not in ('listening', 'waiting')):
                    raise ValueError('Listener has unfinished input or a stale floor epoch')
                self.epoch = epoch
                self.capture_epoch += 1
                self.endpoint_until = time.monotonic()+1.5
                self.state.update(listening=not bool(data.get('speaking')), phase='listening')
                self.room_phase = 'listening'
                self.publish(decision=None, live_identity=dict(status='waiting', track_id=None, epoch=epoch))
            elif action == 'room-grant':
                if epoch in self.granted: return {'ok': True}
                if (epoch != self.epoch or self.turn_active or self.input_prefix
                        or self.state['listening'] or not self.state['ready']):
                    raise ValueError('Floor was not prepared for this speaker')
                self.granted.add(epoch)
                self.turn_active = True
                self.capture_epoch += 1
                self.state['input_count'] += 1
                self.room_phase = 'speaking'
                self.state['phase'] = 'thinking'
                self.prompts.put_nowait(('speak', bool(data.get('opening'))))
                self.publish()
            else:
                raise ValueError('Unknown room action')
        return {'ok': True}

    def hold(self, ms):
        with self.lock: self.endpoint_until = time.monotonic()+ms/1000

    def decision(self, job, replies):
        _, text, tail, asr, pcm, epoch, live = job
        # ASR/prefill have run concurrently. Never ask for a turn decision until
        # the acoustic worker has attributed this exact completed utterance.
        identity_started = time.monotonic()
        diarisation = live.finish(self.cancel)
        identity_wait_s = time.monotonic()-identity_started
        self.heard += 1
        turn_id = 'e'+str(epoch)+'-h'+str(self.heard)
        duration = len(pcm)/32000
        embedding = diarisation['embedding']
        # Controlled floor excludes other node playback, but does not detect
        # human overlap. That limitation is explicit in every identity event.
        with self.lock:
            observation = self.identities.observe(turn_id, embedding, text,
                                clean=embedding is not None, duration=duration)
        if (observation['track_id'] not in diarisation['tracks'] and not
                (diarisation.get('status') == 'new_voice' and observation['status'] == 'new_voice')):
            observation['track_id'] = None
            observation['status'] = 'uncertain'
            self.identities.turns[turn_id].update(observation)
        if self.cfg.get('room_identity', {}).get('record_evidence') and pcm:
            import wave
            with wave.open(str(self.directory/(turn_id+'.wav')), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(pcm)
        self.last_observation = observation
        self.last_heard_at = (asr or {}).get('admitted_at', time.time())
        self.last_asr = asr or {}
        self.publish(identity=self.identities.snapshot())
        self.emit('heard', text, epoch=epoch, observation=observation,
                  diarisation={k:v for k,v in diarisation.items() if k != 'embedding'},
                  identity_wait_s=identity_wait_s,
                  overlap_check='controlled-node-floor; human overlap unverified', **(asr or {}))
        if not observation['track_id']:
            raise RuntimeError('Current utterance has no reliable single-speaker attribution; turn decision withheld')
        self.remember(dict(epoch=epoch, voice=observation['track_id'], heard=text))
        learning = None
        name_started = time.monotonic()
        if needs_name(self.identities, observation):
            instruction = name_prompt(self.identities, observation)
            raw_name = self.silent(tail+instruction, replies)
            tail = ''
            if self.cancel.is_set(): return
            learning = self.identities.accept(raw_name if raw_name == 'NONE' else '<speaker_name>'+raw_name+'</speaker_name>', metadata_only=True)
            if learning['rejected']:
                self.emit('room_identity_rejected', raw_name, learning=learning, epoch=epoch)
                raw_name = self.silent('Correct the private name proposal: '+
                    '; '.join(learning['rejected'])+'. '+instruction, replies)
                learning = self.identities.accept(raw_name if raw_name == 'NONE' else '<speaker_name>'+raw_name+'</speaker_name>', metadata_only=True)
            self.emit('room_identity', raw_name, learning=learning, epoch=epoch)
            if learning and learning['accepted']:
                self.refresh_asr_names()
        name_s = time.monotonic()-name_started
        floor_started = time.monotonic()
        instruction = decision_prompt(self.identities, observation, self.speaker, self.state['synth_count'],
            conversation=dict(current_epoch=epoch, own_last_spoken_epoch=self.last_spoken_epoch,
                              own_last_spoken_text=self.last_spoken_text))
        raw_decision = self.silent(tail+instruction, replies)
        if self.cancel.is_set(): return
        known = [t['name'] for t in self.identities.tracks.values() if t['name']]
        known += list(self.identities.pending_names)
        for attempt in range(2):
            try:
                decision = parse_decision(raw_decision, text, aliases(self.speaker), known)
                break
            except (ValueError, TypeError) as exc:
                self.emit('room_decision_invalid', raw_decision, error=str(exc), attempt=attempt+1)
                if attempt:
                    decision = dict(action='defer', reason='defer', error=str(exc))
                    break
                raw_decision = self.silent('Your last output was a PRIVATE invalid decision and was NOT spoken. '
                                          'Correct the format. '+instruction, replies)
        if self.cancel.is_set(): return
        previous_was_me = self.last_spoken_epoch == epoch-1 and self.last_spoken_epoch > 0
        if decision.get('action') == 'speak' and not decision.get('explicit_address'):
            if decision['reason'] in ('addressed', 'followup'):
                decision['reason'] = 'followup' if previous_was_me else 'answer'
        decision.update(epoch=epoch, at=time.time(), heard=turn_id, previous_was_me=previous_was_me)
        self.last_decision_at = decision['at']
        self.last_decision_timing = dict(identity_wait_s=identity_wait_s, name_s=name_s,
                                        floor_decision_s=time.monotonic()-floor_started)
        self.emit('room_decision', raw_decision, decision=decision, learning=learning,
                  **self.last_decision_timing)
        self.renew_context()
        with self.lock:
            self.turn_active = False
            self.capture_epoch += 1
            self.room_phase = 'listening'
            self.state.update(phase='listening', last_input=text)
            self.publish(decision=decision, identity=self.identities.snapshot())

    def silent(self, prompt, replies):
        record_wire = self.cfg.get('room_identity', {}).get('record_evidence')
        if record_wire:
            with (self.directory/'silent-wire.log').open('ab') as wire: wire.write(b'\n-- silent reply --\n')
        self.client.stdin.write((prompt.replace('\n', ' ')+'\n').encode())
        self.client.stdin.flush()
        result = ReplyStream()
        progress = time.monotonic()
        wire_bytes = 0
        while not result.done and not self.cancel.is_set():
            try: part = replies.get(timeout=.2)
            except queue.Empty:
                if time.monotonic()-progress > 90: raise TimeoutError('Silent model output stalled')
                continue
            if part is None: raise RuntimeError('Engine disconnected during name metadata')
            wire_bytes += len(part)
            if wire_bytes > 8192: raise RuntimeError('Silent metadata exceeded its bounded response size')
            if record_wire:
                with (self.directory/'silent-wire.log').open('ab') as wire: wire.write(part)
            result.feed(part)
            if len(result.text) > 2048: raise RuntimeError('Silent metadata exceeded its bounded response size')
            self.state['model_bytes'] += len(part)
            progress = time.monotonic()
        self.context_turns += 1
        return result.text.strip()

    def speak(self, opening, replies, env):
        spoken = self.state['synth_count']
        instruction = ('ROOM_SPEAK. You now have the floor. spoken_turns='+str(spoken)+'. ')
        if opening:
            instruction += ('Begin the class about '+self.topic+'. Introduce your invited guest and invite student questions. '
                            'Then explicitly address the guest and ask them to introduce themself. ')
        else:
            instruction += 'Respond to the last heard utterance as yourself. '
        instruction += ('This is your first spoken turn; introduce yourself briefly if appropriate. ' if spoken == 0
                        else 'You have already spoken. Do not repeat your introduction or "My name is", unless asked your name or spelling. ')
        if spoken == 0 and self.speaker['role'] == 'student':
            instruction += 'After your introduction, ask the guest one short question about the subject. '
        instruction += 'Use one or two short spoken sentences, at most 45 words. No JSON or control metadata.'
        heard = self.last_observation['raw_transcript'] if self.last_observation else 'Opening the class about '+self.topic
        timing = dict(self.last_asr, heard_at=self.last_heard_at, decision_at=self.last_decision_at) if self.last_observation else None
        if timing is not None: timing.update(self.last_decision_timing)
        self.spoken_turn((heard, instruction, timing), self.client, replies, self.output, env)
        if self.cancel.is_set(): return
        self.last_spoken_epoch = self.epoch
        self.last_spoken_text = self.state['reply']
        self.context_turns += 1
        self.remember(dict(epoch=self.epoch, own_speech=self.state['reply']))
        if self.last_observation:
            self.identities.ask_spelling(self.last_observation['track_id'], self.state['reply'])
        self.renew_context()
        self.room_phase = 'waiting'
        self.publish(identity=self.identities.snapshot())

    def connect(self):
        """One reader belongs to one connection, including across renewal."""
        if sys.platform == 'win32':
            client = self.spawn('client', [self.cfg['engine_client'], self.engine_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.engine_env)
        else:
            client = SocketClient(self.engine_path)
        replies = queue.Queue()
        decoder = self.reply_decoder()
        def reader():
            try:
                while not self.cancel.is_set():
                    part = client.stdout.read1(4096)
                    if not part: break
                    for text in decoder.feed(part) if decoder else (part,): replies.put(text)
                if decoder: decoder.finish()
            except Exception as exc:
                self.emit('engine_stream_error', str(exc))
            finally: replies.put(None)
        threading.Thread(target=reader, daemon=True).start()
        self.client = self.input_client = client
        self.replies = replies

    def renew_context(self):
        # Metadata also consumes engine context. Renew only after a completed
        # response, before enabling capture or advertising a ready floor request.
        if self.context_turns < self.context_renew_turns or self.cancel.is_set(): return
        with self.lock:
            if self.input_prefix: raise RuntimeError('Cannot renew an open input turn')
            active = self.turn_active
            self.turn_active = True
            self.room_phase = 'renewing'
            self.state['phase'] = 'thinking'
            self.publish()
        began = time.monotonic()
        previous = self.client
        if isinstance(previous, SocketClient): previous.close()
        else:
            with self.lock: self.procs.pop('client', None)
            previous.stdin.close()
            previous.wait(timeout=10)
            previous.stdout.close()
        self.connect()
        self.context_turns = 0
        memory = dict(identity=local_context(self.identities), topic=self.topic[:500],
                      spoken_turns=self.state['synth_count'], own_last_spoken_text=self.last_spoken_text[:800],
                      conversation=[dict(entry) for entry in self.conversation[-6:]])
        trial = self.memory_trial.retained() if self.memory_trial else None
        if trial:
            # Identity, topic, spoken count and last-own speech above remain
            # application-owned. An edit can replace only older narrative data.
            memory.update(trial)
        # Bounded excerpts preserve local acoustic/name memory without overflowing
        # the engine's conservative character-based input admission limit.
        for entry in memory['conversation']:
            for key in ('heard', 'own_speech'):
                if key in entry: entry[key] = entry[key][:600]
        prefix = ('ROOM_CONTEXT '+json.dumps(memory, separators=(',', ':'))+
                  ' END_ROOM_CONTEXT. This was retained history, not new speech. Current input follows: ')
        if trial:
            prefix = prefix.replace('<', '＜').replace('>', '＞')
            self.emit('room_memory_applied', covered=trial['memory_covered_event'],
                      retained_context=memory, prefix_chars=len(prefix))
        # Prefill retained context without closing the new input turn or asking
        # the model to generate an acknowledgement. The next ASR stream (or
        # floor grant) follows it on the same ordered binary connection.
        for offset in range(0, len(prefix), 512):
            data = prefix[offset:offset+512].encode('utf-8')
            self.client.stdin.write(b'\x01T'+struct.pack('<HHI', 0, 0, len(data))+data)
        self.client.stdin.flush()
        self.emit('room_context_renewed', elapsed=time.monotonic()-began,
                  spoken_turns=self.state['synth_count'], heard=self.heard, prefix_chars=len(prefix))
        with self.lock: self.turn_active = active

    def open_output(self, output, rate):
        if sys.platform != 'win32': return LinuxOutput(self, rate)
        import sounddevice as sd
        return sd.RawOutputStream(device=output, samplerate=rate, channels=2, dtype='int16',
                                  extra_settings=sd.WasapiSettings(auto_convert=True))

    def linux_capture(self, blocks, errors):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(self.ff_socket)
                sock.sendall(struct.pack('<BBHHI', 1, ord('e'), 0, 0, 0))
                with sock.makefile('rb') as stream:
                    pending = bytearray()
                    while not self.cancel.is_set():
                        header = read_exact(stream, 10)
                        if not header: raise RuntimeError('Audio service disconnected')
                        magic, kind, _, _, size = struct.unpack('<BBHHI', header)
                        if magic != 1 or size > 1048576: raise ValueError('Invalid microphone frame')
                        payload = read_exact(stream, size)
                        if kind != ord('P'): continue
                        pending.extend(payload)
                        while len(pending) >= 640:
                            pcm = bytes(pending[:640]); del pending[:640]
                            self.state['capture_blocks'] += 1
                            blocks.put_nowait((time.monotonic(), self.capture_epoch, pcm, ''))
        except Exception as exc:
            if not self.cancel.is_set(): errors.put(exc)

    def work(self, profile, prompt, listen=False, turn_limit=0):
        from server import engine_command
        capture_stream = capture_thread = None
        try:
            self.device.directory = Path(tempfile.mkdtemp(prefix='r-', dir=self.root))
            with self.lock:
                self.procs.clear()
                self.state.update(phase='loading', ready=False, error='', input='', reply='', history=[],
                    input_count=0, reply_count=0, synth_count=0, synth_bytes=0, played_bytes=0,
                    model_bytes=0, reply_spoken=False, alignment=False, audible_until=0, listening=False,
                    finishing=False, turn_limit=0, profile=profile, started=time.time(),
                    capture_rms=0, capture_peak_rms=0, capture_blocks=0, input_commits=0, room={},
                    last_sound_event=None, last_sound_error=None, last_sound_at=None,
                    pipeline_mode='passive' if self.passive else 'active')
            self.publish(decision=None, live_identity=dict(status='waiting', track_id=None))
            self.emit('reset')
            identity = self.cfg['room_identity']
            self.encoder = AcousticEncoder(identity['model'], identity['sha256'])
            scenario = ClassroomScenario()
            known = scenario.priors(self.speaker['speaker_id'], lambda key: self.encoder.wav(identity['priors'][key])[0])
            self.identities = SpeakerSession(known, canonicalize=False)
            self.prepare_identity()
            self.refresh_asr_names()
            with (self.directory/'speaker-id.log').open('ab') as log:
                self.identity_worker = ParallelIdentity(identity['model'], identity['sha256'], log)
            self.procs['speaker-id'] = self.identity_worker.proc
            env = {k:v for k,v in os.environ.items() if not k.startswith('LG_')}
            env.update(self.cfg.get('env', {}))
            if not self.passive:
                selected = self.cfg['profiles'][profile]
                env.update(selected.get('env', {}))
                system = self.directory/'system.txt'
                prompt = prompt.replace("Introduce yourself explicitly with 'My name is Chey' before asking the guest a question.",
                                        "Introduce yourself with 'My name is Chey' only on your first spoken turn, unless asked again.")
                system.write_text(prompt+' '+self.policy, encoding='utf-8')
                self.prepare_engine_observer(profile, env)
                path = str(self.directory/'lg.sock')
                if len(path.encode()) >= 108: raise ValueError('Runtime socket path is too long')
            blocks, errors = queue.Queue(maxsize=150), queue.Queue()
            if sys.platform == 'win32':
                import sounddevice as sd
                if not self.passive:
                    self.output = select_output(sd, self.cfg['sink'])
                    sd.check_output_settings(device=self.output, channels=2, dtype='int16',
                        samplerate=self.cfg.get('sample_rate', 22050), extra_settings=sd.WasapiSettings(auto_convert=True))
                channels = self.cfg.get('capture_channels', 2)
                source = select_input(sd, self.cfg['source'], self.cfg.get('capture_hostapi', 'Windows WASAPI'), channels)
                rate = int(self.cfg.get('capture_rate', sd.query_devices(source)['default_samplerate']))
                converter = CapturePCM(rate, channels)
                def microphone(data, frames, timing, status):
                    self.state['capture_blocks'] += 1
                    try: blocks.put_nowait((time.monotonic(), self.capture_epoch, converter.convert(data), status))
                    except queue.Full: errors.put(RuntimeError('Microphone queue overflow; refusing to drop speech'))
                capture_stream = sd.RawInputStream(device=source, samplerate=rate, channels=channels,
                    dtype='int16', blocksize=rate//50, callback=microphone)
            else:
                self.ff_socket = str(self.directory/'ff.sock')
                self.restore_microphone_mute()
                self.spawn('audio', [self.cfg['far_field'], '-s', self.ff_socket, '--source', self.cfg['source'],
                    '--sink', self.cfg['sink'], '--channels', str(self.cfg.get('channels', 6)),
                    '--use-channel', '0', '--gain-db', '0', '--scene', '--no-aec'])
            if not self.passive:
                self.spawn('engine', engine_command(self.cfg, selected, system, path), env=env)
            self.spawn('whisper', self.cfg['whisper_command'], env=env)
            if not self.passive:
                self.synth = self.spawn('piper', self.synthesis_command()+['--mux-boundaries'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
                self.frames = queue.Queue(maxsize=64)
                self.synth_sent = self.synth_completed = 0
                synth_reader = threading.Thread(target=self.read_synthesis, daemon=True); synth_reader.start()
            deadline = time.monotonic()+210
            while not self.cancel.wait(.2):
                self.device.check_processes('Startup', include_optional=True)
                healthy = False
                try:
                    with urllib.request.urlopen(self.cfg['whisper_health'], timeout=1) as response:
                        healthy = response.status == 200
                except OSError: pass
                if healthy and (self.passive or 'listening on ' in (self.directory/'engine.log').read_text(errors='replace')):
                    if sys.platform == 'win32' or Path(self.ff_socket).is_socket(): break
                if time.monotonic() > deadline: raise TimeoutError('Room pipeline load timed out')
            if self.cancel.is_set(): return
            if not self.passive:
                self.engine_path, self.engine_env = path, env
                self.connect()
                if sys.platform != 'win32' and self.persistent_player:
                    self.player = self.spawn('player', [self.cfg['far_field'], '--speak', self.ff_socket,
                        '--rate', str(self.cfg.get('sample_rate', 22050))], stdin=subprocess.PIPE)
                frame = self.frames.get(timeout=90)
                if isinstance(frame, Exception): raise frame
                if not frame or frame[0] != b'C': raise RuntimeError('Missing Piper configuration')
                self.speech_rate = self.synthesis_config(frame[1])
            with self.lock:
                self.room_phase = 'waiting'
                self.state.update(ready=True, phase='paused')
                self.publish(identity=self.identities.snapshot())
            capture_thread = threading.Thread(target=self.capture, args=(blocks, errors), daemon=True)
            capture_thread.start()
            if capture_stream: capture_stream.start()
            else: threading.Thread(target=self.linux_capture, args=(blocks, errors), daemon=True).start()
            self.emit('ready', 'Passive listening ready' if self.passive else 'Room pipeline ready')
            while not self.cancel.is_set():
                if not errors.empty(): raise errors.get_nowait()
                if self.identity_worker.error: raise RuntimeError(self.identity_worker.error)
                self.device.check_processes('Room pipeline', include_optional=True)
                try: job = self.prompts.get(timeout=.1)
                except queue.Empty: continue
                if job[0] == 'observe': self.decision(job, None if self.passive else self.replies)
                elif self.passive: raise ValueError('Passive mode does not generate responses')
                else: self.speak(job[1], self.replies, env)
        except Exception as exc:
            if not self.cancel.is_set():
                self.state.update(error=str(exc), phase='error')
                self.emit('error', str(exc))
        finally:
            self.cancel.set()
            self.asr_input.close(self.state.get('error', ''))
            self.close_engine_observer()
            if capture_stream:
                capture_stream.abort(); capture_stream.close()
            self.terminate_owned()
            if isinstance(self.client, SocketClient): self.client.close()
            if capture_thread: capture_thread.join(timeout=35)
            if self.identity_worker: self.identity_worker.close()
            self.turn_active = False
            self.room_phase = 'stopped'
            self.state.update(ready=False, listening=False, audible_until=0,
                              phase='error' if self.state.get('error') else 'stopped')
            self.publish()
            self.emit('stopped')

"""One generation per acoustic input: defer or stream speech, with duplex capture.

The hub supplies budget permits only. It never supplies identities, transcripts,
addressees or a successor. CAM++ runs in the existing independent CPU worker.
"""
import json
import queue
import re
import select
import socket
import struct
import sys
import threading
import time

from reply_stream import ReplyStream
from decision_timing import TimedHiddenReplyDecoder, decision_stamp, compare_decision_timing
from room_identity import LiveUtterance
from room_hold import WindowsWordOutput, word_ends
from room_policy import DIRECT_ROOM_POLICY, DIRECT_ROOM_TAG_POLICY, response_prompt, accept_response_names, needs_response_name
from room_runtime import RoomRuntime
from session_speakers import unit
from speech_names import piper_names
from speech_titles import normalize_titles


class SelfAwareVoices:
    """A self-echo competitor; never adds self or a room roster to learned tracks."""
    SELF = 'self-output'

    def __init__(self, session, embedding):
        self.session, self.embedding = session, unit(embedding)
        self.match_threshold = session.match_threshold

    def preview(self, embedding):
        vector = unit(embedding)
        own = sum(a*b for a, b in zip(vector, self.embedding))
        other = self.session.preview(vector)
        best = other['similarity'] if other['similarity'] is not None else -1
        if own >= self.match_threshold and own-best >= self.session.margin:
            return dict(status='matched', track_id=self.SELF, name=None, similarity=own)
        if other['status'] == 'matched' and best-own >= self.session.margin:
            return other
        return dict(status='unknown' if max(own, best) < self.session.new_threshold else 'uncertain',
                    track_id=None, name=None, similarity=max(own, best))


class DuplexOutput:
    """A per-response Linux mouth. Disconnect flushes PCM without stopping capture."""
    def __init__(self, runtime, rate):
        self.runtime, self.rate = runtime, rate
        self.latency = runtime.cfg.get('sync_offset_ms', 100)/1000
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(2)
        self.socket.connect(runtime.ff_socket)
        self.frame(b'm', ('rate='+str(rate)+'\n').encode())
        self.lock = threading.Lock()
        self.boundaries = []
        self.hold_requested = False

    def frame(self, kind, data=b''):
        self.socket.sendall(struct.pack('<BBHHI', 1, kind[0], 0, 0, len(data))+data)

    def start(self): pass

    def write(self, stereo):
        import numpy as np
        with self.lock:
            self.frame(b'P', np.frombuffer(stereo, '<i2')[::2].tobytes())
        return False

    def add_boundaries(self, marks):
        with self.lock:
            self.boundaries.extend(round(mark*16000/self.rate) for mark in marks)
            self.boundaries = sorted(set(self.boundaries))[-256:]
            if self.hold_requested: self.send_hold()

    def send_hold(self):
        marks = struct.pack('<'+'Q'*len(self.boundaries), *self.boundaries)
        if self.runtime.hold_grace_ms:
            # G anchors the grace at the service's render cursor even when
            # alignment has not arrived. Further G frames only add boundaries.
            self.frame(b'G', struct.pack('<I', round(self.runtime.hold_grace_ms*16))+marks)
        elif self.boundaries:
            self.frame(b'H', marks)

    def hold(self):
        with self.lock:
            self.hold_requested = True
            self.send_hold()

    def resume(self):
        with self.lock:
            if self.hold_requested:
                self.hold_requested = False
                self.frame(b'R')

    def stop(self):
        # Keep the control connection readable by the service while waiting.
        # Its D command blocks the service's reader and cannot be interrupted by
        # another command. The finite PCM-tail estimate allows immediate abort.
        while self.hold_requested:
            if self.runtime.cancel.wait(.02) or self.runtime.yielded.is_set(): return
        while time.time() < self.runtime.state['audible_until']:
            if self.runtime.cancel.wait(.02) or self.runtime.yielded.is_set(): return
        self.frame(b'D')
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if self.runtime.cancel.is_set() or self.runtime.yielded.is_set(): return
            if select.select([self.socket], [], [], .02)[0]:
                if self.socket.recv(1) != b'\x01': raise RuntimeError('Audio drain failed')
                return
        raise TimeoutError('Audio drain acknowledgement timed out')

    def abort(self):
        self.close()

    def close(self):
        try: self.socket.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.socket.close()


class DirectRoomRuntime(RoomRuntime):
    policy = DIRECT_ROOM_POLICY
    persistent_player = False

    def __init__(self, device, settings):
        super().__init__(device, settings)
        head_settings = self.cfg.get('turn_head', {})
        self.head_control = head_settings.get('mode') == 'control'
        self.text_decision_tags = head_settings.get('text_decision_tags', False)
        self.prompt_lane = head_settings.get('prompt_lane', 'legacy')
        if self.prompt_lane not in ('legacy', 'v1_wait_assurance', 'v2_sound_event'):
            raise ValueError('Unknown turn-head prompt lane')
        if not isinstance(self.text_decision_tags, bool):
            raise ValueError('text_decision_tags must be a boolean')
        if self.text_decision_tags and head_settings.get('mode') not in ('shadow', 'control'):
            raise ValueError('Decision tags require shadow or control mode')
        if self.head_control and not self.text_decision_tags:
            raise ValueError('Head control requires decision-first tags')
        if self.text_decision_tags: self.policy = DIRECT_ROOM_TAG_POLICY
        self.sound_input = self.prompt_lane == 'v2_sound_event'
        if self.sound_input:
            if (not self.text_decision_tags
                    or self.cfg.get('acoustic_input', {}).get('mode') != 'whisper_at_v2'):
                raise ValueError('v2 prompt lane requires decision tags and acoustic adapter')
            from sound_event import tag_policy
            self.policy = tag_policy(self.policy)
        self.prompts = queue.Queue(maxsize=4)
        self.engine_busy = False
        self.capture_sent = ''
        self.response_serial = 0
        self.responses = {}
        self.current = None
        self.yielded = threading.Event()
        self.playing = False
        self.opened = False
        self.generation_open = False
        self.request_count = 0
        self.shadow = None
        self.shadow_connection = 0
        self.shadow_status = dict(mode='passive' if self.passive else 'off', controlling=False)
        self.introduction_reminder = self.cfg.get('room_memory', {}).get('introduction_reminder', 'legacy')
        if self.introduction_reminder not in ('legacy', 'spoken_evidence'):
            raise ValueError('Unknown introduction reminder mode')
        # Session-local receipts survive engine renewal, scoped to the voice
        # being answered. Another guest must not inherit an earlier receipt.
        self.introduced_to = {}
        hold = head_settings.get('barge_hold', {})
        if not isinstance(hold, dict): raise ValueError('barge_hold must be an object')
        self.hold_threshold = float(hold.get('speak_below', 0))
        if not 0 <= self.hold_threshold <= 1: raise ValueError('Invalid barge-hold speak threshold')
        self.hold_grace_ms = hold.get('grace_ms', 0)
        if (type(self.hold_grace_ms) not in (int, float)
                or not 0 <= self.hold_grace_ms <= 2000):
            raise ValueError('Barge-hold grace_ms must be a number from 0 to 2000')
        if self.hold_threshold and not self.head_control:
            raise ValueError('Barge hold requires a controlling decision head')
        self.active_stream = None
        self.hold_requested = False
        self.hold_segment = 0

    def publish(self, **extra):
        super().publish(protocol='direct-v1', responses=self.responses,
                        generation_count=self.request_count, turn_head=self.shadow_status, **extra)

    def prepare_engine_observer(self, profile, env):
        # No observer flag may leak in from a profile without a matching binary
        # frame reader. Both shadow and control require explicit configuration.
        env.pop('LG_SNAPSHOT_HIDDEN', None)
        env.pop('LG_TURN_CONTROL', None)
        settings = self.cfg.get('turn_head', {})
        if settings.get('mode', 'off') == 'off': return
        if settings.get('mode') not in ('shadow', 'control'): raise ValueError('Unknown turn-head mode')
        backend = settings.get('backend', 'numpy')
        native = settings.get('models', {}).get(profile, {})
        package = native.get('path') if backend == 'c' else settings.get('packages', {}).get(profile)
        if not package:
            if self.head_control: raise ValueError('Head control requires a package for this model profile')
            self.shadow_status = dict(mode='unavailable', controlling=False, error='No head for this model profile')
            self.emit('turn_head_unavailable', profile)
            return
        from turn_head import ShadowHead, sha256
        contract = {}
        if self.sound_input:
            from sound_event import INPUT_FORMAT, CAPTURE_BOUNDARY
            contract = dict(input_format=INPUT_FORMAT, capture_boundary=CAPTURE_BOUNDARY)
        self.shadow = ShadowHead(package, self.cfg['profiles'][profile]['model'],
            self.directory/'turn-head', self.shadow_result, self.shadow_error,
            save_hidden=settings.get('save_hidden', True),
            decision=self.apply_head_decision if self.head_control else None,
            backend=backend, library=settings.get('library'), expected_sha256=native.get('sha256'), **contract)
        provenance = dict(mode=settings['mode'], controlling=self.head_control, profile=profile,
            selection='argmax', silent_wait='await further acoustic input; no timed retry',
            text_decision_source='head_forced' if self.head_control else 'model',
            text_decision_format='decision_tags_first' if self.text_decision_tags else 'native',
            contract_status='live_prompt_unvalidated', head_contract=self.shadow.head.contract,
            runtime=self.shadow.runtime,
            live_system_sha256=sha256(self.directory/'system.txt'),
            engine_sha256=sha256(self.cfg['engine']), mtp_enabled=bool(self.cfg['profiles'][profile].get('head')),
            mtp_depth=env.get('LG_MTP_N'), inference_device='cpu',
            note='Current conversational prompt and history differ from training; no accuracy/calibration claim.')
        (self.directory/'turn-head'/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf8')
        env['LG_SNAPSHOT_HIDDEN'] = '1'
        if self.head_control: env['LG_TURN_CONTROL'] = '1'
        self.shadow_status = dict(mode=settings['mode'], controlling=self.head_control, status='ready',
                                  contract_status='live_prompt_unvalidated', runtime=self.shadow.runtime)
        self.emit('turn_head_ready', 'Experimental head control' if self.head_control else 'Shadow only', **provenance)

    def close_engine_observer(self):
        if self.shadow: self.shadow.close()

    def reply_decoder(self):
        if self.shadow is None: return None
        if self.head_control and 'turn control: same-turn prefix protocol v1' not in (self.directory/'engine.log').read_text(errors='replace'):
            raise ValueError('Deployed engine does not support head control')
        self.shadow_connection += 1
        connection = self.shadow_connection
        def snapshot(frame):
            with self.lock:
                record = self.current
                if record is None or record.get('head_snapshot'):
                    self.shadow_error('Unexpected or duplicate hidden snapshot', {})
                    return
                record['head_snapshot'] = True
                context = dict(session=self.session, connection=connection, response=record['id'],
                    heard=record['heard'], observation_epoch=self.epoch,
                    opening=record.get('opening', False), generation_count=self.request_count)
            self.shadow.submit(frame, context)
        return TimedHiddenReplyDecoder(snapshot)

    def shadow_error(self, message, context):
        if self.cancel.is_set(): return
        with self.lock:
            self.shadow_status = dict(mode='control' if self.head_control else 'shadow',
                                      controlling=self.head_control, status='error', error=message)
            self.emit('turn_head_error', message, context=context)
            if self.head_control:
                self.state['error'] = 'Head control failed: '+message
                self.yielded.set()
                self.cancel.set()
            self.publish()

    def apply_head_decision(self, result, context):
        """Select the assistant prefix once, before logging or publishing it."""
        if self.cancel.is_set(): return
        with self.lock:
            record = self.responses.get(context.get('response'))
            if (context.get('session') != self.session or record is not self.current
                    or record is None or context.get('connection') != self.shadow_connection
                    or not self.generation_open or record.get('control_action')):
                raise ValueError('Stale or duplicate controlling head result')
            if not result['answer_channel_matches']:
                raise ValueError('Controlling head snapshot boundary differs from training')
            explicit_open = self.sound_input and record.get('opening', False)
            action = 'speak' if explicit_open else result['action']
            selection = 'explicit_room_open' if explicit_open else 'argmax'
            record['control_source'] = selection
            record['control_action'] = action
            record['speak_score'] = None if explicit_open else result.get('probabilities', {}).get('speak')
            message = struct.pack('<BIBB', 3, result['snapshot']['turn'],
                                  {'speak': 1, 'defer': 2, 'wait': 3}[action], int(record['needs_name']))
            self.client.stdin.write(message); self.client.stdin.flush()
            record.update(control_sent_at=time.time(), control_sent_monotonic=time.monotonic())
            self.emit('turn_head_control', action, response=record['id'], heard=record['heard'],
                      action=action, predicted_action=result['action'], needs_name=record['needs_name'], selection=selection,
                      ready_to_control_s=record['control_sent_monotonic']-result['ready_monotonic'],
                      control_sent_monotonic=record['control_sent_monotonic'], controlling=True)

    def shadow_result(self, result, context):
        if self.cancel.is_set() or context.get('session') != self.session: return
        with self.lock:
            record = self.responses.get(context['response'])
            if record is None: return
            record['head'] = result
            self.emit('turn_head_decision', result['action'], response=record['id'], heard=record['heard'], **result)
            if int(record['id']) == self.response_serial:
                self.shadow_status = dict(result, status='ready', response=record['id'])
            self.compare_shadow(record)
            self.compare_text_timing(record)
            self.publish()

    def compare_text_timing(self, record):
        head, text = record.get('head'), record.get('text_decision')
        if not head or not text or record.get('timing_compared'): return
        record['timing_compared'] = True
        timing = compare_decision_timing(head, text, record['request_sent_monotonic'])
        timing.update(controlling=self.head_control,
                      independent_text_decision=not self.head_control)
        record['decision_timing'] = timing
        self.emit('turn_decision_timing', response=record['id'], heard=record['heard'],
                  opening=record.get('opening', False), text_decision_tags=self.text_decision_tags, **timing)

    def compare_shadow(self, record):
        # Agreement is not correctness. A head wait has no direct counterpart
        # in the existing speech/[defer] output contract. Opening is synthetic.
        if not record.get('head') or 'text_action' not in record or record.get('head_compared'): return
        record['head_compared'] = True
        self.emit('turn_head_comparison', response=record['id'], heard=record['heard'],
            head_action=record['head']['action'], text_action=record['text_action'],
            agrees=None if self.head_control else record['head']['action'] == record['text_action'],
            opening=record.get('opening', False), controlling=self.head_control,
            independent_text_decision=not self.head_control,
            applied_action=record.get('control_action'), control_source=record.get('control_source'),
            contract_status='live_prompt_unvalidated')

    def prepare_identity(self):
        reference = self.cfg['room_identity'].get('self_reference')
        if not reference: raise ValueError('Duplex room capture requires a self-voice reference')
        self.voices = SelfAwareVoices(self.identities, self.encoder.wav(reference)[0])

    def capture_enabled(self):
        return self.state['listening']

    def capture_audio(self, utterance, pcm):
        if not utterance.active: return
        if self.live is None:
            token = self.capture_epoch
            def publish(identity):
                if token != self.capture_epoch or self.cancel.is_set(): return
                identity.update(at=time.time(), epoch=token)
                old = self.state.get('room', {}).get('live_identity', {})
                self.publish(live_identity=identity)
                if (identity['status'], identity.get('track_id')) != (old.get('status'), old.get('track_id')):
                    self.emit('room_identity_live', identity.get('name') or '', identity=identity)
                if (self.playing and identity['status'] == 'identified'
                        and identity.get('track_id') in self.identities.tracks):
                    self.barge(identity)
                elif self.playing and identity['status'] == 'identifying':
                    self.request_early_hold()
                elif identity['status'] == 'quiet' and self.hold_requested:
                    self.release_early_hold('quiet_without_identification')
            self.live = LiveUtterance(self.identity_worker, self.voices, self.lock, publish,
                                      self.cfg.get('vad_level', 200))
        self.live.feed(pcm, self.state['capture_rms'])
        self.audio.extend(pcm if self.audio else utterance.audio)
        del self.audio[32000*8:]

    def send_prefix(self, piece):
        for offset in range(0, len(piece), 512):
            data = piece[offset:offset+512].encode('utf-8')
            self.client.stdin.write(b'\x01T'+struct.pack('<HHI', 0, 0, len(data))+data)
        self.client.stdin.flush()
        self.capture_sent += piece
        self.state['input_commits'] += 1
        self.emit('input_commit', piece, bytes=len(piece.encode()), prefix_chars=len(self.capture_sent))

    def commit_input(self, piece):
        # Capture keeps running during a generation/playback. Its committed
        # words wait in order until the previous engine turn has closed.
        if not piece: return
        self.input_prefix += piece
        self.state['input'] = self.input_prefix
        self.emit('input', piece)
        if not self.passive and not self.engine_busy and self.prompts.empty(): self.send_prefix(piece)

    def admit(self, text, tail, asr=None):
        if self.input_prefix+tail != text: raise ValueError('Invalid room input boundary')
        if self.live is None: raise ValueError('Current utterance has no acoustic evidence')
        self.live.seal()
        job = ('observe', text, text[len(self.capture_sent):],
               dict(asr or {}, admitted_at=time.time()), bytes(self.audio), self.capture_epoch, self.live)
        try: self.prompts.put_nowait(job)
        except queue.Full:
            raise RuntimeError('Room input queue filled; refusing to discard heard speech') from None
        self.engine_busy = True
        self.turn_active = True
        if not self.playing: self.room_phase = 'responding'
        self.capture_epoch += 1
        self.input_prefix = self.capture_sent = ''
        self.state.update(last_input=text, input='', capture_active=False,
                          last_sound_event=job[3].get('sound_event'),
                          last_sound_error=job[3].get('sound_error'),
                          last_sound_at=job[3]['admitted_at'])
        self.publish()

    def control(self, action, data):
        with self.lock:
            if data.get('session') != self.session: raise ValueError('Stale room session')
            if action == 'room-listen':
                self.state.update(listening=True, phase='listening')
                self.room_phase = 'listening'
            elif action == 'room-open':
                if self.passive: raise ValueError('Passive listeners cannot open a conversation')
                if self.opened: return {'ok': True}
                if self.speaker['role'] != 'teacher': raise ValueError('Only the teacher opens the class')
                if self.engine_busy or self.input_prefix: raise ValueError('Opening would interrupt an input')
                self.opened = True
                self.engine_busy = True
                self.prompts.put_nowait(('speak', True))
            elif action == 'room-permit':
                record = self.responses.get(data.get('response'))
                if record is None or type(data.get('allowed')) is not bool:
                    raise ValueError('Invalid budget permit')
                if record['permission'] is not None:
                    if record['permission'] != data['allowed']: raise ValueError('Conflicting budget permit')
                    return {'ok': True}
                record['permission'] = data['allowed']
                record['permit_at'] = time.time()
            else:
                raise ValueError('Unknown direct room action')
            self.publish()
        return {'ok': True}

    def barge(self, identity):
        # Called with the device lock held. No microphone reset: the interrupting
        # utterance, including its beginning, stays in the existing ASR buffer.
        if self.yielded.is_set() or not self.playing: return
        self.yielded.set()
        held_at = getattr(self.active_stream, 'held_at', None)
        if not isinstance(held_at, (int, float)): held_at = None
        self.hold_requested = False
        self.state.update(expression='neutral', gesture='quiet', gesture_until=0)
        self.emit('cut')  # Cancel pending mouth/gesture timers with the audio.
        if self.current: self.current['interrupted_by'] = identity['track_id']
        # Model output is already short and usually complete before interruption.
        # If it is still generating, cut at the engine's next token boundary.
        if self.generation_open:
            self.client.stdin.write(b'\x02')
            self.client.stdin.flush()
        self.emit('room_barge', identity.get('name') or identity['track_id'], identity=identity,
                  response=self.current['id'] if self.current else None,
                  held_at=held_at, held_s=time.time()-held_at if held_at else None)
        self.publish()

    def request_early_hold(self):
        if (not self.hold_threshold or self.hold_requested or not self.current
                or self.live is None or self.hold_segment == self.live.segment
                or self.current.get('control_action') != 'speak'
                or not isinstance(self.current.get('speak_score'), (int, float))
                or self.current['speak_score'] >= self.hold_threshold):
            return
        self.hold_segment = self.live.segment
        self.hold_requested = True
        if self.active_stream: self.active_stream.hold()
        self.emit('room_hold_request', response=self.current['id'],
                  speak_score=self.current['speak_score'], threshold=self.hold_threshold,
                  segment=self.hold_segment, grace_ms=self.hold_grace_ms)

    def release_early_hold(self, reason):
        self.hold_requested = False
        held_at = getattr(self.active_stream, 'held_at', None)
        if not isinstance(held_at, (int, float)): held_at = None
        if self.active_stream: self.active_stream.resume()
        self.emit('room_hold_release', reason, response=self.current['id'] if self.current else None,
                  held_at=held_at, held_s=time.time()-held_at if held_at else None)

    def open_output(self, output, rate):
        if sys.platform != 'win32': return DuplexOutput(self, rate)
        return WindowsWordOutput(super().open_output(output, rate), self.yielded,
                                 rate=rate, grace_ms=self.hold_grace_ms)

    def reset_context(self):
        # Every in-flight capture prefix here is buffered, never sent into the
        # previous response. Keep it while renewing only between generations.
        with self.lock:
            if self.capture_sent: raise RuntimeError('Cannot reset an open capture prefix')
            prefix, self.input_prefix = self.input_prefix, ''
            try:
                self.context_turns = max(self.context_turns, self.context_renew_turns)
                super().renew_context()
            finally: self.input_prefix = prefix

    def release_engine(self):
        with self.lock:
            if not self.passive and self.context_turns >= self.context_renew_turns: self.reset_context()
            self.engine_busy = not self.prompts.empty()
            if not self.passive and not self.engine_busy and self.input_prefix:
                self.send_prefix(self.input_prefix[len(self.capture_sent):])
            self.turn_active = self.engine_busy
            self.room_phase = 'responding' if self.engine_busy else 'listening'
            self.state.update(phase='hearing' if self.state.get('capture_active') else 'listening')
            self.publish(identity=self.identities.snapshot())

    def decision(self, job, replies):
        _, text, tail, asr, pcm, epoch, live = job
        self.turn_active = True
        self.room_phase = 'responding'
        began = time.monotonic()
        diarisation = live.finish(self.cancel)
        identity_wait = time.monotonic()-began
        if diarisation['tracks'] == [SelfAwareVoices.SELF]:
            self.emit('room_echo', 'Own output ignored', identity_wait_s=identity_wait)
            if not self.passive and tail != text: self.reset_context()
            self.release_engine()
            return
        if not text and self.sound_input:
            # No ASR words means no speaker binding, name update, conversational
            # epoch, partner change or answered question, regardless of CAM++.
            from sound_event import standalone_prompt
            self.emit('room_sound', sound_event=asr['sound_event'], identity_wait_s=identity_wait,
                      sound_timings=asr.get('sound_timings'))
            if self.passive:
                self.release_engine()
                return
            suffix = standalone_prompt(asr['sound_event'], self.state['synth_count'],
                                       bool(self.conversation and 'own_speech' in self.conversation[-1]))
            self.respond('', suffix, asr, None, identity_wait)
            self.release_engine()
            return
        self.heard += 1
        self.epoch = epoch
        turn_id = 'h'+str(self.heard)
        with self.lock:
            observation = self.identities.observe(turn_id, diarisation['embedding'], text,
                clean=diarisation['embedding'] is not None, duration=len(pcm)/32000)
            if (observation['track_id'] not in diarisation['tracks'] and not
                    (diarisation['status'] == 'new_voice' and observation['status'] == 'new_voice')):
                observation.update(track_id=None, status='uncertain')
                self.identities.turns[turn_id].update(observation)
        if self.cfg['room_identity'].get('record_evidence'):
            import wave
            with wave.open(str(self.directory/(turn_id+'.wav')), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed')); wav.writeframes(pcm)
        self.emit('heard', text, observation=observation, identity_wait_s=identity_wait,
                  diarisation={k:v for k,v in diarisation.items() if k != 'embedding'},
                  overlap_check='self competitor and sequential voice changes; simultaneous mixture unverified', **asr)
        if self.passive:
            self.last_observation = observation
            self.emit('room_passive_heard', text, observation=observation)
            self.release_engine()
            return
        if not observation['track_id']:
            self.emit('room_withheld', 'No reliable single-speaker attribution; waiting for clean speech')
            # Discard an attributed-input failure without a fabricated decision
            # or an extra generation. A sent prefix requires a fresh connection.
            if tail != text: self.reset_context()
            self.release_engine()
            return
        previous = bool(self.conversation and 'own_speech' in self.conversation[-1])
        self.remember(dict(epoch=epoch, voice=observation['track_id'], heard=text))
        self.last_observation = observation
        introduced = self.introduction_confirmed(observation)
        if self.introduction_reminder == 'spoken_evidence' and self.speaker['role'] == 'student' and self.state['synth_count']:
            self.emit('room_introduction_reminder', 'omitted' if introduced else 'retained',
                      voice=observation['track_id'], evidence=self.introduced_to.get(observation['track_id']))
        suffix = response_prompt(self.identities, observation, self.speaker, self.state['synth_count'],
            dict(previous_partner_is_me=previous, current_epoch=epoch,
                 own_last_spoken_epoch=self.last_spoken_epoch, own_last_spoken_text=self.last_spoken_text,
                 introduction_confirmed=introduced),
            decision_tags=self.text_decision_tags,
            wait_assurance=self.prompt_lane in ('v1_wait_assurance', 'v2_sound_event'))
        if self.sound_input:
            from sound_event import speech_suffix
            suffix = speech_suffix(suffix, asr.get('sound_event'))
        self.respond(text, tail+suffix, asr, observation, identity_wait)
        self.release_engine()

    def introduction_confirmed(self, observation):
        return bool(self.introduction_reminder == 'spoken_evidence' and observation
                    and observation.get('track_id') in self.introduced_to)

    def remember_introduction(self, text, observation, record):
        # Called only after output drain. A permitted draft or partly played
        # sentence is insufficient evidence that the introduction was spoken.
        if (self.introduction_reminder != 'spoken_evidence' or not record.get('spoken')
                or record.get('status') != 'spoken' or self.cancel.is_set()
                or self.yielded.is_set() or not observation or not observation.get('track_id')
                or self.speaker['role'] != 'student'):
            return
        voice = observation['track_id']
        if voice in self.introduced_to: return
        # Conservative literal self-introduction in generated own speech.
        # Unrecognized wording stays unconfirmed; never parse heard ASR,
        # another person's name, a quoted example, or a question as completion.
        pattern = (r"\A\s*(?:(?:hello|hi|hey)[,.!]\s*)?"
                   r"(?:my name is|i am|i['’]m|call me)\s+"
                   +re.escape(self.speaker['display_name'])+r"(?=\s*[,.;!]|\s*\Z)")
        match = re.match(pattern, text, re.I)
        if not match: return
        evidence = dict(text=match.group().strip(), response=record['id'],
                        spoken_turn=self.state['synth_count'], heard=observation['turn_id'])
        self.introduced_to[voice] = evidence
        self.emit('room_introduction_confirmed', voice=voice, evidence=evidence)

    def speak(self, opening, replies, env):
        if not opening: raise ValueError('Only the opening has a synthetic input')
        text = 'Opening the class about '+self.topic
        prompt = ('ROOM_OPEN. Begin the class about '+self.topic+'. Introduce your invited guest, '
                  'direct student questions to the guest, then explicitly ask the guest to introduce themself. '
                  'At most 45 words. No name tag for this opening.')
        if self.text_decision_tags: prompt += ' FIRST output [speak], then your spoken introduction.'
        self.turn_active = True
        self.respond(text, prompt, dict(admitted_at=time.time()), None, 0, opening=True)
        self.release_engine()

    def respond(self, text, prompt, asr, observation, identity_wait, *, opening=False):
        if self.passive: raise ValueError('Passive mode does not generate responses')
        if opening and (not self.opened or self.speaker['role'] != 'teacher' or observation is not None):
            raise ValueError('Synthetic opening requires the explicit teacher room-open grant')
        self.response_serial += 1
        record = dict(id=str(self.response_serial), permission=None, status='generating', spoken=False,
                      done=False, at=time.time(), opening=opening, heard=observation['turn_id'] if observation else None,
                      needs_name=bool(observation and needs_response_name(self.identities, observation, self.speaker)))
        self.current = record
        self.responses[record['id']] = record
        self.yielded.clear()
        self.hold_requested = False
        self.hold_segment = 0
        self.state.update(last_input=text, reply='', model_bytes=0, reply_spoken=False, phase='thinking')
        self.room_phase = 'responding'
        self.publish()
        current_chunk = None
        def decided(action, source, metadata_before):
            # Capture immediately at recognition, before event logging or TTS.
            if self.head_control:
                expected = 'speak' if record.get('control_action') == 'speak' else 'defer'
                if (not record.get('control_action') or source != 'explicit_tag'
                        or metadata_before or action != expected):
                    raise ValueError('Engine reply does not match the controlling head decision')
                source = 'head_forced'
            record['text_decision'] = decision_stamp(action, source, metadata_before, current_chunk)
        parser = ReplyStream(room=True, decision_tags=self.text_decision_tags, on_decision=decided)
        def publish_decision():
            with self.lock:
                if not record.get('text_decision') or record.get('text_decision_emitted'): return
                record['text_decision_emitted'] = True
                self.emit('turn_text_decision', response=record['id'], heard=record['heard'],
                          text_decision_tags=self.text_decision_tags, **record['text_decision'])
                self.compare_text_timing(record)
        finished, errors = threading.Event(), queue.Queue()
        stamps, clauses = {}, []
        synthesis_clauses = queue.Queue()
        def produce():
            nonlocal current_chunk
            try:
                with self.lock:
                    self.generation_open = True
                    self.request_count += 1
                    record.update(request_sent_at=time.time(), request_sent_monotonic=time.monotonic())
                    self.client.stdin.write((prompt.replace('\n', ' ')+'\n').encode())
                    self.client.stdin.flush()
                    self.emit('room_generation', response=record['id'], heard=record['heard'],
                              request_sent_at=record['request_sent_at'],
                              request_sent_monotonic=record['request_sent_monotonic'])
                    if self.shadow:
                        self.emit('turn_head_input', response=record['id'], heard=record['heard'],
                            opening=record['opening'],
                            connection=self.shadow_connection, transcript=text, suffix=prompt,
                            observation=observation, listener_identity=self.identities.snapshot(),
                            conversation=[dict(entry) for entry in self.conversation],
                            snapshot_scope='actual live generation prompt; no separate decision request')
                progress = time.monotonic()
                while not parser.done and not self.cancel.is_set():
                    try: part = self.replies.get(timeout=.1)
                    except queue.Empty:
                        if time.monotonic()-progress > 90: raise TimeoutError('No model output for 90 seconds')
                        continue
                    if part is None: raise RuntimeError('Engine disconnected during room response')
                    current_chunk = part
                    if 'token' not in stamps:
                        part = part.lstrip(b' \t\r\n')
                        if not part: continue
                        stamps['token'] = time.time()
                    progress = time.monotonic()
                    if self.cfg['room_identity'].get('record_evidence'):
                        with (self.directory/'direct-wire.log').open('ab') as wire: wire.write(part)
                    try:
                        try: ready = parser.feed(part)
                        finally: publish_decision()
                    except ValueError as exc:
                        # An acoustic interruption may cut a private suffix.
                        # Never pass its partial contents to speech or learning.
                        if not self.yielded.is_set():
                            self.emit('room_protocol_invalid', str(exc), response=record['id'])
                            record['protocol_error'] = str(exc)
                            self.yielded.set()
                        parser.metadata.clear()
                        while b'<turn|>' not in part and b'<eos>' not in part:
                            part = self.replies.get(timeout=90)
                            if part is None: raise RuntimeError('Engine disconnected after barge-in')
                        break
                    with self.lock:
                        self.state.update(reply=parser.text.strip(), model_bytes=parser.size)
                        if (parser.text.strip() and record['status'] == 'generating'
                                and (not self.head_control or record.get('control_action') == 'speak')):
                            record.update(status='candidate', at=time.time())
                            self.publish()
                    for clause in ready:
                        visuals = parser.clause_visuals.popleft()
                        stamps.setdefault('clause', time.time())
                        clauses.append(clause)
                        if (not self.yielded.is_set() and record['permission'] is not False
                                and (not self.head_control or record.get('control_action') == 'speak')):
                            rendered = piper_names(normalize_titles(clause), self.cfg.get('name_pronunciations', {}))
                            synthesis_clauses.put((clause, time.time(), visuals))
                            self.synth_sent += 1
                            self.synth.stdin.write((rendered+'\n').encode()); self.synth.stdin.flush()
                with self.lock:
                    tag = record.get('text_decision', {})
                    self.emit('turn_text_decision_outcome', response=record['id'], heard=record['heard'],
                        text_decision_tags=self.text_decision_tags, source=tag.get('source', 'missing'),
                        action=tag.get('action'), metadata_before=tag.get('metadata_before'),
                        parser_done=parser.done, protocol_error=record.get('protocol_error'),
                        interrupted=self.yielded.is_set(), cancelled=self.cancel.is_set(),
                        explicit_tag_complete=tag.get('source') in ('explicit_tag', 'head_forced'),
                        decision_first=tag.get('source') in ('explicit_tag', 'head_forced') and tag['metadata_before'] == 0,
                        controlling=self.head_control, applied_action=record.get('control_action'))
                    self.generation_open = False
                    self.context_turns += 1
                    learning = accept_response_names(self.identities, observation, parser.metadata, self.speaker)
                    self.emit('room_identity', learning=learning, proposals=parser.metadata, response=record['id'])
                    if learning['accepted']:
                        self.refresh_asr_names()
                    if learning['rejected']: self.emit('room_identity_rejected', learning=learning)
                    self.publish(identity=self.identities.snapshot())
            except Exception as exc:
                errors.put(exc)
            finally:
                with self.lock: self.generation_open = False
                finished.set()
        producer = threading.Thread(target=produce, name='room-response', daemon=True)
        producer.start()
        stream = None
        rate = self.speech_rate
        alignment, synthesis_clause = None, None
        progress = time.monotonic()
        try:
            while not self.cancel.is_set():
                if not errors.empty(): raise errors.get_nowait()
                if self.yielded.is_set() and stream is not None:
                    stream.abort(); stream.close(); stream = None
                    self.playing = False
                    self.state['audible_until'] = 0
                if finished.is_set() and self.synth_completed == self.synth_sent: break
                try: frame = self.frames.get(timeout=.05)
                except queue.Empty:
                    if time.monotonic()-progress > 90 and self.synth_sent != self.synth_completed:
                        raise TimeoutError('Speech synthesis stalled')
                    continue
                if isinstance(frame, Exception): raise frame
                if frame is None: raise RuntimeError('Piper exited during room response')
                kind, payload = frame
                progress = time.monotonic()
                if synthesis_clause is None:
                    synthesis_clause = synthesis_clauses.get_nowait()
                if kind == b'D':
                    if int(payload) != self.synth_completed+1: raise RuntimeError('Out-of-order Piper boundary')
                    self.synth_completed = int(payload)
                    alignment, synthesis_clause = None, None
                elif kind == b'A':
                    # Hold metadata until this PCM is permitted for playback.
                    # Discarded synthesis must never animate a silent responder.
                    alignment = payload.decode()
                elif kind == b'P':
                    if len(payload) % 2: raise ValueError('Misaligned Piper audio')
                    if not payload: continue
                    pcm_at = time.time()
                    while record['permission'] is None and not self.cancel.wait(.01):
                        if time.time()-record['at'] > 15: raise TimeoutError('Dashboard budget permit timed out')
                    if self.cancel.is_set(): break
                    if (record['permission'] is False or self.yielded.is_set()
                            or (self.head_control and record.get('control_action') != 'speak')): continue
                    if stream is None:
                        self.emit('input_done', text, heard_at=asr['admitted_at'], decision_at=record['at'],
                                  identity_wait_s=identity_wait, direct_response=True,
                                  **{k:v for k,v in asr.items() if k != 'admitted_at'})
                        if 'token' in stamps: self.emit('model_first_token', at=stamps['token'])
                        self.emit('speech_first_pcm', at=pcm_at)
                        stream = self.open_output(self.output, rate); stream.start()
                        stream.base_samples = self.state['played_bytes']//2
                        self.active_stream = stream
                        with self.lock:
                            self.playing = True
                            record.update(spoken=True, status='speaking')
                            self.state['input_count'] += 1
                            self.publish()
                    # Release non-spoken cues only with this clause's permitted
                    # PCM. Discarded/deferred/interrupted replies stay quiet.
                    with self.lock:
                        for kind, cue in synthesis_clause[2]:
                            self.state[kind] = cue
                            if kind == 'gesture':
                                self.state['gesture_until'] = time.time()+1.95 if cue in ('nod', 'shake') else 0
                            self.emit(kind, cue)
                        synthesis_clause[2].clear()
                    if alignment is not None:
                        stream.add_boundaries(word_ends(alignment,
                            self.state['played_bytes']//2-stream.base_samples, rate))
                        if self.hold_requested: stream.hold()
                        self.emit('clause', synthesis_clause[0], at=synthesis_clause[1])
                        self.state['alignment'] = True
                        self.emit('alignment', f'{self.state["played_bytes"]}\t{rate}\n'+alignment)
                        alignment = None
                    # Bound blocking WASAPI writes so cancellation can clear its
                    # queue promptly. Linux sends the same bounded PCM chunks.
                    import numpy as np
                    sent = 0
                    frame_start = None
                    frame_position = self.state['played_bytes']
                    for offset in range(0, len(payload), 2*(rate//50)):
                        if self.yielded.is_set() or self.cancel.is_set(): break
                        chunk = payload[offset:offset+2*(rate//50)]
                        start = max(time.time()+stream.latency, self.state['audible_until'])
                        if frame_start is None: frame_start = start
                        underflow = stream.write(np.repeat(np.frombuffer(chunk, '<i2'), 2).tobytes())
                        written = getattr(stream, 'last_written', None)
                        if not isinstance(written, int): written = len(chunk)//2
                        played = written*2
                        with self.lock:
                            self.state['synth_bytes'] += played
                            self.state['played_bytes'] += played
                            self.state.update(phase='speaking', reply_spoken=True,
                                              audible_until=start+written/rate)
                        sent += played
                        if self.yielded.is_set(): break
                        # Publish the first 20 ms immediately, then at most
                        # 100 ms per update. Waiting for a whole blocking Piper
                        # frame made its phonemes expire before the UI saw them.
                        if offset == 0 or sent >= rate//10*2 or offset+len(chunk) == len(payload):
                            self.emit('play', f'{frame_start}\t{frame_position}\t{sent}\t{rate}', underflow=bool(underflow))
                            frame_position += sent
                            frame_start, sent = None, 0
                else: raise ValueError('Unexpected Piper frame')
            if not errors.empty(): raise errors.get_nowait()
            if stream is not None:
                if self.cancel.is_set() or self.yielded.is_set(): stream.abort()
                else: stream.stop()
        finally:
            self.playing = False
            self.active_stream = None
            if stream is not None: stream.close()
            producer.join(timeout=2)
        if self.cancel.is_set(): return
        with self.lock:
            record['text_action'] = ('invalid' if record.get('protocol_error') else
                'defer' if parser.deferred else 'speak' if parser.text.strip() else 'empty')
            if self.shadow:
                if not record.get('head_snapshot'):
                    self.shadow_error('Engine returned no hidden snapshot; check deployed binary',
                                      dict(response=record['id']))
                self.compare_shadow(record)
            record.update(done=True, status='invalid' if record.get('protocol_error') else 'interrupted' if self.yielded.is_set() else
                          'spoken' if record['spoken'] else 'waiting' if record.get('control_action') == 'wait' else
                          'deferred' if parser.deferred else 'budget-denied')
            if record['spoken']:
                self.state['reply_count'] += 1
                self.state['synth_count'] += 1
                self.remember_introduction(parser.text.strip(), observation, record)
                self.last_spoken_text = '' if self.yielded.is_set() else parser.text.strip()
                self.last_spoken_epoch = self.epoch
                self.remember(dict(own_speech=self.last_spoken_text or '[Interrupted; remainder unheard]',
                                   interrupted=self.yielded.is_set()))
                if observation and not self.yielded.is_set():
                    self.identities.ask_spelling(observation['track_id'], parser.text.strip())
                self.state['history'] = (self.state['history']+[dict(heard=text, said=parser.text.strip())])[-4:]
                self.emit('reply', parser.text.strip())
                self.emit('reply_done')
                self.emit('synth_done', str(self.state['synth_bytes']), interrupted=self.yielded.is_set())
            self.state['audible_until'] = 0
            self.emit('room_response_done', response=record['id'], status=record['status'],
                      spoken=record['spoken'], generation_count=self.request_count,
                      applied_action=record.get('control_action'), controlling=self.head_control)
            self.current = None
            self.publish()

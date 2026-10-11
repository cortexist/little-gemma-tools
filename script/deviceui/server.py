#!/usr/bin/env python3
"""Device-owned voice pipeline monitor. Configuration is local, never browser shell input."""
import argparse
from collections import deque
import json
import math
import os
import re
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recording import Recording
from speaker_collections import load_collection, identity_prompt
sys.path.insert(0, str(HERE.parent))
from voicedemo import load_visemes, PHONEME_TO_VISEME


def run(args):
    return subprocess.check_output(args, text=True, timeout=5)


def decision_head_details(cfg):
    """Configured assets only; loaded runtime comes from room.turn_head.runtime."""
    settings = cfg.get('turn_head', {})
    backend = settings.get('backend', 'numpy')
    mode = settings.get('mode', 'off')
    result = {}
    for profile in cfg['profiles']:
        package = settings.get('packages', {}).get(profile)
        model = (settings.get('models', {}).get(profile, {}).get('path') if backend == 'c'
                 else str(Path(package)/'head.npz') if package else None)
        result[profile] = dict(model=model, backend=backend, mode=mode)
    return result


def stop_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    pairs = [tuple(map(int, line.split())) for line in run(['ps', '-eo', 'pid=,ppid=']).splitlines()]
    ids = {proc.pid}
    while True:
        expanded = ids | {pid for pid, parent in pairs if parent in ids}
        if expanded == ids:
            break
        ids = expanded
    for pid in sorted(ids, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def engine_command(cfg, selected, system_file, socket_path):
    """Build the node's engine invocation; classroom turn policy is opt-in."""
    argv = [cfg['engine'], '-m', selected['model'], '-sys', str(system_file), '-s', socket_path]
    if selected.get('head'):
        argv += ['-mtp', selected['head']]
    if cfg.get('end_on_question', False):
        argv += ['-end-on-question']
    return argv


class Device:
    def __init__(self, config):
        self.cfg = config
        self.instance_id = uuid.uuid4().hex
        self.collection = load_collection(config)
        self.speaker = self.collection.for_model(config['voice']) if self.collection else None
        identity_cfg = config.get('speaker_identity', {})
        self.peer_speakers = self.collection.allowed(identity_cfg) if identity_cfg.get('enabled') else set()
        if self.speaker and config.get('role') and self.speaker['role'] != config['role']:
            raise ValueError('Configured device role and speaker role differ')
        self.root = Path(config['runtime_dir']).expanduser()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.capture_lock = threading.RLock()
        self.mic_mute_file = self.root/'microphone-muted'
        self.lock = threading.RLock()
        self.lifecycle = threading.Lock()
        self.cancel = threading.Event()
        self.events = deque(maxlen=1000)
        self.serial = 0
        self.procs = {}
        self.optional_procs = {'speaker-id'}
        self.thread = None
        self.telemetry = None
        self.directory = None
        self.room = None
        self.audio_before = None
        self.pending_reaction = None
        self.state = dict(name=self.speaker['display_name'] if self.speaker else config['name'],
                          phase='stopped', ready=False, listening=False, finishing=False,
                          input='', reply='', reply_spoken=False, history=[], input_count=0, reply_count=0, expression='neutral',
                          gesture='quiet', synth_count=0, synth_bytes=0, played_bytes=0, error='', alignment=False, audible_until=0,
                          profile=None, started=None, volume=None, audio_error='', components={},
                          speaker_identity={'status': 'stopped' if config.get('room_identity') else 'disabled'}, speaker_suppress_until=0)
        threading.Thread(target=self.audio_poll, daemon=True).start()

    def emit(self, kind, text='', **fields):
        with self.lock:
            if kind in ('cut', 'stopped', 'reset'):
                self.pending_reaction = None
            if kind == 'play' and getattr(self, 'pending_reaction', None):
                cue, self.pending_reaction = self.pending_reaction, None
                self.state['expression'] = cue
                self.emit('expression', cue, source='manual_reaction')
            self.serial += 1
            event = dict(id=self.serial, kind=kind, text=text, at=time.time())
            event.update(fields)
            self.events.append(event)
            if self.directory:
                with (self.directory/'events.jsonl').open('a') as f:
                    f.write(json.dumps(event)+'\n')
            return event

    def audio_poll(self):
        while True:
            try:
                sinks = json.loads(run(['pactl', '-f', 'json', 'list', 'sinks']))
                sink = next(s for s in sinks if s['name'] == self.cfg['sink'])
                self.restore_playback_calibration(sink)
                values = [v['value'] for v in sink['volume'].values()]
                with self.lock:
                    self.state['volume'] = dict(percent=round(sum(values)/len(values)*100/65536),
                                                mute=sink['mute'], channels=sink['volume'])
                    self.state['audio_error'] = ''
                self.restore_microphone_mute()
                microphone = self.microphone_status()
                with self.lock:
                    self.state['microphone'] = microphone
            except Exception as exc:
                with self.lock:
                    self.state['audio_error'] = str(exc)
            time.sleep(2)

    def restore_playback_calibration(self, sink):
        # USB resets can reset an amplifier control without changing Pulse's
        # visible volume. Pin only explicitly configured controls, never mute.
        for control in self.cfg.get('playback_controls', []):
            card = sink['properties']['alsa.card']
            name, target = control['name'], control['value']
            if not isinstance(target, int):
                raise ValueError('Playback calibration must use integer ALSA values')
            args = ['amixer', '-c', card]
            output = run(args+['sget', name])
            levels = [int(v) for v in re.findall(r': Playback (\d+) \[', output)]
            limits = re.search(r'Limits: Playback (\d+) - (\d+)', output)
            if not levels or not limits or not int(limits[1]) <= target <= int(limits[2]):
                raise ValueError('Playback calibration control unavailable or out of range')
            if any(v != target for v in levels):
                run(args+['sset', name, str(target)])
                verified = run(args+['sget', name])
                actual = [int(v) for v in re.findall(r': Playback (\d+) \[', verified)]
                if not actual or any(v != target for v in actual):
                    raise ValueError('Playback calibration readback did not match')

    def synthesis_command(self):
        # A correction belongs to one exact voice, not every model on a node.
        gain = float(self.cfg.get('voice_gains_db', {}).get(self.cfg['voice'], 0))
        if not math.isfinite(gain) or not -12 <= gain <= 12:
            raise ValueError('Voice calibration must be between -12 and 12 dB')
        return self.cfg['piper']+['-m', self.cfg['voice'], '--output-mux', '--stream',
                                  '--volume', format(10 ** (gain / 20), '.8g')]

    def volume(self, data):
        if 'percent' in data:
            value = float(data['percent'])
            if not 0 <= value <= 100:
                raise ValueError('volume must be 0–100%')
            # Pulse cubic volume units, matching pactl/desktop percentage exactly.
            run(['pactl', 'set-sink-volume', self.cfg['sink'], str(round(value*65536/100))])
        if 'mute' in data:
            if not isinstance(data['mute'], bool):
                raise ValueError('mute must be boolean')
            run(['pactl', 'set-sink-mute', self.cfg['sink'], '1' if data['mute'] else '0'])
        return {'ok': True}

    def capture_source(self):
        sources = json.loads(run(['pactl', '-f', 'json', 'list', 'sources']))
        # Profiles change between idle and six-channel capture; retain identity.
        base = self.cfg['source'].rsplit('.', 1)[0]+'.'
        matches = [s for s in sources if s['name'] == self.cfg['source'] or s['name'].startswith(base)]
        if len(matches) != 1: raise ValueError('Microphone source unavailable or ambiguous')
        return matches[0]

    def microphone_status(self):
        source = self.capture_source()
        controls = self.cfg.get('capture_controls', [])
        switches = []
        for control in controls:
            output = run(['amixer', '-c', source['properties']['alsa.card'], 'sget', control])
            values = re.findall(r'Capture .*?\[(on|off)\]', output)
            if not values: raise ValueError('Hardware capture switch unavailable')
            switches.extend(values)
        volumes = [v['value'] for v in source['volume'].values()]
        return dict(percent=round(sum(volumes)/len(volumes)*100/65536),
                    mute=source['mute'], hardware_mute=all(v == 'off' for v in switches) if switches else None,
                    source=source['name'])

    def restore_microphone_mute(self):
        with self.capture_lock:
            if self.mic_mute_file.exists(): self.microphone({'mute': True})

    def microphone(self, data):
        with self.capture_lock:
            return self.set_microphone(data)

    def set_microphone(self, data):
        # Validate the whole request before changing any capture controls.
        if 'percent' in data and not 0 <= float(data['percent']) <= 100:
            raise ValueError('volume must be 0–100%')
        if 'mute' in data and not isinstance(data['mute'], bool):
            raise ValueError('mute must be boolean')
        controls = self.cfg.get('capture_controls', [])
        if 'mute' in data and not controls: raise ValueError('Hardware capture mute is not configured')
        source = self.capture_source()
        if 'percent' in data:
            run(['pactl', 'set-source-volume', source['name'], str(round(float(data['percent'])*65536/100))])
        if 'mute' in data:
            # Remember explicit mute across profile changes and agent restarts.
            if data['mute']: self.mic_mute_file.touch(mode=0o600)
            run(['pactl', 'set-source-mute', source['name'], '1' if data['mute'] else '0'])
            for control in controls:
                run(['amixer', '-c', source['properties']['alsa.card'], 'sset', control,
                     'nocap' if data['mute'] else 'cap'])
            if not data['mute']: self.mic_mute_file.unlink(missing_ok=True)
        status = self.microphone_status()
        if 'mute' in data and status['hardware_mute'] != data['mute']:
            raise ValueError('Hardware mute readback did not match request')
        with self.lock: self.state['microphone'] = status
        if data.get('mute') is False and self.state['ready']:
            self.listen(True)
        return {'ok': True, 'microphone': status}

    def snapshot(self, after=0):
        with self.lock:
            st = self.state
            now = time.time()
            if now >= st.get('gesture_until', 0):
                st['gesture'] = 'quiet'
            # Generation/synthesis completion precedes audible completion.
            # Keep the expression through queued PCM, including sentence gaps.
            if (st['reply_count'] >= st['input_count'] and st['synth_count'] >= st['reply_count']
                    and st['played_bytes'] >= st['synth_bytes'] and now >= st['audible_until']):
                st['expression'] = 'neutral'
            state = dict(st)
            identity = dict(st.get('speaker_identity', {'status': 'disabled'}))
            if identity.get('status') not in ('disabled', 'stopped'):
                if now-st.get('speaker_identity_at', 0) > 3:
                    identity.update(status='unavailable', speaker_id=None, voice_id=None, display_name=None, face_color=None)
                elif (not st['listening'] and not st.get('finishing')) or now < st.get('speaker_suppress_until', 0):
                    identity.update(status='paused', speaker_id=None, voice_id=None, display_name=None, face_color=None)
            state['speaker_identity'] = identity
            state['components'] = {name: {'pid': p.pid, 'alive': p.poll() is None} for name, p in self.procs.items()}
            # Cancellation deliberately reaps children before final cleanup.
            # Polling during that interval must not report a worker crash.
            if state['ready'] and self.cancel.is_set():
                state.update(ready=False, phase='error' if state.get('error') else 'stopping')
            if state['ready']:
                if any(not p['alive'] for name, p in state['components'].items() if name not in self.optional_procs):
                    state['phase'] = 'error'
                    state['error'] = state.get('error') or 'A pipeline process exited; stop and reload to recover.'
                    state['ready'] = False
                elif time.time() < state['audible_until']:
                    state['phase'] = 'speaking'
                elif state['phase'] == 'speaking' or (state['phase'] == 'thinking'
                        and state['reply_count'] >= state['input_count']
                        and state['synth_count'] >= state['reply_count']
                        and state['played_bytes'] >= state['synth_bytes']):
                    state['phase'] = 'listening' if state['listening'] else 'finishing' if state.get('finishing') else 'paused'
            if self.room and state['ready'] and self.room.turn_active:
                state['phase'] = 'speaking' if now < state['audible_until'] else 'thinking'
            return dict(server_time=time.time(), state=state, events=[e for e in self.events if e['id'] > after], cursor=self.serial,
                        instance_id=self.instance_id,
                        default_profile=self.cfg.get('default_profile'),
                        capabilities={'pipeline': not self.cfg.get('controls_only', False),
                                      'capture': sys.platform != 'win32' and not self.cfg.get('controls_only', False),
                                      'turn_limits': sys.platform == 'win32' or bool(self.cfg.get('turn_limit_support', False)),
                                      'room_session': bool(self.cfg.get('room_identity'))},
                        profiles={k: {'label': v.get('label', k), 'model': v['model'], 'head': v.get('head'), 'env': v.get('env', {})}
                                  for k, v in self.cfg['profiles'].items()},
                        decision_heads=decision_head_details(self.cfg),
                        speaker=self.speaker, collection_id=self.collection.id if self.collection else None,
                        role=self.cfg.get('role'),
                        host={'hostname': socket.gethostname().split('.')[0],
                              'platform': self.cfg.get('platform', 'Unknown platform')},
                        asr=self.cfg['whisper_model'], voice=self.cfg['voice'], sink=self.cfg['sink'],
                        source=self.cfg['source'], sync_offset_ms=self.cfg.get('sync_offset_ms', 100))

    def command(self, text):
        if not self.directory or not self.state['ready']:
            raise ValueError('pipeline is not ready')
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(text.encode(), str(self.directory/'control.sock'))

    def listen(self, enabled, *, finishing=False):
        # Closing model input at the turn limit must not hide active peer tracking.
        # Explicit listening pauses still gate identity; only final playback drains
        # enable this exception. Playback suppression remains independent.
        with self.lock:
            previous = self.state.get('finishing', False)
            self.state['finishing'] = finishing and not enabled
            try:
                self.command('listen '+('1' if enabled else '0'))
            except Exception:
                self.state['finishing'] = previous
                raise
        return {'ok': True}

    def prompt(self, text, reaction=None):
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 3500:
            raise ValueError('prompt must be 1–3500 bytes')
        if self.state.get('finishing') or self.snapshot()['state']['phase'] in ('speaking', 'thinking', 'hearing'):
            raise ValueError('pipeline is busy')
        if self.state.get('turn_limit') and self.state['input_count'] >= self.state['turn_limit']:
            raise ValueError('turn limit reached')
        if reaction is not None and reaction not in ('happy', 'sad', 'angry', 'neutral'):
            raise ValueError('unknown expression reaction')
        with self.lock:
            self.pending_reaction = reaction
            try:
                self.command('prompt '+text)
            except Exception:
                self.pending_reaction = None
                raise
        return {'ok': True}

    def spawn(self, name, argv, **kwargs):
        f = (self.directory/(name+'.log')).open('wb')
        p = subprocess.Popen(argv, stderr=f, stdout=kwargs.pop('stdout', f), start_new_session=True, **kwargs)
        f.close()
        with self.lock:
            self.procs[name] = p
        self.emit('process', name, argv=argv, pid=p.pid)
        return p

    def check_processes(self, stage, *, include_optional=False):
        with self.lock:
            exited = [(name, p.poll()) for name, p in self.procs.items()
                      if include_optional or name not in self.optional_procs]
        for name, code in exited:
            if code is not None:
                raise RuntimeError(f'{stage} process exited: {name} (exit {code}); '
                                   f'see {self.directory / (name+".log")}')

    def start(self, data):
        if self.cfg.get('controls_only', False):
            raise ValueError('The trio voice pipeline is not configured on this node yet')
        profile = data.get('profile')
        if profile not in self.cfg['profiles']:
            raise ValueError('unknown model profile')
        prompt = data.get('system_prompt', '')
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 16000:
            raise ValueError('system prompt must be 1–16000 characters')
        turn_limit = data.get('turn_limit', 0)
        if type(turn_limit) is not int or not 0 <= turn_limit <= 48:
            raise ValueError('turn limit must be an integer from 1–48, or 0 for an unbounded manual load')
        if turn_limit and not self.snapshot()['capabilities']['turn_limits']:
            raise ValueError('This node needs a voicecat build with turn-limit support')
        peer_id = data.get('peer_speaker_id')
        if peer_id is not None and (not self.collection or data.get('peer_collection_id') != self.collection.id
                                    or not self.collection.speaker(peer_id)):
            raise ValueError('Peer speaker must belong to the configured collection')
        prompt += identity_prompt(self.collection, self.speaker, peer_id)
        room_settings = data.get('room')
        if room_settings is not None:
            if (not self.cfg.get('room_identity') or not isinstance(room_settings, dict)
                    or not re.fullmatch(r'[a-f0-9-]{36}', str(room_settings.get('session', '')))
                    or not isinstance(room_settings.get('topic'), str)
                    or not 1 <= len(room_settings['topic']) <= 1000
                    or room_settings.get('mode', 'active') not in ('active', 'passive')):
                raise ValueError('Invalid or unavailable room session')
        with self.lifecycle:
            if self.thread and self.thread.is_alive():
                raise ValueError('pipeline is already loading or running; stop it first')
            with self.lock:
                if any(p.poll() is None for p in self.procs.values()):
                    raise ValueError('previous pipeline processes are still running; stop them first')
                # Stopped room workers used to leave dead Piper entries here.
                # Every new load must check only the processes it starts.
                self.procs.clear()
            self.cancel.clear()
            self.room = None
            worker = self.work
            if room_settings is not None:
                from room_direct import DirectRoomRuntime
                self.room = DirectRoomRuntime(self, room_settings)
                worker = self.room.work
            else:
                self.state.pop('room', None)
                self.state['pipeline_mode'] = 'active'
            self.thread = threading.Thread(target=worker, args=(profile, prompt, bool(data.get('listen', False)), turn_limit), daemon=True)
            self.thread.start()
        return {'ok': True}

    def work(self, profile, prompt, listen, turn_limit=0):
        diagnostics = None
        try:
            self.directory = Path(tempfile.mkdtemp(prefix='session-', dir=self.root))
            with self.lock:
                self.state.update(phase='loading', ready=False, error='', input='', reply='', history=[], input_count=0,
                                  reply_count=0, reply_spoken=False, expression='neutral', gesture='quiet', synth_count=0, synth_bytes=0, played_bytes=0, last_input='', profile=profile, started=time.time(), alignment=False,
                                  audible_until=0, listening=False, finishing=False, turn_limit=turn_limit, speaker_identity={'status': 'disabled'}, speaker_identity_at=0, speaker_suppress_until=0)
            self.emit('reset')
            (self.directory/'system.txt').write_text(prompt)
            cfg = self.cfg
            selected = cfg['profiles'][profile]
            lg_sock = str(self.directory/'lg.sock')
            ff_sock = cfg.get('ff_socket') or str(self.directory/'ff.sock')
            if not cfg.get('ff_socket'):
                if cfg.get('card') and cfg.get('capture_profile'):
                    cards = json.loads(run(['pactl', '-f', 'json', 'list', 'cards']))
                    self.audio_before = next(c for c in cards if c['name'] == cfg['card'])['active_profile']
                    run(['pactl', 'set-card-profile', cfg['card'], cfg['capture_profile']])
                self.restore_microphone_mute()
                self.spawn('audio', [cfg['far_field'], '-s', ff_sock, '--source', cfg['source'], '--sink', cfg['sink'],
                                    '--channels', str(cfg.get('channels', 6)), '--use-channel', '0', '--gain-db', '0',
                                    '--scene', '--no-aec'])
            elif not Path(ff_sock).is_socket():
                raise ValueError('configured existing audio service socket is missing')
            env = {k: v for k, v in os.environ.items() if not k.startswith('LG_')}
            env.update(cfg.get('env', {}))
            env.update(selected.get('env', {}))
            engine = engine_command(cfg, selected, self.directory/'system.txt', lg_sock)
            self.spawn('engine', engine, env=env)
            if cfg.get('whisper_command'):
                self.spawn('whisper', cfg['whisper_command'])
            deadline = time.time()+180
            while True:
                if self.cancel.wait(.2):
                    return
                self.check_processes('Startup')
                healthy = False
                try:
                    with urllib.request.urlopen(cfg['whisper_health'], timeout=1) as response:
                        healthy = response.status == 200
                except Exception:
                    pass
                if Path(lg_sock).is_socket() and Path(ff_sock).is_socket() and healthy:
                    break
                if time.time() > deadline:
                    raise RuntimeError('engine/audio/ASR startup timeout')
            self.telemetry = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.telemetry.bind(str(self.directory/'events.sock'))
            self.telemetry.settimeout(.2)
            threading.Thread(target=self.receive, daemon=True).start()
            voice_env = os.environ.copy()
            voice_env.update(cfg.get('env', {}))
            if 'name_pronunciations' in cfg:
                voice_env['VOICE_NAME_PRONUNCIATIONS'] = json.dumps(cfg['name_pronunciations'])
            voice_env.pop('LG_AUDIO_DIAGNOSTICS', None)
            if cfg.get('audio_diagnostics_dir'):
                diagnostics = Path(cfg['audio_diagnostics_dir']).expanduser()/self.directory.name
                diagnostics.mkdir(parents=True, mode=0o700, exist_ok=False)
                voice_env['LG_AUDIO_DIAGNOSTICS'] = str(diagnostics)
                manifest = dict(session=self.directory.name, profile=profile, started=time.time(),
                                speaker_source=cfg['sink']+'.monitor', microphone_source=cfg['source'],
                                microphone='s16le mono 16000 Hz; processed tap before listening gate',
                                speaker='s16le stereo 16000 Hz',
                                timing='microphone-clock.tsv: wall-clock receive time, sample offset, sample count; not ADC timestamps',
                                voicecat=cfg['voicecat'])
                (diagnostics/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                with (diagnostics/'speaker.s16le').open('wb') as output:
                    self.spawn('speaker-capture', ['parec', '--record', '--raw',
                               '--device='+cfg['sink']+'.monitor', '--format=s16le', '--rate=16000',
                               '--channels=2', '--channel-map=front-left,front-right',
                               '--latency-msec=20', '--process-time-msec=10'], stdout=output)
                self.emit('audio_capture', str(diagnostics))
            tap = self.spawn('mic', [cfg['far_field'], '--tap', ff_sock, '--mux'], stdout=subprocess.PIPE)
            synth = shlex.join(self.synthesis_command())
            play = shlex.join([cfg['far_field'], '--speak', ff_sock, '--rate', str(cfg.get('sample_rate', 22050))])
            args = [cfg['voicecat'], lg_sock, '--stdin-mux', '--vad-level', str(cfg.get('vad_level', 200)), '--hang-ms', '500',
                    '--commit-ms', '1100', '--settle-ms', '300', '--clock', '0', '--idle-compress', '0',
                    '--barge-mult', '7', '--barge-onset', '9', '--barge-hang', '7', '--hush-tail', '--sound-tags',
                    '--mood-route', '--duck-sock', ff_sock, '--whisper-url', cfg['whisper_url'],
                    '--mouth-synth', synth, '--mouth-play', play, '--monitor-sock', str(self.directory/'events.sock'),
                    '--control-sock', str(self.directory/'control.sock'), '--start-muted']
            if turn_limit:
                args += ['--turn-limit', str(turn_limit)]
            self.spawn('voice', args, stdin=tap.stdout, env=voice_env)
            tap.stdout.close()
            deadline = time.time()+60
            while not (self.directory/'control.sock').is_socket() or not self.state.get('piper_config'):
                if self.cancel.wait(.1):
                    return
                if self.procs['voice'].poll() is not None or time.time() > deadline:
                    raise RuntimeError('voice/Piper startup timeout')
            with self.lock:
                self.state.update(ready=True, phase='paused')
            self.listen(listen)
            self.emit('ready', profile)
            identity_cfg = dict(cfg.get('speaker_identity', {}))
            if identity_cfg.get('enabled', False):
                with self.lock:
                    self.state['speaker_identity'] = {'status': 'waiting'}
                    self.state['speaker_identity_at'] = time.time()
                identity_cfg.update(collection_id=self.collection.id, collection={'speakers': self.collection.speakers})
                identity_path = self.directory/'speaker-identity.json'
                identity_path.write_text(json.dumps(identity_cfg))
                try:
                    self.spawn('speaker-id', [identity_cfg.get('python', sys.executable), str(HERE/'speaker_identity.py'),
                               '--config', str(identity_path), '--audio-socket', ff_sock,
                               '--events-socket', str(self.directory/'events.sock')])
                except OSError as exc:
                    self.emit('speaker_identity_error', str(exc))
                    with self.lock:
                        self.state['speaker_identity'] = {'status': 'unavailable', 'error': str(exc)}
            while not self.cancel.wait(.25):
                self.check_processes('Pipeline')
        except Exception as exc:
            with self.lock:
                self.state.update(error=str(exc), phase='error')
            self.emit('error', str(exc))
        finally:
            with self.lock:
                self.state['ready'] = False
                if not self.state['error']: self.state['phase'] = 'stopping'
            for p in reversed(list(self.procs.values())):
                stop_tree(p)
            self.procs.clear()
            if self.telemetry:
                self.telemetry.close()
                self.telemetry = None
            if self.audio_before:
                try:
                    run(['pactl', 'set-card-profile', self.cfg['card'], self.audio_before])
                except Exception as exc:
                    self.emit('error', 'audio profile restore: '+str(exc))
                self.audio_before = None
            with self.lock:
                self.state.update(ready=False, listening=False, finishing=False, audible_until=0, piper_config=None,
                                  expression='neutral', gesture='quiet', gesture_until=0, speaker_identity={'status': 'stopped'})
                if not self.state['error']:
                    self.state['phase'] = 'stopped'
            self.emit('stopped')
            if diagnostics:
                for name in ('events.jsonl', 'voice.log', 'audio.log', 'speaker-capture.log', 'system.txt'):
                    source = self.directory/name
                    if source.exists(): shutil.copy2(source, diagnostics/name)

    def receive(self):
        stream = self.telemetry
        while stream is self.telemetry:
            try:
                event = json.loads(stream.recv(65536))
            except socket.timeout:
                continue
            except OSError:
                return
            except (ValueError, UnicodeError):
                continue
            kind, text = event['kind'], event['text']
            with self.lock:
                st = self.state
                if kind == 'speaker_identity':
                    self.receive_identity(text, event['at'])
                elif kind == 'config':
                    st['piper_config'] = text
                elif kind == 'input':
                    st['input'] += text
                    st['phase'] = 'hearing'
                elif kind == 'input_done':
                    st['input'] += text
                    st['last_input'] = st['input'].strip()
                    st['input'] = ''
                    st['reply'] = ''
                    st['reply_spoken'] = False
                    st['input_count'] += 1
                    if st.get('turn_limit') and st['input_count'] >= st['turn_limit']:
                        st['finishing'] = True
                    st['phase'] = 'thinking'
                elif kind == 'reply':
                    st['reply'] = (st['reply']+text)[-16000:]
                elif kind == 'clause':
                    st['reply_spoken'] = True
                elif kind == 'reply_done':
                    st['reply_count'] += 1
                    st['history'] = (st['history']+[{'heard':st.get('last_input',''), 'said':st['reply']}])[-4:]
                elif kind == 'synth_done':
                    st['synth_count'] += 1
                    st['synth_bytes'] = int(text)
                elif kind == 'expression':
                    st['expression'] = text.strip()
                elif kind == 'gesture':
                    st['gesture'] = text.strip()
                    st['gesture_until'] = event['at'] + 1.95 if st['gesture'] in ('nod', 'shake') else 0
                elif kind == 'alignment':
                    st['alignment'] = True
                elif kind == 'play':
                    start, offset, count, rate = text.split('\t')
                    st['played_bytes'] = int(offset)+int(count)
                    st['audible_until'] = float(start)+int(count)/(int(rate)*2)+self.cfg.get('sync_offset_ms', 100)/1000
                    st['phase'] = 'speaking'
                    st['speaker_suppress_until'] = st['audible_until']+.2
                elif kind == 'listening':
                    if text == '1' and not st['listening']:
                        st['speaker_suppress_until'] = max(st.get('speaker_suppress_until', 0), event['at'])
                    st['listening'] = text == '1'
                    if st['phase'] in ('paused', 'listening', 'finishing'):
                        st['phase'] = 'listening' if st['listening'] else 'finishing' if st.get('finishing') else 'paused'
                elif kind == 'cut':
                    st.update(audible_until=0, expression='neutral', gesture='quiet', gesture_until=0,
                              speaker_suppress_until=event['at']+.2)
                self.emit(kind, text, at=event['at'])

    def receive_identity(self, text, at):
        """UI-only estimates; final playback drains may track with model input closed."""
        try:
            if at < self.state.get('speaker_identity_at', 0):
                return
            incoming = json.loads(text)
            old = self.state.get('speaker_identity', {})
            last = old.get('last_peer')
            def canonical(candidate):
                if not isinstance(candidate, dict) or not self.collection:
                    return None
                speaker_id = candidate.get('speaker_id')
                if candidate.get('collection_id') != self.collection.id or speaker_id not in self.peer_speakers:
                    return None
                info = self.collection.speaker(speaker_id)
                if candidate.get('voice_id') != info['voice_id']:
                    return None
                return info
            candidate = incoming.get('last_peer')
            info = canonical(candidate)
            candidate_at = float(candidate.get('at', 0)) if info else 0
            blocked = ((not self.state['listening'] and not self.state.get('finishing'))
                       or at < self.state.get('speaker_suppress_until', 0))
            valid = (info and math.isfinite(candidate_at) and
                     self.state.get('speaker_suppress_until', 0) < candidate_at <= at)
            if not blocked and valid and candidate_at >= (last or {}).get('at', 0):
                last = dict(info, at=candidate_at)
            current = canonical(incoming)
            fresh = valid and current and info['speaker_id'] == current['speaker_id']
            status = incoming.get('status', 'uncertain')
            if blocked:
                status = 'paused'
            elif status == 'identified' and not fresh:
                status = 'identifying'
            if blocked or not fresh or status != 'identified':
                current = {}
            incoming = dict(current or {}, status=status,
                            collection_id=self.collection.id if self.collection else None,
                            speaker_id=(current or {}).get('speaker_id'),
                            voice_id=(current or {}).get('voice_id'),
                            display_name=(current or {}).get('display_name'),
                            last_peer=last, attribution='estimate', mode='single_peer',
                            allow_named_address=False, overlap_checked=False)
            self.state['speaker_identity'] = incoming
            self.state['speaker_identity_at'] = at
        except (ValueError, TypeError, AttributeError):
            self.state['speaker_identity'] = {'status': 'unavailable', 'error': 'Invalid identity event'}

    def stop(self):
        with self.lifecycle:
            with self.lock:
                self.cancel.set()
                if self.state['ready']:
                    self.state.update(ready=False, listening=False,
                                      phase='error' if self.state.get('error') else 'stopping')
            if self.room: self.room.terminate_owned()
            if self.thread:
                self.thread.join(timeout=20)
                if self.thread.is_alive():
                    raise RuntimeError('stop still in progress')
        return {'ok': True}


def create_app(config):
    app = Flask(__name__, template_folder=str(HERE), static_folder=str(HERE))
    app.config['MAX_CONTENT_LENGTH'] = 65536
    device_type = Device
    if sys.platform == 'win32':
        from windows_audio import WindowsDeviceMixin
        class WindowsDevice(WindowsDeviceMixin, Device):
            pass
        device_type = WindowsDevice
    device = device_type(config)
    key = config['key']
    session = dict(active=False, phase='idle', turns=0, error='')
    session_lock = threading.Lock()
    session_stop = threading.Event()
    session_thread = [None]
    playback_capture = [None]
    playback_lock = threading.Lock()

    def local(action, data=None):
        if action == 'status':
            return device.snapshot(int((data or {}).get('after', 0)))
        if action == 'start': return device.start(data)
        if action == 'stop': return device.stop()
        if action == 'volume': return device.volume(data)
        if action == 'microphone': return device.microphone(data)
        if action in ('room-prepare', 'room-grant', 'room-idle', 'room-listen', 'room-open', 'room-permit'):
            if not device.room: raise ValueError('No room session is loaded')
            return device.room.control(action, data)
        if device.room and device.state['ready'] and action in ('listen', 'finish-turn', 'prompt'):
            raise ValueError('The room coordinator owns listening and speaking; stop the session for manual use')
        if action == 'listen': return device.listen(bool(data['enabled']))
        if action == 'finish-turn': return device.listen(False, finishing=True)
        if action == 'endpoint-hold':
            ms = int(data['ms'])
            if not 0 <= ms <= 5000: raise ValueError('hold must be 0–5000 ms')
            if device.room: device.room.hold(ms)
            else: device.command('endpoint-hold '+str(ms))
            return {'ok': True}
        if action == 'prompt': return device.prompt(data['text'], reaction=data.get('reaction'))
        raise ValueError('unknown action')

    def call(side, action, data=None):
        if side == 'local':
            return local(action, data)
        peer = config.get('peer')
        if not peer:
            raise ValueError('peer is not configured')
        url = peer['url'].rstrip('/')+'/api/local/'+action
        if action == 'status':
            url += '?after='+str(int((data or {}).get('after', 0)))
        body = None if action == 'status' else json.dumps(data or {}).encode()
        req = urllib.request.Request(url, data=body, headers={'X-Voice-Key': peer['key'], 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=25 if action == 'stop' else 5) as response:
            return json.load(response)

    def recording_models():
        # The coordinator supplies its ready states before enabling speech.
        if session['active'] and session['phase'] == 'loading': return None
        states = [device.snapshot()['state']]
        if not states[0]['ready']: return None
        if config.get('peer'): states.append(call('peer', 'status')['state'])
        return [s['profile'] for s in states] if all(s['ready'] for s in states) else None

    recording = Recording(config, recording_models, lambda: device.capture_source()['name'])

    def conversation(data):
        def status(side):
            return call(side, 'status', {'after': 9223372036854775807})['state']
        def expired():
            return bool(data['seconds'] and session.get('started') and time.time()-session['started'] >= data['seconds'])
        def drained(state):
            return (state['reply_count'] >= state['input_count'] and state['synth_count'] >= state['reply_count']
                    and state['played_bytes'] >= state['synth_bytes'] and state['phase'] not in ('speaking', 'thinking'))
        try:
            session.update(phase='loading', error='', turns=0, started=None)
            participants = {side: call(side, 'status').get('speaker') for side in ['local', 'peer']}
            for side in ['peer', 'local']:
                call(side, 'stop')
                settings = dict(data[side], listen=False)
                other = participants['local' if side == 'peer' else 'peer']
                settings['peer_speaker_id'] = other['speaker_id'] if other else None
                settings['peer_collection_id'] = other['collection_id'] if other else None
                call(side, 'start', settings)
            deadline = time.time()+240
            while not session_stop.wait(.2) and not expired():
                states = {side: status(side) for side in ['local', 'peer']}
                if all(st['ready'] for st in states.values()): break
                if any(st['error'] for st in states.values()): raise RuntimeError(str(states))
                if time.time()>deadline: raise RuntimeError('devices did not become ready')
            if session_stop.is_set() or expired(): return
            recording.models_ready([states[side]['profile'] for side in ['local', 'peer']])
            if session_stop.is_set() or expired(): return
            for side in ['local', 'peer']:
                call(side, 'listen', {'enabled': True})
            call('peer', 'endpoint-hold', {'ms': 1500})
            held = {'local': False, 'peer': True}
            call('local', 'prompt', {'text': 'Begin the lesson about '+data['topic']+'. Give one short explanation and ask your student one question. Then wait.'})
            session['phase'] = 'running'
            if session['started'] is None: session['started'] = time.time()
            seen = {'local': 0, 'peer': 0}
            last = time.time()
            while not session_stop.wait(.05) and not expired():
                states = {side: status(side) for side in ['local', 'peer']}
                for side, other in [('local', 'peer'), ('peer', 'local')]:
                    state = states[side]
                    # A peer's sentence pause is not a completed turn. Keep
                    # capturing/prefilling; defer only silence-based closure.
                    # Leases expire if the coordinator disappears. Retain a
                    # short acoustic tail after estimated playback drains.
                    busy = not drained(states[other])
                    if busy or held[side]:
                        call(side, 'endpoint-hold', {'ms': 1500 if busy else 200})
                    held[side] = busy
                    if not state['ready']: raise RuntimeError(side+' stopped: '+state['error'])
                    if state['input_count'] > seen[side]:
                        seen[side] = state['input_count']
                        session['turns'] += 1
                        last = time.time()
                limit = bool(data['turns'] and session['turns'] >= data['turns'])
                # Both mouths have drained and a model deliberately completed
                # without speech: there is no next acoustic turn to wait for.
                if (all(drained(st) and time.time() >= st['audible_until'] for st in states.values())
                        and any(st['input_count'] > 0 and not st.get('reply_spoken', True) for st in states.values())):
                    return
                if limit:
                    for side in ['local', 'peer']: call(side, 'finish-turn', {})
                    deadline = time.time()+45
                    while not session_stop.wait(.1) and not expired():
                        states = {side: status(side) for side in ['local', 'peer']}
                        if all(drained(st) for st in states.values()): break
                        if time.time()>deadline: raise RuntimeError('Final audio did not drain within 45 seconds.')
                    return
                if time.time()-last > 90: raise RuntimeError('No new turn for 90 seconds; check microphones and transcripts.')
        except Exception as exc:
            session['error'] = str(exc)
        finally:
            recording.stop(wait=True)
            for side in ['local', 'peer']:
                try: call(side, 'stop')
                except Exception as exc: session['error'] += ' '+str(exc)
            session.update(active=False, phase='error' if session['error'] else 'stopped')

    @app.before_request
    def guard():
        if request.path.startswith('/api/'):
            if not secrets.compare_digest(request.headers.get('X-Voice-Key', ''), key):
                return jsonify(error='invalid device key'), 403
            origin = request.headers.get('Origin')
            if origin and origin != request.host_url.rstrip('/'):
                return jsonify(error='cross-origin control is disabled'), 403

    @app.get('/')
    def index():
        return render_template('index.html', key=key, visemes=load_visemes(), phonemes=PHONEME_TO_VISEME,
                               name=device.state['name'], peer=bool(config.get('peer')), role=config.get('role', 'teacher'),
                               capture=device.snapshot()['capabilities']['capture'], default_prompt=config.get('system_prompt', ''))

    @app.route('/api/<side>/<action>', methods=['GET', 'POST'])
    def api(side, action):
        try:
            if side not in ('local', 'peer'): raise ValueError('unknown device')
            if request.method == 'GET' and action != 'status': return jsonify(error='POST required'), 405
            if request.method == 'POST' and not request.is_json: return jsonify(error='JSON required'), 415
            return jsonify(call(side, action, request.args if request.method == 'GET' else request.get_json()))
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    @app.get('/api/recording/asr')
    def recording_asr():
        with device.lock:
            room = device.room
            if not room or not device.state['ready'] or device.cancel.is_set():
                return jsonify(error='Room ASR input is not running'), 503
            if request.args.get('session') != room.session:
                return jsonify(error='Stale room session'), 409
        try: capture = room.asr_input.subscribe(device.speaker)
        except ValueError as exc: return jsonify(error=str(exc)), 409
        if not capture.ready.wait(5) or capture.error or capture.ended.is_set():
            capture.close()
            return jsonify(error=capture.error or 'No PCM from room ASR input'), 503
        response = Response(iter(capture), mimetype='application/octet-stream', headers={
            'X-ASR-Protocol': 'little-gemma-asr-input-v1', 'Cache-Control': 'no-store',
            'X-Accel-Buffering': 'no'})
        response.call_on_close(capture.close)
        return response

    @app.post('/api/recording/asr/stop')
    def recording_asr_stop():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data.get('capture_id'):
            return jsonify(error='ASR capture ID required'), 400
        with device.lock:
            room = device.room
            if not room or data.get('session') != room.session:
                return jsonify(error='Stale room session'), 409
        try: return jsonify(room.asr_input.stop(data['capture_id']))
        except ValueError as exc: return jsonify(error=str(exc)), 409

    @app.get('/api/recording/pcm')
    def recording_pcm():
        from recording_audio import PlaybackCapture
        with playback_lock:
            current = playback_capture[0]
            if current and current.thread.is_alive():
                return jsonify(error='Speaker output is already being recorded'), 409
            options = {}
            if sys.platform != 'win32' and config.get('recording_playback_tap'):
                with device.lock:
                    path = (getattr(device.room, 'ff_socket', None) if device.room else
                            config.get('ff_socket') or (str(device.directory/'ff.sock') if device.directory else None))
                if not path: return jsonify(error='Voice playback is not running'), 503
                options['playback_socket'] = path
            capture = PlaybackCapture(config['sink'], **options)
            playback_capture[0] = capture
            try: capture.start()
            except Exception as exc: return jsonify(error=str(exc)), 503
        response = Response(iter(capture), mimetype='application/octet-stream', headers={
            'X-PCM-Format': 's16le', 'X-PCM-Rate': str(capture.rate),
            'X-PCM-Channels': str(capture.channels), 'Cache-Control': 'no-store',
            'X-Accel-Buffering': 'no'})
        response.call_on_close(capture.close)
        return response

    @app.route('/api/recording', methods=['GET', 'POST', 'DELETE'])
    def recording_api():
        try:
            if request.method == 'POST': return jsonify(recording.arm())
            if request.method == 'DELETE': return jsonify(recording.stop())
            return jsonify(recording.snapshot())
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    @app.route('/api/conversation', methods=['GET', 'POST', 'DELETE'])
    def conversation_api():
        with session_lock:
            if request.method == 'POST':
                if session['active']: return jsonify(error='conversation already running'), 409
                if not config.get('peer'): return jsonify(error='peer is not configured'), 400
                data = request.get_json()
                try:
                    if not isinstance(data['topic'], str) or not 1 <= len(data['topic']) <= 1000: raise ValueError('topic required, max 1000 characters')
                    data['seconds'] = int(data.get('seconds', 180)); data['turns'] = int(data.get('turns', 48))
                    if not 1 <= data['seconds'] <= 86400 or not 1 <= data['turns'] <= 48: raise ValueError('duration must be positive and turn limit must be 1–48')
                    for side in ['local', 'peer']:
                        if not isinstance(data[side]['system_prompt'], str) or not data[side]['system_prompt']: raise ValueError('both system prompts are required')
                except (KeyError, TypeError, ValueError) as exc: return jsonify(error=str(exc)), 400
                session_stop.clear(); session.update(active=True, error='')
                session_thread[0] = threading.Thread(target=conversation, args=(data,), daemon=True)
                session_thread[0].start()
            elif request.method == 'DELETE':
                session_stop.set()
            return jsonify(**session, recording=recording.snapshot())

    def shutdown():
        session_stop.set()
        recording.stop(wait=True)
        if playback_capture[0]: playback_capture[0].close()
        if session_thread[0] and session_thread[0].is_alive():
            session_thread[0].join(timeout=60)
        device.stop()
        if hasattr(device, 'audio_shutdown'):
            device.audio_shutdown.set()

    app.shutdown = shutdown
    app.device = device
    app.recording = recording
    app.session_stop = session_stop
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8095)
    args = p.parse_args()
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    app = create_app(cfg)
    def shutdown(*_):
        app.shutdown()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        app.shutdown()


if __name__ == '__main__': main()

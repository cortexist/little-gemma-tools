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

from flask import Flask, jsonify, render_template, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recording import Recording
sys.path.insert(0, str(HERE.parent))
from voicedemo import load_visemes, PHONEME_TO_VISEME


def run(args):
    return subprocess.check_output(args, text=True, timeout=5)


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


class Device:
    def __init__(self, config):
        self.cfg = config
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
        self.thread = None
        self.telemetry = None
        self.directory = None
        self.audio_before = None
        self.state = dict(name=config['name'], phase='stopped', ready=False, listening=False,
                          input='', reply='', reply_spoken=False, history=[], input_count=0, reply_count=0, expression='neutral',
                          gesture='quiet', synth_count=0, synth_bytes=0, played_bytes=0, error='', alignment=False, audible_until=0,
                          profile=None, started=None, volume=None, audio_error='', components={})
        threading.Thread(target=self.audio_poll, daemon=True).start()

    def emit(self, kind, text='', **fields):
        with self.lock:
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
            state['components'] = {name: {'pid': p.pid, 'alive': p.poll() is None} for name, p in self.procs.items()}
            if state['ready']:
                if any(not p['alive'] for p in state['components'].values()):
                    state['phase'] = 'error'
                    state['error'] = 'A pipeline process exited; stop and reload to recover.'
                    state['ready'] = False
                elif time.time() < state['audible_until']:
                    state['phase'] = 'speaking'
                elif state['phase'] == 'speaking' or (state['phase'] == 'thinking'
                        and state['reply_count'] >= state['input_count']
                        and state['synth_count'] >= state['reply_count']
                        and state['played_bytes'] >= state['synth_bytes']):
                    state['phase'] = 'listening' if state['listening'] else 'paused'
            return dict(server_time=time.time(), state=state, events=[e for e in self.events if e['id'] > after], cursor=self.serial,
                        profiles={k: {'label': v.get('label', k), 'model': v['model'], 'head': v.get('head'), 'env': v.get('env', {})}
                                  for k, v in self.cfg['profiles'].items()},
                        asr=self.cfg['whisper_model'], voice=self.cfg['voice'], sink=self.cfg['sink'],
                        source=self.cfg['source'], sync_offset_ms=self.cfg.get('sync_offset_ms', 100))

    def command(self, text):
        if not self.directory or not self.state['ready']:
            raise ValueError('pipeline is not ready')
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(text.encode(), str(self.directory/'control.sock'))

    def listen(self, enabled):
        self.command('listen '+('1' if enabled else '0'))
        return {'ok': True}

    def prompt(self, text):
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 3500:
            raise ValueError('prompt must be 1–3500 bytes')
        if self.snapshot()['state']['phase'] in ('speaking', 'thinking', 'hearing'):
            raise ValueError('pipeline is busy')
        self.command('prompt '+text)
        return {'ok': True}

    def spawn(self, name, argv, **kwargs):
        f = (self.directory/(name+'.log')).open('wb')
        p = subprocess.Popen(argv, stderr=f, stdout=kwargs.pop('stdout', f), start_new_session=True, **kwargs)
        f.close()
        with self.lock:
            self.procs[name] = p
        self.emit('process', name, argv=argv, pid=p.pid)
        return p

    def start(self, data):
        profile = data.get('profile')
        if profile not in self.cfg['profiles']:
            raise ValueError('unknown model profile')
        prompt = data.get('system_prompt', '')
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 16000:
            raise ValueError('system prompt must be 1–16000 characters')
        with self.lifecycle:
            if self.thread and self.thread.is_alive():
                raise ValueError('pipeline is already loading or running; stop it first')
            self.cancel.clear()
            self.thread = threading.Thread(target=self.work, args=(profile, prompt, bool(data.get('listen', False))), daemon=True)
            self.thread.start()
        return {'ok': True}

    def work(self, profile, prompt, listen):
        diagnostics = None
        try:
            self.directory = Path(tempfile.mkdtemp(prefix='session-', dir=self.root))
            with self.lock:
                self.state.update(phase='loading', ready=False, error='', input='', reply='', history=[], input_count=0,
                                  reply_count=0, reply_spoken=False, expression='neutral', gesture='quiet', synth_count=0, synth_bytes=0, played_bytes=0, last_input='', profile=profile, started=time.time(), alignment=False,
                                  audible_until=0, listening=False)
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
            env.update(selected.get('env', {}))
            engine = [cfg['engine'], '-m', selected['model'], '-sys', str(self.directory/'system.txt'), '-s', lg_sock]
            if selected.get('head'):
                engine += ['-mtp', selected['head']]
            self.spawn('engine', engine, env=env)
            if cfg.get('whisper_command'):
                self.spawn('whisper', cfg['whisper_command'])
            deadline = time.time()+180
            while True:
                if self.cancel.wait(.2):
                    return
                if any(p.poll() is not None for p in self.procs.values()):
                    raise RuntimeError('startup process exited; see session logs')
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
            while not self.cancel.wait(.25):
                if any(p.poll() is not None for p in self.procs.values()):
                    raise RuntimeError('pipeline process exited unexpectedly')
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
                self.state.update(ready=False, listening=False, audible_until=0, piper_config=None,
                                  expression='neutral', gesture='quiet', gesture_until=0)
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
                if kind == 'config':
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
                elif kind == 'listening':
                    st['listening'] = text == '1'
                    if st['phase'] in ('paused', 'listening'):
                        st['phase'] = 'listening' if st['listening'] else 'paused'
                elif kind == 'cut':
                    st.update(audible_until=0, expression='neutral', gesture='quiet', gesture_until=0)
                self.emit(kind, text, at=event['at'])

    def stop(self):
        with self.lifecycle:
            self.cancel.set()
            if self.thread:
                self.thread.join(timeout=20)
                if self.thread.is_alive():
                    raise RuntimeError('stop still in progress')
        return {'ok': True}


def create_app(config):
    app = Flask(__name__, template_folder=str(HERE), static_folder=str(HERE))
    app.config['MAX_CONTENT_LENGTH'] = 65536
    device = Device(config)
    key = config['key']
    session = dict(active=False, phase='idle', turns=0, error='')
    session_lock = threading.Lock()
    session_stop = threading.Event()
    session_thread = [None]

    def local(action, data=None):
        if action == 'status':
            return device.snapshot(int((data or {}).get('after', 0)))
        if action == 'start': return device.start(data)
        if action == 'stop': return device.stop()
        if action == 'volume': return device.volume(data)
        if action == 'microphone': return device.microphone(data)
        if action == 'listen': return device.listen(bool(data['enabled']))
        if action == 'endpoint-hold':
            ms = int(data['ms'])
            if not 0 <= ms <= 5000: raise ValueError('hold must be 0–5000 ms')
            device.command('endpoint-hold '+str(ms))
            return {'ok': True}
        if action == 'prompt': return device.prompt(data['text'])
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
            for side in ['peer', 'local']:
                call(side, 'stop')
                settings = dict(data[side], listen=False)
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
                    for side in ['local', 'peer']: call(side, 'listen', {'enabled': False})
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
                               name=config['name'], peer=bool(config.get('peer')), role=config.get('role', 'teacher'))

    @app.route('/api/<side>/<action>', methods=['GET', 'POST'])
    def api(side, action):
        try:
            if side not in ('local', 'peer'): raise ValueError('unknown device')
            if request.method == 'GET' and action != 'status': return jsonify(error='POST required'), 405
            if request.method == 'POST' and not request.is_json: return jsonify(error='JSON required'), 415
            return jsonify(call(side, action, request.args if request.method == 'GET' else request.get_json()))
        except Exception as exc:
            return jsonify(error=str(exc)), 400

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
        if session_thread[0] and session_thread[0].is_alive():
            session_thread[0].join(timeout=60)
        device.stop()

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
    cfg = json.loads(args.config.read_text())
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

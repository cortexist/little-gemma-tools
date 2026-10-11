"""Host-owned Wayland recording. Arming never starts audio or screen capture."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=5).strip()


def classroom_geometry(target='window', title='Voice classroom', window_id=None, crop=None):
    tree = json.loads(command(['swaymsg', '-t', 'get_tree', '-r']))
    visible = {w['id'] for w in json.loads(command(['swaymsg', '-t', 'get_workspaces', '-r'])) if w['visible']}
    matches = []
    def visit(node, output=None, workspace=None):
        if node.get('type') == 'output': output = node
        if node.get('type') == 'workspace': workspace = node['id']
        if (node.get('id') == window_id if window_id is not None else (node.get('name') or '').startswith(title)):
            if workspace in visible: matches.append((node, output))
        for child in node.get('nodes', []) + node.get('floating_nodes', []): visit(child, output, workspace)
    visit(tree)
    focused = [pair for pair in matches if pair[0].get('focused')]
    if len(focused or matches) != 1:
        raise ValueError('Open and focus the recording window on this computer, then retry.')
    window, output = (focused or matches)[0]
    rect = (output if target == 'display' else window)['rect']
    if rect['width'] < 2 or rect['height'] < 2: raise ValueError('Classroom window is not visible')
    if crop:
        if (not isinstance(crop, list) or len(crop) != 4 or
                any(type(v) not in (int, float) or not 0 <= v <= 1 for v in crop)):
            raise ValueError('Recording crop must be four fractions: x, y, width, height')
        x, y, width, height = crop
        if width <= 0 or height <= 0 or x+width > 1 or y+height > 1:
            raise ValueError('Recording crop must fit inside its window')
        rect = dict(x=rect['x']+round(x*rect['width']), y=rect['y']+round(y*rect['height']),
                    width=round(width*rect['width']), height=round(height*rect['height']))
        if rect['width'] < 2 or rect['height'] < 2: raise ValueError('Recording crop is too small')
    return f"{rect['x']},{rect['y']} {rect['width']}x{rect['height']}"


class Recording:
    def __init__(self, config, models, microphone, digital_factory=None, input_factory=None):
        self.cfg, self.models, self.microphone = config, models, microphone
        self.digital_factory = digital_factory
        self.input_factory = input_factory
        self.binary = shutil.which('wf-recorder')
        self.available = bool(self.binary and os.environ.get('WAYLAND_DISPLAY'))
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.started = threading.Event()
        self.thread = None
        self.prepared_models = None
        self.state = dict(available=self.available, phase='idle', file='', error='', started=None, asr_directory='',
                          audio_mode='room', audio_modes=['room', 'digital'] if digital_factory else ['room'])

    def snapshot(self):
        with self.lock: return dict(self.state)

    def update(self, **values):
        with self.lock: self.state.update(values)

    def geometry(self):
        return classroom_geometry(self.cfg.get('recording_target', 'window'),
                                  self.cfg.get('recording_window_title', 'Voice classroom'),
                                  self.cfg.get('recording_window_id'), self.cfg.get('recording_crop'))

    def arm(self, audio_mode='room'):
        with self.lock:
            if audio_mode not in self.state['audio_modes']: raise ValueError('Unsupported recording audio mode')
            if not self.available: raise ValueError('wf-recorder and a Wayland session are required on this host')
            if self.thread and self.thread.is_alive(): raise ValueError('Recording is already armed or running')
            # Validate the intended window without capturing it. Resolve again
            # at start, because loading the models may take several seconds.
            self.geometry()
            self.cancel.clear(); self.started.clear(); self.prepared_models = None
            self.state.update(phase='armed', file='', error='', started=None, audio_mode=audio_mode, asr_directory='')
            self.thread = threading.Thread(target=self.work, daemon=True)
            self.thread.start()
        return self.snapshot()

    def models_ready(self, models):
        """Coordinator barrier: start recording before the teacher's opening."""
        with self.lock:
            if self.state['phase'] not in ('armed', 'starting'): return
            self.prepared_models = models
        self.started.wait(timeout=30)

    def stop(self, wait=False):
        with self.lock:
            thread = self.thread
            if thread and thread.is_alive():
                self.state['phase'] = 'stopping'
                self.cancel.set()
        if wait and thread: thread.join(timeout=35)
        return self.snapshot()

    def destination(self, models):
        root = Path(self.cfg.get('recording_directory', '~/Videos/screen_recordings')).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        names = [re.sub(r'[^a-z0-9_-]', '', str(m).lower())[:64] or 'model' for m in models]
        prefix = re.sub(r'[^a-z0-9_-]', '', self.cfg.get('recording_prefix', 'voice-classroom')) or 'voice-classroom'
        if self.digital_factory: prefix += '-'+self.state['audio_mode']
        stem = prefix + '-' + '-'.join(names) + '-' + datetime.now().strftime('%Y%m%d-%H%M')
        for number in range(1, 1000):
            path = root/(stem + (f'-{number}' if number > 1 else '') + '.mp4')
            try:
                with path.open('xb'): pass
                return path
            except FileExistsError: continue
        raise ValueError('Too many recordings with the same name')

    def audio_source(self, modules):
        prepare = self.cfg.get('recording_prepare_command')
        if prepare:
            if not isinstance(prepare, list) or not all(isinstance(arg, str) for arg in prepare):
                raise ValueError('recording_prepare_command must be an argument list')
            try:
                subprocess.run(prepare, check=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, timeout=15)
            except (subprocess.SubprocessError, OSError) as exc:
                detail = getattr(exc, 'stderr', '') or str(exc)
                raise RuntimeError('Recording audio preparation failed: '+detail.strip()) from exc
        source = self.cfg.get('recording_audio_source')
        if source:
            sources = json.loads(command(['pactl', '-f', 'json', 'list', 'sources']))
            if not any(s['name'] == source for s in sources):
                raise ValueError('Configured recording audio source is unavailable: '+source)
            relay = self.cfg.get('recording_local_relay')
            if relay:
                if source != relay['sink']+'.monitor':
                    raise ValueError('Local recording relay must feed the configured mix')
                volume = int(relay.get('volume', 65536))
                if not 0 <= volume <= 65536:
                    raise ValueError('Recording relay volume must be 0–65536')
                module = command(['pactl', 'load-module', 'module-loopback',
                                  'source='+self.cfg['sink']+'.monitor', 'sink='+relay['sink'],
                                  'latency_msec=20', 'source_dont_move=true', 'sink_dont_move=true'])
                modules.append(module)
                # Own a fresh native relay for this recording; do not inherit a
                # long-running shell pipe's queued audio or restored stream gain.
                for kind, action, gain in [('sink-inputs', 'set-sink-input-volume', volume),
                                           ('source-outputs', 'set-source-output-volume', 65536)]:
                    for _ in range(20):
                        streams = json.loads(command(['pactl', '-f', 'json', 'list', kind]))
                        owned = [s for s in streams if str(s.get('owner_module')) == module]
                        if owned:
                            for stream in owned:
                                command(['pactl', action, str(stream['index']), str(gain)])
                            break
                        time.sleep(.05)
                    else:
                        raise RuntimeError('Recording loopback stream did not appear')
            return source  # Borrow the existing mix; never unload its modules.
        pair = self.cfg.get('recording_room_channels')
        if pair is not None:
            # Select simultaneous physical channels explicitly. The first two
            # USB channels on a DSP array need not be a matched stereo pair.
            if (not isinstance(pair, list) or len(pair) != 2 or
                    any(type(i) is not int or i < 0 for i in pair) or pair[0] == pair[1]):
                raise ValueError('Room stereo needs two distinct zero-based microphone channel indices')
            gain = self.cfg.get('recording_room_gain_db', 0)
            if type(gain) not in (int, float) or not math.isfinite(gain) or not -48 <= gain <= 48:
                raise ValueError('Room microphone gain must be between -48 and 48 dB')
            master = self.microphone()
            sources = json.loads(command(['pactl', '-f', 'json', 'list', 'sources']))
            matches = [s for s in sources if s['name'] == master]
            if len(matches) != 1: raise ValueError('Room microphone unavailable or ambiguous')
            mapping = matches[0]['channel_map'].split(',')
            if max(pair) >= len(mapping):
                raise ValueError('Room stereo microphone channels unavailable; use the configured multichannel capture profile')
            source = 'voice_room_stereo_'+secrets.token_hex(6)
            modules.append(command(['pactl', 'load-module', 'module-remap-source',
                'source_name='+source, 'master='+master, 'channels=2',
                'master_channel_map='+','.join(mapping[i] for i in pair),
                'channel_map=front-left,front-right', 'remix=no',
                # PipeWire otherwise clamps applied gain to 10x (+20 dB),
                # even when Pulse reports a higher requested volume. Permit
                # the validated +48 dB range on this recording-owned source.
                'source_properties=channelmix.max-volume=256']))
            # Common gain preserves the pair's relative levels. No speaker
            # monitor is added: raw microphones already hear local playback.
            command(['pactl', 'set-source-volume', source, str(round(65536*10**(gain/60)))])
            return source
        sink = 'voice_classroom_record_' + secrets.token_hex(6)
        def module(name, *args):
            modules.append(command(['pactl', 'load-module', name, *args]))
        module('module-null-sink', 'sink_name='+sink, 'channels=1', 'channel_map=mono')
        # Standalone fallback: local speaker plus processed microphone channel 0.
        for source in (self.cfg['sink']+'.monitor', self.microphone()):
            module('module-loopback', 'source='+source, 'sink='+sink,
                   'channels=1', 'channel_map=front-left', 'remix=no',
                   'latency_msec=20', 'source_dont_move=true', 'sink_dont_move=true')
        return sink+'.monitor'

    def work(self):
        proc, path, modules, error, digital, inputs = None, None, [], '', None, None
        try:
            while not self.cancel.is_set():
                with self.lock: models = self.prepared_models
                models = models or self.models()
                if models: break
                self.cancel.wait(.2)
            if self.cancel.is_set(): return
            self.update(phase='starting')
            geometry = self.geometry()
            path = self.destination(models)
            self.update(file=str(path).replace(str(Path.home())+'/', '~/', 1))
            if self.state['audio_mode'] == 'digital':
                digital = self.digital_factory()
                source = digital.start(self.cancel, log_dir=path.with_suffix('.audio'))
            else:
                source = self.audio_source(modules)
            if self.cancel.is_set(): return
            if self.input_factory:
                inputs = self.input_factory()
                inputs.start(self.cancel, path)
                self.update(asr_directory=str(inputs.directory).replace(str(Path.home())+'/', '~/', 1))
            if self.cancel.is_set(): return
            log_path = Path(self.cfg['runtime_dir']).expanduser()/'recording.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('wb') as log:
                proc = subprocess.Popen([self.binary, '-g', geometry, '-f', str(path), '-y',
                    '-c', 'libx264', '-p', 'preset=ultrafast', '-p', 'crf=23',
                    '-x', 'yuv420p', '-r', '30', '-F', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '--audio='+source, '-C', 'aac', '-P', 'b=192000', '-R', '48000'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            deadline = time.monotonic()+10
            while not self.cancel.is_set():
                if digital: digital.check()
                if inputs: inputs.check()
                if proc.poll() is not None: raise RuntimeError('wf-recorder exited; see recording.log on this host')
                if path.stat().st_size: break
                if time.monotonic() > deadline: raise RuntimeError('Recorder did not produce video within 10 seconds')
                self.cancel.wait(.1)
            if self.cancel.is_set(): return
            self.update(phase='recording', started=time.time())
            if inputs: inputs.video_started(self.state['started'])
            self.started.set()
            while not self.cancel.wait(.2):
                if digital: digital.check()
                if inputs: inputs.check()
                if proc.poll() is not None: raise RuntimeError('Recorder stopped unexpectedly; see recording.log on this host')
        except Exception as exc:
            error = str(exc)
        finally:
            if inputs:
                self.update(phase='stopping')
                try: inputs.close()
                except Exception as exc: error = error or 'Could not finalize ASR tracks: '+str(exc)
            if proc:
                self.update(phase='stopping')
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)  # MP4 trailer must be written.
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        error = error or 'Recorder did not finalize; the video may be incomplete'
                        proc.kill(); proc.wait(timeout=5)
                if proc.returncode not in (0, 130, -signal.SIGINT):
                    error = error or 'Recorder failed; see recording.log on this host'
            if digital:
                try: digital.close()
                except Exception as exc: error = error or 'Could not close digital recording: '+str(exc)
            for module in reversed(modules):
                try: command(['pactl', 'unload-module', module])
                except Exception as exc: error = error or 'Could not remove recording audio mix: '+str(exc)
            if path and path.exists() and path.stat().st_size == 0:
                path.unlink(); self.update(file='')
            self.update(phase='error' if error else 'idle', error=error)
            self.started.set()

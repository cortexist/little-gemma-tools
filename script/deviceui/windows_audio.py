"""Native Windows voice node: device-owned capture and WASAPI playback."""
import math
import os
from pathlib import Path
import queue
import re
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from reply_stream import ReplyStream
from speech_names import piper_names
from speech_titles import normalize_titles
from windows_capture import CapturePCM, select_input, transcribe
from streaming_asr import StreamingUtterance, parse_vtt, asr_prompt


def select_output(sd, name):
    """Require one exact WASAPI endpoint; never fall back to a default device."""
    apis = sd.query_hostapis()
    matches = [i for i, d in enumerate(sd.query_devices())
               if d['name'] == name and d['max_output_channels'] >= 2
               and apis[d['hostapi']]['name'] == 'Windows WASAPI']
    if len(matches) != 1:
        raise ValueError(f'Expected one WASAPI output named {name!r}; found {len(matches)}')
    return matches[0]


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count-len(data))
        if not chunk:
            if not data:
                return b''
            raise RuntimeError('Truncated Piper mux frame')
        data.extend(chunk)
    return bytes(data)


class WindowsDeviceMixin:
    def __init__(self, config):
        self.prompts = queue.Queue(maxsize=1)
        self.turn_active = False
        self.capture_epoch = 0
        self.endpoint_until = 0
        self.input_prefix = ''
        self.audio_shutdown = threading.Event()
        super().__init__(config)

    def endpoint(self, which, data=None):
        # COM interfaces stay within their owning thread/apartment.
        import comtypes
        from pycaw.pycaw import AudioUtilities
        comtypes.CoInitialize()
        try:
            endpoints = AudioUtilities.GetAllDevices(data_flow=0 if which == 'sink' else 1,
                                                     device_state=1)
            matches = [d for d in endpoints if d.id == self.cfg[which+'_id']
                       and d.FriendlyName == self.cfg[which]]
            if len(matches) != 1:
                raise ValueError(f'Configured Windows {which} endpoint is unavailable')
            volume = matches[0].EndpointVolume
            if data:
                if 'percent' in data:
                    value = float(data['percent'])
                    if not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError('volume must be 0–100 percent')
                if 'mute' in data and not isinstance(data['mute'], bool):
                    raise ValueError('mute must be a boolean')
                if 'percent' in data:
                    volume.SetMasterVolumeLevelScalar(value/100, None)
                if 'mute' in data:
                    volume.SetMute(data['mute'], None)
            result = dict(percent=round(volume.GetMasterVolumeLevelScalar()*100),
                          mute=bool(volume.GetMute()))
            if which == 'source':
                # Windows endpoint mute, not a claim about firmware/hardware mute.
                result.update(hardware_mute=result['mute'], source=self.cfg['source'],
                              mute_scope='Windows endpoint', capture_enabled=bool(self.cfg.get('capture_enabled')))
            return result
        finally:
            comtypes.CoUninitialize()

    def audio_poll(self):
        while not self.audio_shutdown.is_set():
            try:
                volume, mic = self.endpoint('sink'), self.endpoint('source')
                with self.lock:
                    self.state.update(volume=volume, microphone=mic, audio_error='')
            except Exception as exc:
                with self.lock:
                    self.state['audio_error'] = str(exc)
            self.audio_shutdown.wait(.5)

    def volume(self, data):
        return self.endpoint('sink', data)

    def microphone(self, data):
        return self.endpoint('source', data)

    def listen(self, enabled, *, finishing=False):
        if enabled and not getattr(self, 'cfg', {}).get('capture_enabled'):
            raise ValueError('Microphone conversation is not configured on this Windows node')
        with self.lock:
            if getattr(self, 'input_prefix', '') and not self.turn_active:
                raise ValueError('Input is already committed to an open turn; use Stop to discard it safely')
            if enabled and self.state.get('turn_limit') and self.state['input_count'] >= self.state['turn_limit']:
                raise ValueError('turn limit reached')
            self.capture_epoch = getattr(self, 'capture_epoch', 0)+1
            self.state.update(listening=enabled, finishing=finishing)
            if not self.turn_active and self.state['ready']:
                self.state['phase'] = 'listening' if enabled else 'finishing' if finishing else 'paused'
        self.emit('listening', '1' if enabled else '0')
        return {'ok': True}

    def command(self, text):
        if text.startswith('endpoint-hold '):
            ms = int(text[14:])
            if not 0 <= ms <= 5000:
                raise ValueError('endpoint hold must be 0–5000 ms')
            with self.lock:
                self.endpoint_until = time.monotonic()+ms/1000
            return
        if not text.startswith('prompt '):
            raise ValueError('Unsupported Windows pipeline command')
        with self.lock:
            if getattr(self, 'input_prefix', ''):
                raise ValueError('Microphone input is already committed; finish the utterance or Stop')
            text = ' '.join(text[7:].split())
            self.admit(text, text)

    def admit(self, text, tail, asr=None):
        """Reserve one completed input, not one turn per ASR prefix."""
        if not self.state['ready'] or self.turn_active or self.state['phase'] not in ('paused', 'listening', 'hearing'):
            raise ValueError('pipeline is not ready or is busy')
        if self.state.get('turn_limit') and self.state['input_count'] >= self.state['turn_limit']:
            raise ValueError('turn limit reached')
        if getattr(self, 'input_prefix', '')+tail != text:
            raise ValueError('Committed input and final suffix do not reconstruct the transcript')
        self.prompts.put_nowait((text, tail, asr))
        self.input_prefix = ''
        self.state['phase'] = 'thinking'
        self.state['input_count'] += 1
        self.turn_active = True
        self.capture_epoch = getattr(self, 'capture_epoch', 0)+1
        if self.state.get('turn_limit') and self.state['input_count'] >= self.state['turn_limit']:
            self.state.update(listening=False, finishing=True)

    def commit_input(self, piece):
        # Called with the device lock held after checking the capture epoch.
        if not piece:
            return
        data = piece.encode('utf-8')
        if len(data) > 4096:
            raise ValueError('ASR commit exceeds the engine text-frame limit')
        self.input_client.stdin.write(b'\x01T'+struct.pack('<HHI', 0, 0, len(data))+data)
        self.input_client.stdin.flush()
        self.input_prefix += piece
        self.state['input'] = self.input_prefix
        self.state['input_commits'] += 1
        self.emit('input', piece)
        self.emit('input_commit', piece, bytes=len(data), prefix_chars=len(self.input_prefix))

    def snapshot(self, after=0):
        with self.lock:
            result = super().snapshot(after)
            result.setdefault('capabilities', {})['capture'] = bool(getattr(self, 'cfg', {}).get('capture_enabled'))
            result['capabilities']['incremental_input'] = bool(getattr(self, 'cfg', {}).get('engine_client'))
            # The Linux controller infers completion from queue timestamps. On
            # Windows the blocking WASAPI stream owns the actual drain boundary.
            if result['state']['ready'] and self.turn_active:
                phase = 'speaking' if time.time() < self.state['audible_until'] else 'thinking'
                result['state']['phase'] = self.state['phase'] = phase
            return result

    def capture(self, blocks, errors):
        import numpy as np
        def new_utterance():
            return StreamingUtterance(level=self.cfg.get('vad_level', 200),
                                      silence=self.cfg.get('capture_silence', .8))
        utterance = new_utterance()
        epoch = -1
        future = pending = owner = None
        diagnostic = None
        def close_diagnostic(complete=False):
            nonlocal diagnostic
            if diagnostic is not None:
                diagnostic.close()
                diagnostic = None
                if complete:
                    (self.directory/'capture-pending.wav').replace(self.directory/'last-capture.wav')
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='whisper')
        def infer(request):
            began = time.monotonic()
            self.emit('asr_start', request['mode'], seconds=len(request['pcm'])/32000)
            acoustic = getattr(self, 'sound_input', False) and request['mode'] != 'partial'
            response = transcribe(self.cfg['whisper_url'], request['pcm'],
                                  response_format='vtt', prompt=request['prompt'], sound=acoustic)
            result = parse_vtt(response['vtt'] if acoustic else response)
            if acoustic:
                result.update(sound_event=response.get('sound_event'),
                              sound_timings=response.get('timings'), sound_error=response.get('sound_error'))
            result['elapsed'] = time.monotonic()-began
            self.emit('asr', ' '.join(result['words']), mode=request['mode'],
                      seconds=len(request['pcm'])/32000, elapsed=result['elapsed'],
                      **({k:result.get(k) for k in ('sound_event', 'sound_timings', 'sound_error')} if acoustic else {}))
            return result
        try:
            while not self.cancel.is_set():
                try:
                    captured, block_epoch, pcm, status = blocks.get(timeout=.2)
                except queue.Empty:
                    continue
                if status:
                    raise RuntimeError('Microphone stream overflow: '+str(status))
                with self.lock:
                    enabled = (self.capture_enabled() if hasattr(self, 'capture_enabled') else
                               self.state['listening'] and not self.turn_active)
                    current = self.capture_epoch
                    held = time.monotonic() < self.endpoint_until
                if current != epoch:
                    close_diagnostic()
                    utterance = new_utterance()
                    epoch = current
                    if hasattr(self, 'reset_capture_audio'): self.reset_capture_audio()
                eligible = enabled and block_epoch == epoch and time.monotonic()-captured <= .5
                record_input = getattr(self, 'record_input_block', None)
                if record_input is not None: record_input(captured, pcm, eligible)
                if not eligible:
                    if enabled and block_epoch == epoch and utterance.active and time.monotonic()-captured > .5:
                        raise RuntimeError('Capture fell behind during incremental ASR; refusing to drop speech')
                    continue
                values = np.frombuffer(pcm, dtype='<i2').astype(np.float32)
                rms = float(np.sqrt(np.mean(values*values)))
                with self.lock:
                    self.state['capture_rms'] = round(rms, 1)
                    self.state['capture_peak_rms'] = max(round(rms, 1), self.state.get('capture_peak_rms', 0))
                utterance.feed(pcm, rms)
                self.state['capture_active'] = utterance.active
                if hasattr(self, 'capture_audio'): self.capture_audio(utterance, pcm)
                if self.cfg.get('capture_diagnostics') and utterance.active:
                    if diagnostic is None:
                        import wave
                        diagnostic = wave.open(str(self.directory/'capture-pending.wav'), 'wb')
                        diagnostic.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                        diagnostic.writeframesraw(utterance.audio)
                    else:
                        diagnostic.writeframesraw(pcm)
                with self.lock:
                    if not self.turn_active:
                        self.state['phase'] = 'hearing' if utterance.active else 'listening'
                if future is not None and future.done():
                    result = future.result()
                    future = None
                    if owner is utterance:
                        piece = utterance.apply(result, pending)
                        with self.lock:
                            if epoch == self.capture_epoch and self.state['listening'] and not self.cancel.is_set():
                                self.commit_input(piece)
                if future is None and utterance.ready(held):
                    text, tail = utterance.finish()
                    close_diagnostic(complete=bool(text))
                    with self.lock:
                        if self.cancel.is_set() or epoch != self.capture_epoch or not self.state['listening']:
                            continue
                        acoustic = getattr(self, 'sound_input', False)
                        if text or (acoustic and (utterance.sound_event is not None or
                                                getattr(self, 'passive', False) and utterance.sound_error)):
                            self.state['input'] = text
                            if tail: self.emit('input', tail)
                            self.admit(text, tail, dict(asr_elapsed=utterance.last_elapsed,
                                       asr_mode=utterance.last_mode, asr_passes=utterance.passes,
                                       **(dict(sound_event=utterance.sound_event, sound_timings=utterance.sound_timings,
                                               sound_error=utterance.sound_error) if acoustic else {})))
                        else:
                            self.state['phase'] = 'listening'
                            if acoustic:
                                # No words or usable observation: do not carry
                                # this event's identity/audio into the next one.
                                self.capture_epoch += 1
                    utterance = new_utterance()
                elif future is None:
                    pending = utterance.request(held)
                    if pending:
                        with self.lock:
                            name_hint = getattr(self, 'asr_name_hint', '')
                        pending['prompt'] = asr_prompt(name_hint, pending['prompt'])
                        owner = utterance
                        future = worker.submit(infer, pending)
        except Exception as exc:
            if not self.cancel.is_set():
                errors.put(exc)
        finally:
            close_diagnostic()
            worker.shutdown(wait=True, cancel_futures=True)

    def spawn(self, name, argv, **kwargs):
        with (self.directory/(name+'.log')).open('ab') as log:
            proc = subprocess.Popen(argv, stderr=log, stdout=kwargs.pop('stdout', log),
                                    creationflags=subprocess.CREATE_NO_WINDOW, **kwargs)
        with self.lock:
            self.procs[name] = proc
        self.emit('process', name, argv=argv, pid=proc.pid)
        return proc

    def terminate_owned(self):
        with self.lock:
            processes = list(self.procs.values())
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        with self.lock:
            for name, proc in list(self.procs.items()):
                if any(proc is owned for owned in processes) and proc.poll() is not None:
                    self.procs.pop(name)

    def work(self, profile, prompt, listen, turn_limit=0):
        from server import engine_command
        import sounddevice as sd
        capture_stream = capture_thread = None
        try:
            if listen and not self.cfg.get('capture_enabled'):
                raise ValueError('Microphone conversation is not configured')
            self.directory = Path(tempfile.mkdtemp(prefix='s-', dir=self.root))
            self.prompts = queue.Queue(maxsize=1)
            with self.lock:
                self.procs.clear()
                self.turn_active = False
                self.input_prefix = ''
                self.capture_epoch += 1
                self.endpoint_until = 0
                self.state.update(phase='loading', ready=False, error='', input='', reply='', history=[],
                                  input_count=0, reply_count=0, reply_spoken=False, synth_count=0,
                                  synth_bytes=0, played_bytes=0, alignment=False, audible_until=0,
                                  listening=False, finishing=False, turn_limit=turn_limit, profile=profile, started=time.time())
                self.state.update(capture_rms=0, capture_peak_rms=0, capture_blocks=0, input_commits=0)
            self.emit('reset')
            selected = self.cfg['profiles'][profile]
            system = self.directory/'system.txt'
            system.write_text(prompt, encoding='utf-8')
            sock = str(self.directory/'lg.sock')
            if len(sock.encode()) >= 108:
                raise ValueError('Windows AF_UNIX path is too long; shorten runtime_dir')
            output = select_output(sd, self.cfg['sink'])
            # Validate the endpoint before paying the model load cost.
            rate = self.cfg.get('sample_rate', 22050)
            sd.check_output_settings(device=output, channels=2, dtype='int16', samplerate=rate,
                                     extra_settings=sd.WasapiSettings(auto_convert=True))
            env = dict(os.environ, **self.cfg.get('env', {}), **selected.get('env', {}))
            self.synth = self.spawn('piper', self.synthesis_command()+['--mux-boundaries'],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            self.frames = queue.Queue(maxsize=64)
            self.synth_sent = self.synth_completed = 0
            self.synth_reader = threading.Thread(target=self.read_synthesis, daemon=True)
            self.synth_reader.start()
            capture_errors = queue.Queue()
            if self.cfg.get('capture_enabled'):
                channels = self.cfg.get('capture_channels', 2)
                source = select_input(sd, self.cfg['source'], self.cfg.get('capture_hostapi', 'Windows WASAPI'), channels)
                capture_rate = int(self.cfg.get('capture_rate', sd.query_devices(source)['default_samplerate']))
                converter = CapturePCM(capture_rate, channels)
                blocks = queue.Queue(maxsize=150)
                def microphone(data, frames, timing, status):
                    pcm = converter.convert(data)
                    self.state['capture_blocks'] += 1
                    try:
                        blocks.put_nowait((time.monotonic(), self.capture_epoch, pcm, status))
                    except queue.Full:
                        capture_errors.put(RuntimeError('Microphone queue overflow; refusing to drop speech'))
                capture_stream = sd.RawInputStream(device=source, samplerate=capture_rate, channels=channels,
                    dtype='int16', blocksize=capture_rate//50, callback=microphone)
                whisper = self.spawn('whisper', self.cfg['whisper_command'], env=env)
                deadline = time.monotonic()+90
                while not self.cancel.wait(.2):
                    if whisper.poll() is not None:
                        raise RuntimeError('Whisper exited; see whisper.log')
                    try:
                        with urllib.request.urlopen(self.cfg['whisper_health'], timeout=1) as response:
                            if response.status == 200: break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Whisper load timed out')
                if self.cancel.is_set(): return
            engine = self.spawn('engine', engine_command(self.cfg, selected, system, sock), env=env)
            deadline = time.monotonic()+180
            while not self.cancel.wait(.2):
                if engine.poll() is not None:
                    raise RuntimeError('Engine exited while loading; see engine.log')
                if 'listening on ' in (self.directory/'engine.log').read_text(errors='replace'):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Engine load timed out')
            if self.cancel.is_set():
                return
            bridge = self.cfg.get('engine_client')
            if not bridge or not Path(bridge).is_file():
                raise ValueError('Windows incremental input requires a built binary-safe socket_cat engine_client')
            client = self.spawn('client', [bridge, sock],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
            self.input_client = client
            if self.cancel.wait(.4):
                return
            if client.poll() is not None:
                raise RuntimeError('Engine client failed to connect')
            replies = queue.Queue()
            def reader():
                try:
                    while True:
                        part = client.stdout.read1(4096)
                        if not part:
                            break
                        replies.put(part)
                finally:
                    replies.put(None)
            threading.Thread(target=reader, daemon=True).start()
            deadline = time.monotonic()+90
            while not self.cancel.is_set():
                try:
                    frame = self.frames.get(timeout=.2)
                except queue.Empty:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Piper load timed out')
                    continue
                if isinstance(frame, Exception): raise frame
                if frame is None or frame[0] != b'C':
                    raise RuntimeError('Piper did not provide its stream configuration')
                self.speech_rate = self.synthesis_config(frame[1])
                break
            if self.cancel.is_set(): return
            with self.lock:
                self.state.update(ready=True, phase='paused')
            if capture_stream:
                capture_thread = threading.Thread(target=self.capture, args=(blocks, capture_errors), daemon=True)
                capture_thread.start()
                # Loading models can take minutes. Start producing microphone
                # blocks only once their consumer is running, never into the
                # bounded queue while waiting for Whisper/engine/Piper.
                capture_stream.start()
            self.listen(listen)
            self.emit('ready', 'Windows voice pipeline ready' if capture_stream else 'Windows playback ready')
            while not self.cancel.is_set():
                if engine.poll() is not None or client.poll() is not None or self.synth.poll() is not None:
                    raise RuntimeError('Engine or Piper process exited')
                if not capture_errors.empty():
                    raise capture_errors.get_nowait()
                if capture_stream and whisper.poll() is not None:
                    raise RuntimeError('Whisper process exited')
                if capture_stream and not capture_stream.active:
                    raise RuntimeError('Microphone stream stopped unexpectedly')
                try:
                    text = self.prompts.get(timeout=.2)
                except queue.Empty:
                    continue
                self.turn(text, client, replies, output, env)
        except Exception as exc:
            if not self.cancel.is_set():
                with self.lock:
                    self.state.update(error=str(exc), phase='error')
                self.emit('error', str(exc))
        finally:
            was_cancelled = self.cancel.is_set()
            self.cancel.set()
            if capture_stream:
                capture_stream.abort()
                capture_stream.close()
            self.terminate_owned()
            if getattr(self, 'synth_reader', None):
                self.synth_reader.join(timeout=2)
            if capture_thread:
                capture_thread.join(timeout=35)
            with self.lock:
                self.state.update(ready=False, listening=False, audible_until=0)
                self.turn_active = False
                if was_cancelled:
                    self.state['phase'] = 'stopped'
            self.emit('stopped')

    def read_synthesis(self):
        def put(frame):
            while not self.cancel.is_set():
                try:
                    self.frames.put(frame, timeout=.2)
                    return
                except queue.Full:
                    pass
        try:
            while not self.cancel.is_set():
                head = read_exact(self.synth.stdout, 5)
                if not head:
                    break
                size = struct.unpack('<I', head[1:])[0]
                if size > 16*1024*1024:
                    raise ValueError('Invalid Piper frame length')
                payload = read_exact(self.synth.stdout, size)
                if len(payload) != size:
                    raise RuntimeError('Truncated Piper payload')
                put((head[:1], payload))
        except Exception as exc:
            put(exc)
        finally:
            put(None)

    def synthesis_config(self, payload):
        config = dict(line.split('=', 1) for line in payload.decode().splitlines() if '=' in line)
        rate = int(config['rate'])
        if int(config['width']) != 2 or int(config['channels']) != 1 or not 8000 <= rate <= 192000:
            raise ValueError('Expected mono signed 16-bit Piper audio')
        self.emit('config', payload.decode())
        return rate

    def stream_reply(self, text, client, replies, finished, errors, tail=None):
        """Feed clauses while playback drains on the worker thread."""
        reply = ReplyStream()
        first = True
        try:
            client.stdin.write(((text if tail is None else tail)+'\n').encode('utf-8'))
            client.stdin.flush()
            progress = time.monotonic()
            while not reply.done and not self.cancel.is_set():
                if time.monotonic()-progress > 90:
                    raise TimeoutError('No model output for 90 seconds')
                try:
                    part = replies.get(timeout=.2)
                except queue.Empty:
                    continue
                if part is None:
                    raise RuntimeError('Engine disconnected before completing its reply')
                if first:
                    part = part.lstrip(b' \t\r\n')  # Previous client turn's framing newline.
                    if not part: continue
                progress = time.monotonic()
                if first:
                    self.emit('model_first_token')
                    first = False
                clauses = reply.feed(part)
                with self.lock:
                    self.state['reply'] = reply.text.strip()
                    self.state['model_bytes'] = reply.size
                for clause in clauses:
                    self.emit('clause', clause)
                    rendered = piper_names(normalize_titles(clause), self.cfg.get('name_pronunciations', {}))
                    self.synth.stdin.write((rendered+'\n').encode('utf-8'))
                    self.synth.stdin.flush()
                    self.synth_sent += 1
            if not self.cancel.is_set():
                with self.lock:
                    self.state['reply_count'] += 1
                    self.state['history'] = (self.state['history']+[dict(heard=text, said=reply.text.strip())])[-4:]
                self.emit('reply', reply.text.strip())
                self.emit('reply_done')
        except Exception as exc:
            errors.put(exc)
        finally:
            finished.set()

    def turn(self, text, client, replies, output, env):
        import numpy as np
        text, tail, asr = text if isinstance(text, tuple) else (text, text, None)
        with self.lock:
            self.state.update(last_input=text, reply='', model_bytes=0, reply_spoken=False, phase='thinking')
        self.emit('input_done', text, **(asr or {}))
        finished, errors = threading.Event(), queue.Queue()
        initial_sent = self.synth_sent
        producer = threading.Thread(target=self.stream_reply, args=(text, client, replies, finished, errors, tail), daemon=True)
        producer.start()
        stream = None
        rate = self.speech_rate
        played = 0
        progress = time.monotonic()
        try:
            while not self.cancel.is_set():
                if not errors.empty(): raise errors.get_nowait()
                if finished.is_set() and self.synth_completed == self.synth_sent:
                    break
                if time.monotonic()-progress > 90:
                    raise TimeoutError('No speech synthesis progress for 90 seconds')
                try:
                    frame = self.frames.get(timeout=.1)
                except queue.Empty:
                    # Generation has its own watchdog; there is no pending
                    # synthesis until a complete clause has been submitted.
                    if self.synth_completed == self.synth_sent and not finished.is_set():
                        progress = time.monotonic()
                    continue
                if frame is None:
                    raise RuntimeError('Piper exited; see piper.log')
                if isinstance(frame, Exception): raise frame
                kind, payload = frame
                progress = time.monotonic()
                if kind == b'D':
                    completed = int(payload)
                    if completed != self.synth_completed+1:
                        raise RuntimeError('Piper synthesis boundary out of order')
                    self.synth_completed = completed
                elif kind == b'A':
                    self.state['alignment'] = True
                    self.emit('alignment', f'{self.state["synth_bytes"]}\t{rate}\n'+payload.decode())
                elif kind == b'P':
                    if len(payload) % 2:
                        raise ValueError('Misaligned Piper PCM')
                    if not payload: continue
                    if stream is None:
                        self.emit('speech_first_pcm')
                        if hasattr(self, 'open_output'):
                            stream = self.open_output(output, rate)
                        else:
                            import sounddevice as sd
                            stream = sd.RawOutputStream(device=output, samplerate=rate, channels=2,
                                dtype='int16', extra_settings=sd.WasapiSettings(auto_convert=True))
                        stream.start()
                    pcm = np.frombuffer(payload, dtype='<i2')
                    start = max(time.time()+stream.latency, self.state['audible_until'])
                    offset = self.state['played_bytes']
                    underflow = stream.write(np.repeat(pcm, 2).tobytes())
                    duration = len(pcm)/rate
                    with self.lock:
                        self.state['synth_bytes'] += len(payload)
                        self.state['played_bytes'] += len(payload)
                        self.state.update(phase='speaking', reply_spoken=True, audible_until=start+duration)
                    played += len(payload)
                    self.emit('play', f'{start}\t{offset}\t{len(payload)}\t{rate}', underflow=bool(underflow))
                else:
                    raise ValueError('Unexpected Piper frame: '+repr(kind))
            if not errors.empty(): raise errors.get_nowait()
            if not self.cancel.is_set() and self.synth_sent > initial_sent and not played:
                raise RuntimeError('Piper completed without producing audio')
        except Exception:
            # Also wake a producer blocked on the engine or synthesis pipes.
            self.terminate_owned()
            raise
        finally:
            if stream is not None:
                if self.cancel.is_set(): stream.abort()
                else: stream.stop()  # Drain the actual WASAPI queue, not a timer.
                stream.close()
            producer.join(timeout=2)
        if self.cancel.is_set(): return
        with self.lock:
            self.state['synth_count'] += 1
            self.turn_active = False
            self.capture_epoch += 1
            self.state.update(phase='listening' if self.state['listening'] else 'finishing'
                              if self.state.get('finishing') else 'paused', audible_until=0)
        self.emit('synth_done', str(self.state['synth_bytes']))

    def stop(self):
        # Wake blocked pipe reads before joining the worker.
        self.cancel.set()
        self.terminate_owned()
        return super().stop()

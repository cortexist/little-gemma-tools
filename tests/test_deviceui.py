"""HTTP boundary and volume-unit regression tests; no audio devices required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'script/deviceui/server.py'
spec = importlib.util.spec_from_file_location('deviceui', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = dict(name='test', key='test-key', runtime_dir=self.tmp.name, sink='test-sink', source='test-source',
                        profiles={'test': {'model': '/model.gguf'}}, whisper_model='/asr', voice='/tts')
        with patch.object(module.Device, 'audio_poll'):
            self.app = module.create_app(self.cfg)
        self.client = self.app.test_client()
        self.headers = {'X-Voice-Key': 'test-key'}

    def tearDown(self):
        self.tmp.cleanup()

    def test_question_boundary_is_opt_in_and_keeps_selected_model(self):
        cfg = dict(engine='/engine', end_on_question=True)
        selected = dict(model='/12b.gguf', head='/mtp.gguf')
        expected = ['/engine', '-m', '/12b.gguf', '-sys', '/system', '-s', '/socket',
                    '-mtp', '/mtp.gguf']
        self.assertEqual(module.engine_command(cfg, selected, '/system', '/socket'),
                         expected + ['-end-on-question'])
        cfg.pop('end_on_question')
        self.assertEqual(module.engine_command(cfg, selected, '/system', '/socket'), expected)
        self.assertNotIn('-mtp', module.engine_command(cfg, dict(model='/e4b.gguf'), '/system', '/socket'))

    def test_control_requires_key_and_same_origin(self):
        self.assertEqual(self.client.post('/api/local/volume', json={'percent':82}).status_code, 403)
        self.assertEqual(self.client.post('/api/local/volume', json={'percent':82},
            headers=dict(self.headers, Origin='https://unrelated.example')).status_code,403)
        self.assertEqual(self.client.get('/api/local/stop',headers=self.headers).status_code,405)

    def test_conversation_rejects_unbounded_or_oversized_limits(self):
        self.cfg['peer'] = {'url': 'http://unused', 'key': 'unused'}
        for seconds, turns in [(0, 12), (180, 0), (180, 49)]:
            response = self.client.post('/api/conversation', headers=self.headers,
                json={'topic': 'Midwest', 'seconds': seconds, 'turns': turns,
                      'local': {'system_prompt': 'Teacher'}, 'peer': {'system_prompt': 'Student'}})
            self.assertEqual(response.status_code, 400)
            self.assertIn('1–48', response.json['error'])

    def test_metadata_expires_after_playback_and_gesture_duration(self):
        device = self.app.device
        device.state.update(expression='happy', gesture='nod', gesture_until=12,
                            input_count=1, reply_count=1, synth_count=1,
                            synth_bytes=100, played_bytes=100, audible_until=15)
        with patch.object(module.time, 'time', return_value=13):
            state = device.snapshot()['state']
            self.assertEqual(state['gesture'], 'quiet')
            self.assertEqual(state['expression'], 'happy')
        with patch.object(module.time, 'time', return_value=16):
            self.assertEqual(device.snapshot()['state']['expression'], 'neutral')
        # Synthesis is still producing the turn: an interim playback gap is
        # insufficient to end the expression.
        device.state.update(expression='sad', synth_count=0)
        with patch.object(module.time, 'time', return_value=16):
            self.assertEqual(device.snapshot()['state']['expression'], 'sad')

    def test_microphone_validates_and_sets_all_hardware_capture_switches(self):
        device = self.app.device
        device.cfg['capture_controls'] = ['Headset,0', 'Headset,1']
        source = {'name': 'mic', 'properties': {'alsa.card': '3'}}
        for data in [{'percent':101}, {'percent':'nan'}, {'mute':'yes'}]:
            with patch.object(module, 'run') as run:
                with self.assertRaises(ValueError): device.microphone(data)
                run.assert_not_called()
        with patch.object(device, 'capture_source', return_value=source), \
             patch.object(device, 'microphone_status', return_value={'hardware_mute':True}), \
             patch.object(module, 'run') as run:
            device.microphone({'mute':True})
            self.assertTrue(device.mic_mute_file.exists())
            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args_list[-2].args[0], ['amixer','-c','3','sset','Headset,0','nocap'])
            self.assertEqual(run.call_args_list[-1].args[0], ['amixer','-c','3','sset','Headset,1','nocap'])

    def test_silent_reply_leaves_thinking_only_after_synthesis_completion(self):
        state=self.app.device.state
        state.update(ready=True, phase='thinking', listening=True, input_count=1,
                     reply_count=1, synth_count=0, reply_spoken=False)
        self.assertEqual(self.app.device.snapshot()['state']['phase'],'thinking')
        state['synth_count']=1
        self.assertEqual(self.app.device.snapshot()['state']['phase'],'listening')

    def test_volume_matches_desktop_cubic_units(self):
        with patch.object(module, 'run') as run:
            response=self.client.post('/api/local/volume',json={'percent':82},headers=self.headers)
            self.assertEqual(response.status_code,200)
            run.assert_called_once_with(['pactl','set-sink-volume','test-sink','53740'])
        for value in [-1,101,'nan']:
            self.assertEqual(self.client.post('/api/local/volume',json={'percent':value},headers=self.headers).status_code,400)

    def test_metadata_preserves_device_time_and_bounded_history(self):
        device=self.app.device
        device.emit('gesture','nod',at=123.5)
        self.assertEqual(device.snapshot()['events'][0]['at'],123.5)
        for i in range(1001): device.emit('reply',str(i))
        self.assertEqual(len(device.snapshot()['events']),1000)
        self.assertEqual(device.snapshot(after=device.serial)['events'],[])

    def test_page_has_device_controls_without_browser_audio(self):
        page=self.client.get('/').data.decode()
        self.assertIn('Speaker volume',page)
        self.assertNotIn('getUserMedia',page)
        self.assertNotIn('AudioContext',page)


# Run with VOICECAT_TEST_BIN=/path/to/voicecat for a real C protocol smoke test.
class VoicecatProtocolTests(unittest.TestCase):
    def test_control_and_metadata_without_audio_hardware(self):
        self.check_reply(b'<|channel>thought\n<channel|>[happy] Hello [nod] there.<turn|>')

    def test_eos_split_across_reads_completes_without_timeout(self):
        self.check_reply(b'<|channel>thought\n<channel|>[happy] Hello [nod] there.<eos>', fragmented=True)

    def test_generation_cap_is_not_spoken(self):
        self.check_reply(b'<|channel>thought\n<channel|>[happy] Hello [nod] there. [SERVE_GEN cap]<turn|>', fragmented=True)

    def test_silent_eos_still_completes_synthesis(self):
        self.check_reply(b'[neutral]<eos>', fragmented=True, spoken=False)

    def check_reply(self, payload, fragmented=False, spoken=True):
        import os, socket, struct, subprocess, threading, time
        binary=os.environ.get('VOICECAT_TEST_BIN')
        if not binary: self.skipTest('set VOICECAT_TEST_BIN to exercise the compiled binary')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); engine=socket.socket(socket.AF_UNIX); engine.bind(str(root/'lg'));engine.listen()
            monitor=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);monitor.bind(str(root/'events'));monitor.settimeout(.2)
            synth=root/'synth.py';synth.write_text('''import sys,struct\ndef emit(k,b):\n sys.stdout.buffer.write(k+struct.pack('<I',len(b))+b);sys.stdout.buffer.flush()\nemit(b'C',b'rate=22050\\n')\nfor line in sys.stdin:\n if 'vc-ui-done' in line: emit(b'M',b'vc-ui-done');continue\n if line.startswith('<|tool_call>'): continue\n emit(b'A',b'0\\t100\\ta\\n');emit(b'P',bytes(4410))\n''')
            def reply():
                conn,_=engine.accept()
                with conn:
                    data=b''
                    while b'\n' not in data: data+=conn.recv(4096)
                    if fragmented:
                        for byte in payload:
                            conn.sendall(bytes([byte])); time.sleep(.003)
                    else:
                        conn.sendall(payload)
                    time.sleep(2)
            threading.Thread(target=reply,daemon=True).start()
            proc=subprocess.Popen([binary,str(root/'lg'),'--stdin-pcm','--start-muted','--clock','0','--idle-compress','0','--mood-route',
                '--monitor-sock',str(root/'events'),'--control-sock',str(root/'control'),
                '--mouth-synth',module.shlex.join([module.sys.executable,str(synth)]),'--mouth-play','cat >/dev/null'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            done=threading.Event()
            def feed():
                try:
                    while not done.wait(.02):proc.stdin.write(bytes(640));proc.stdin.flush()
                except (BrokenPipeError,ValueError):pass
            threading.Thread(target=feed,daemon=True).start()
            try:
                deadline=time.time()+4
                while not (root/'control').exists() and time.time()<deadline:time.sleep(.02)
                with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as control: control.sendto(b'prompt Say hello.',str(root/'control'))
                kinds=set(); events=[]
                expected={'input_done','reply_done','expression','synth_done'}
                if spoken: expected.update(['alignment','play','gesture'])
                while time.time()<deadline and not expected.issubset(kinds):
                    try:
                        event=json.loads(monitor.recv(65536));events.append(event);kinds.add(event['kind'])
                    except socket.timeout:pass
                self.assertTrue(expected.issubset(kinds),kinds)
                self.assertEqual(sum(e['kind']=='reply_done' for e in events),1)
                clauses=[e['text'] for e in events if e['kind']=='clause']
                self.assertEqual(clauses,['Hello  there.'] if spoken else [])
                if not spoken:self.assertNotIn('play',kinds)
            finally:
                done.set();time.sleep(.03);proc.stdin.close()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:module.stop_tree(proc)
                engine.close();monitor.close()

class PlaybackCalibrationTests(unittest.TestCase):
    def device(self, **cfg):
        d = module.Device.__new__(module.Device)
        d.cfg = cfg
        return d

    def test_usb_reset_restores_gain_without_unmuting(self):
        d = self.device(playback_controls=[{'name':'PCM,1', 'value':54}])
        old = 'Limits: Playback 0 - 60\n  Mono: Playback 40 [67%] [-20.00dB] [off]'
        new = old.replace('Playback 40', 'Playback 54')
        with patch.object(module, 'run', side_effect=[old, '', new]) as run:
            d.restore_playback_calibration({'properties':{'alsa.card':'3'}})
        self.assertEqual(run.call_args_list[1].args[0], ['amixer','-c','3','sset','PCM,1','54'])
        with patch.object(module, 'run', return_value=new) as run:
            d.restore_playback_calibration({'properties':{'alsa.card':'3'}})
        self.assertEqual(run.call_count, 1)

    def test_failed_calibration_readback_is_reported(self):
        d = self.device(playback_controls=[{'name':'PCM,1', 'value':54}])
        old = 'Limits: Playback 0 - 60\n  Mono: Playback 40 [67%] [-20.00dB] [on]'
        with patch.object(module, 'run', return_value=old):
            with self.assertRaisesRegex(ValueError, 'readback'):
                d.restore_playback_calibration({'properties':{'alsa.card':'3'}})

    def test_gain_is_specific_to_voice_and_bounded(self):
        d = self.device(piper=['piper'], voice='/a.onnx', voice_gains_db={'/a.onnx':6})
        self.assertAlmostEqual(float(d.synthesis_command()[-1]), 1.9952623)
        d.cfg['voice']='/b.onnx'
        self.assertEqual(d.synthesis_command()[-1], '1')
        d.cfg['voice_gains_db']['/b.onnx']=float('nan')
        with self.assertRaises(ValueError):d.synthesis_command()

if __name__=='__main__': unittest.main()

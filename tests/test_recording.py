"""Recording lifecycle tests without screen capture or audio hardware."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('classroom_recording',
    Path(__file__).resolve().parents[1]/'script/deviceui/recording.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rec = m.Recording(dict(runtime_dir=self.tmp.name, recording_directory=self.tmp.name,
                                    sink='speaker'), lambda: None, lambda: 'processed-mic')
        self.rec.available = True

    def tearDown(self):
        self.rec.stop(wait=True)
        self.tmp.cleanup()

    def wait_phase(self, phase):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if self.rec.snapshot()['phase'] == phase: return
            time.sleep(.01)
        self.fail(self.rec.snapshot())

    def test_cancel_armed_never_captures_or_creates_file(self):
        with patch.object(m, 'classroom_geometry', return_value='0,0 100x100'), \
             patch.object(m.subprocess, 'Popen') as spawn, patch.object(m, 'command') as command:
            self.rec.arm()
            self.wait_phase('armed')
            self.rec.stop(wait=True)
            spawn.assert_not_called()
            command.assert_not_called()
            self.assertEqual(list(self.root.iterdir()), [])

    def test_model_names_and_same_minute_files_never_overwrite(self):
        one = self.rec.destination(['12B', 'E2B'])
        one.write_bytes(b'original')
        two = self.rec.destination(['12B', 'E2B'])
        self.assertRegex(one.name, r'^voice-classroom-12b-e2b-\d{8}-\d{4}\.mp4$')
        self.assertEqual(two.stem, one.stem+'-2')
        self.assertEqual(one.read_bytes(), b'original')

    def test_ready_barrier_and_stop_wait_for_file_finalization(self):
        release = threading.Event()
        class Child:
            returncode = None
            def poll(self): return self.returncode
            def send_signal(self, sig): self.signal = sig
            def wait(self, timeout):
                release.wait(timeout=2)
                self.returncode = 0
                return 0
        child = Child()
        def spawn(argv, **kwargs):
            Path(argv[argv.index('-f')+1]).write_bytes(b'mp4 header')
            return child
        with patch.object(m, 'classroom_geometry', return_value='0,0 100x100'), \
             patch.object(m, 'command', side_effect=['101', '102', '103', '', '', '']) as command, \
             patch.object(m.subprocess, 'Popen', side_effect=spawn) as popen:
            try:
                self.rec.arm()
                popen.assert_not_called()
                self.rec.models_ready(['12b', 'e4b'])
                self.assertEqual(self.rec.snapshot()['phase'], 'recording')
                self.assertIn('12b-e4b', self.rec.snapshot()['file'])
                self.rec.stop()
                self.wait_phase('stopping')
                self.assertTrue(self.rec.thread.is_alive())
            finally:
                release.set()
                self.rec.stop(wait=True)
            self.assertEqual(child.signal, signal.SIGINT)
            self.assertEqual(self.rec.snapshot()['phase'], 'idle')
            self.assertEqual([c.args[0] for c in command.call_args_list[-3:]],
                             [['pactl', 'unload-module', n] for n in ('103', '102', '101')])
            self.assertIn('source=processed-mic', command.call_args_list[2].args[0])
            self.assertIn('remix=no', command.call_args_list[2].args[0])

    def test_partial_audio_setup_failure_cleans_only_owned_module(self):
        self.rec.models = lambda: ['12b', 'e4b']
        with patch.object(m, 'classroom_geometry', return_value='0,0 100x100'), \
             patch.object(m, 'command', side_effect=['901', RuntimeError('audio failure'), '']) as command:
            self.rec.arm()
            self.wait_phase('error')
            self.assertIn('audio failure', self.rec.snapshot()['error'])
            self.assertEqual(command.call_args_list[-1].args[0], ['pactl', 'unload-module', '901'])
            self.assertFalse(list(self.root.glob('*.mp4')))

    def test_missing_recorder_rejected_before_capture(self):
        self.rec.available = False
        with self.assertRaisesRegex(ValueError, 'wf-recorder'): self.rec.arm()

    def test_existing_digital_mix_is_borrowed_without_microphone_or_modules(self):
        self.rec.cfg['recording_audio_source'] = 'classroom.monitor'
        modules = []
        with patch.object(m, 'command', return_value='[{"name":"classroom.monitor"}]') as command, \
             patch.object(self.rec, 'microphone') as microphone:
            self.assertEqual(self.rec.audio_source(modules), 'classroom.monitor')
            command.assert_called_once_with(['pactl', '-f', 'json', 'list', 'sources'])
            microphone.assert_not_called()
            self.assertEqual(modules, [])

    def test_prepares_remote_feed_before_borrowing_mix(self):
        self.rec.cfg.update(recording_audio_source='classroom.monitor',
                            recording_prepare_command=['relay', '--refresh'])
        order = []
        with patch.object(m.subprocess, 'run', side_effect=lambda *a, **k: order.append('prepare')) as run, \
             patch.object(m, 'command', side_effect=lambda *a: order.append('source') or '[{"name":"classroom.monitor"}]'):
            self.assertEqual(self.rec.audio_source([]), 'classroom.monitor')
        self.assertEqual(order, ['prepare', 'source'])
        self.assertEqual(run.call_args.args[0], ['relay', '--refresh'])
        self.assertEqual(run.call_args.kwargs['timeout'], 15)

    def test_failed_remote_preparation_prevents_capture(self):
        self.rec.cfg['recording_prepare_command'] = ['relay', '--refresh']
        with patch.object(m, 'classroom_geometry', return_value='0,0 100x100'), \
             patch.object(m.subprocess, 'run', side_effect=m.subprocess.CalledProcessError(1, 'relay', stderr='no PCM')), \
             patch.object(m.subprocess, 'Popen') as spawn:
            self.rec.prepared_models = ['12b', 'e4b']
            self.rec.work()
            self.assertIn('Recording audio preparation failed: no PCM', self.rec.snapshot()['error'])
            spawn.assert_not_called()
            self.assertFalse(list(self.root.glob('*.mp4')))

    def test_missing_configured_mix_does_not_silently_switch_to_room_audio(self):
        self.rec.cfg['recording_audio_source'] = 'missing.monitor'
        with patch.object(m, 'command', return_value='[]') as command:
            with self.assertRaisesRegex(ValueError, 'unavailable'): self.rec.audio_source([])
            self.assertEqual(command.call_count, 1)

    def test_native_local_relay_owns_only_its_module_and_sets_both_gains(self):
        self.rec.cfg.update(recording_audio_source='classroom.monitor',
                            recording_local_relay=dict(sink='classroom', volume=58982))
        modules = []
        responses = ['[{"name":"classroom.monitor"}]', '901',
                     '[{"index":51,"owner_module":"901"},{"index":52,"owner_module":null}]', '',
                     '[{"index":61,"owner_module":"901"}]', '']
        with patch.object(m, 'command', side_effect=responses) as command, \
             patch.object(self.rec, 'microphone') as microphone:
            self.assertEqual(self.rec.audio_source(modules), 'classroom.monitor')
            self.assertEqual(modules, ['901'])
            microphone.assert_not_called()
            calls = [c.args[0] for c in command.call_args_list]
            self.assertIn('source=speaker.monitor', calls[1])
            self.assertIn('latency_msec=20', calls[1])
            self.assertIn(['pactl', 'set-sink-input-volume', '51', '58982'], calls)
            self.assertIn(['pactl', 'set-source-output-volume', '61', '65536'], calls)

    def test_native_relay_setup_failure_retains_module_for_cleanup(self):
        self.rec.cfg.update(recording_audio_source='classroom.monitor',
                            recording_local_relay=dict(sink='classroom'))
        modules = []
        with patch.object(m, 'command', side_effect=['[{"name":"classroom.monitor"}]',
                                                    '901', RuntimeError('capture failed')]):
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                self.rec.audio_source(modules)
        self.assertEqual(modules, ['901'])

    def test_hidden_classroom_does_not_capture_another_workspace(self):
        window = dict(name='Voice classroom · test', rect=dict(x=0, y=0, width=900, height=600))
        tree = dict(type='output', rect=dict(x=0, y=0, width=1920, height=1080),
                    nodes=[dict(type='workspace', id=10, nodes=[window])])
        with patch.object(m, 'command', side_effect=[json.dumps(tree), '[{"id":11,"visible":true}]']):
            with self.assertRaisesRegex(ValueError, 'focus'): m.classroom_geometry()
        with patch.object(m, 'command', side_effect=[json.dumps(tree), '[{"id":10,"visible":true}]']):
            self.assertEqual(m.classroom_geometry(), '0,0 900x600')


if __name__ == '__main__': unittest.main()

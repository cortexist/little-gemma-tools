"""Recording lifecycle tests without screen capture or audio hardware."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

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

    def test_default_room_mix_is_centered_mono_with_explicit_processed_mic_selection(self):
        with patch.object(m, 'command', side_effect=['1', '2', '3']) as command:
            self.rec.audio_source([])
        calls = [c.args[0] for c in command.call_args_list]
        self.assertIn('channel_map=mono', calls[0])
        self.assertIn('channel_map=front-left', calls[2])
        self.assertIn('remix=no', calls[2])

    def test_stereo_room_selects_distinct_raw_channels_without_digital_voice_overlay(self):
        self.rec.cfg.update(recording_room_channels=[2, 4], recording_room_gain_db=40)
        sources = json.dumps([dict(name='processed-mic',
            channel_map='front-left,front-right,rear-left,rear-right,front-center,lfe')])
        owned = []
        with patch.object(m, 'command', side_effect=[sources, '123', '']) as command:
            source = self.rec.audio_source(owned)
        self.assertEqual(owned, ['123'])
        self.assertTrue(source.startswith('voice_room_stereo_'))
        calls = [c.args[0] for c in command.call_args_list]
        self.assertIn('master_channel_map=rear-left,front-center', calls[1])
        self.assertIn('channel_map=front-left,front-right', calls[1])
        self.assertIn('remix=no', calls[1])
        self.assertIn('source_properties=channelmix.max-volume=256', calls[1])
        self.assertFalse(any('module-loopback' in c for c in calls))
        self.assertEqual(calls[2], ['pactl', 'set-source-volume', source, str(round(65536*10**(40/60)))])

    def test_invalid_or_unavailable_stereo_pair_never_uses_other_channels(self):
        for pair in ([0, 0], [2], [-1, 4], [True, 4], '2,4'):
            self.rec.cfg['recording_room_channels'] = pair
            with patch.object(m, 'command') as command:
                with self.assertRaises(ValueError): self.rec.audio_source([])
                command.assert_not_called()
        self.rec.cfg['recording_room_channels'] = [2, 4]
        with patch.object(m, 'command', return_value=json.dumps([
                dict(name='processed-mic', channel_map='front-left,front-right,lfe')])) as command:
            with self.assertRaisesRegex(ValueError, 'channels unavailable'): self.rec.audio_source([])
            self.assertEqual(command.call_count, 1)

    def test_failed_stereo_gain_setup_still_cleans_its_remap(self):
        self.rec.cfg['recording_room_channels'] = [0, 1]
        self.rec.prepared_models = ['12b', 'e4b']
        responses = ['[{"name":"processed-mic","channel_map":"front-left,front-right"}]',
                     '901', RuntimeError('gain failed'), '']
        with patch.object(self.rec, 'geometry', return_value='0,0 100x100'), \
                patch.object(m, 'command', side_effect=responses) as command:
            self.rec.work()
        self.assertEqual(command.call_args_list[-1].args[0], ['pactl', 'unload-module', '901'])
        self.assertIn('gain failed', self.rec.snapshot()['error'])

    def test_stereo_recording_gain_rejects_nonfinite_or_excessive_levels(self):
        self.rec.cfg['recording_room_channels'] = [2, 4]
        for gain in (True, float('nan'), float('inf'), 49, -49, '18'):
            self.rec.cfg['recording_room_gain_db'] = gain
            with patch.object(m, 'command') as command:
                with self.assertRaisesRegex(ValueError, 'gain'): self.rec.audio_source([])
                command.assert_not_called()

    def test_digital_mode_is_bound_when_armed_and_cancel_does_not_open_feeds(self):
        self.rec.digital_factory = Mock()
        self.rec.state['audio_modes'] = ['room', 'digital']
        with patch.object(self.rec, 'geometry', return_value='0,0 100x100'):
            self.rec.arm('digital')
            self.assertEqual(self.rec.snapshot()['audio_mode'], 'digital')
            with self.assertRaisesRegex(ValueError, 'already'): self.rec.arm('room')
            self.assertEqual(self.rec.snapshot()['audio_mode'], 'digital')
            self.rec.stop(wait=True)
        self.rec.digital_factory.assert_not_called()

    def test_digital_failure_closes_partial_feeds_and_never_falls_back_to_microphone(self):
        digital = Mock()
        digital.start.side_effect = RuntimeError('somewhere: offline')
        self.rec.digital_factory = Mock(return_value=digital)
        self.rec.state['audio_mode'] = 'digital'
        self.rec.prepared_models = ['12b', 'e4b', '12b']
        with patch.object(self.rec, 'geometry', return_value='0,0 100x100'), \
                patch.object(self.rec, 'audio_source') as room, patch.object(m.subprocess, 'Popen') as spawn:
            self.rec.work()
        room.assert_not_called()
        spawn.assert_not_called()
        digital.close.assert_called_once()
        self.assertEqual(self.rec.snapshot()['phase'], 'error')
        self.assertIn('somewhere', self.rec.snapshot()['error'])
        self.assertFalse(list(self.root.glob('*.mp4')))

    def test_unknown_audio_mode_rejected_before_arming(self):
        with self.assertRaisesRegex(ValueError, 'audio mode'): self.rec.arm('surprise')
        self.assertIsNone(self.rec.thread)

    def test_asr_tracks_start_before_video_and_drain_before_video_finalize(self):
        order = []
        inputs = Mock(directory=self.root/'video.asr')
        inputs.start.side_effect = lambda *a: order.append('inputs-start')
        inputs.close.side_effect = lambda: order.append('inputs-close')
        def video_started(at):
            order.append('video-ready')
            self.rec.cancel.set()
        inputs.video_started.side_effect = video_started
        self.rec.input_factory = Mock(return_value=inputs)
        self.rec.prepared_models = ['12b', 'e4b', '12b']
        class Child:
            returncode = None
            def poll(self): return self.returncode
            def send_signal(self, sig): order.append('video-stop')
            def wait(self, timeout): self.returncode = 0
        def spawn(argv, **kwargs):
            order.append('video-start')
            Path(argv[argv.index('-f')+1]).write_bytes(b'video')
            return Child()
        with patch.object(self.rec, 'geometry', return_value='0,0 100x100'), \
                patch.object(self.rec, 'audio_source', return_value='physical-stereo') as room, \
                patch.object(m.subprocess, 'Popen', side_effect=spawn) as popen:
            self.rec.work()
        self.assertEqual(order, ['inputs-start', 'video-start', 'video-ready', 'inputs-close', 'video-stop'])
        self.assertIn('--audio=physical-stereo', popen.call_args.args[0])
        self.assertEqual(self.rec.snapshot()['phase'], 'idle')
        self.assertTrue(self.rec.snapshot()['asr_directory'].endswith('video.asr'))

    def test_asr_start_failure_keeps_video_unstarted_and_closes_partial_tracks(self):
        inputs = Mock()
        inputs.start.side_effect = RuntimeError('cortex: ASR input unavailable')
        self.rec.input_factory = Mock(return_value=inputs)
        self.rec.prepared_models = ['12b', 'e4b', '12b']
        with patch.object(self.rec, 'geometry', return_value='0,0 100x100'), \
                patch.object(self.rec, 'audio_source', return_value='physical-stereo'), \
                patch.object(m.subprocess, 'Popen') as spawn:
            self.rec.work()
        spawn.assert_not_called()
        inputs.close.assert_called_once()
        self.assertIn('ASR input unavailable', self.rec.snapshot()['error'])
        self.assertFalse(list(self.root.glob('*.mp4')))

    def test_trio_modes_are_identifiable_in_saved_filenames(self):
        self.rec.digital_factory = Mock()
        self.rec.cfg['recording_prefix'] = 'voice-trio'
        for mode in ('room', 'digital'):
            self.rec.state['audio_mode'] = mode
            self.assertTrue(self.rec.destination(['12b', 'e4b', '12b']).name.startswith('voice-trio-'+mode+'-'))

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

    def test_trio_recording_crops_only_the_selected_visible_terminal(self):
        window = dict(id=5, name='Terminal with browser pane', rect=dict(x=4,y=38,width=1912,height=1038))
        tree = dict(type='output', nodes=[dict(type='workspace', id=10, nodes=[window])])
        responses = [json.dumps(tree), '[{"id":10,"visible":true}]']
        with patch.object(m, 'command', side_effect=responses):
            self.assertEqual(m.classroom_geometry(window_id=5, crop=[.5,0,.5,1]), '960,38 956x1038')
        with patch.object(m, 'command', side_effect=responses):
            with self.assertRaisesRegex(ValueError, 'focus'):
                m.classroom_geometry(window_id=99, crop=[.5,0,.5,1])
        with patch.object(m, 'command', side_effect=responses):
            with self.assertRaisesRegex(ValueError, 'fit'):
                m.classroom_geometry(window_id=5, crop=[.5,0,1,1])

    def test_three_model_recording_keeps_trio_filename_separate(self):
        self.rec.cfg['recording_prefix'] = 'voice-trio'
        self.assertRegex(self.rec.destination(['12b','e4b','e4b']).name,
                         r'^voice-trio-12b-e4b-e4b-\d{8}-\d{4}\.mp4$')


if __name__ == '__main__': unittest.main()

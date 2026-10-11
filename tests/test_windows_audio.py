"""Portable Windows-adapter boundary checks; device smoke tests run on Windows."""
import io
from pathlib import Path
import queue
import sys
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script/deviceui'))
from windows_audio import WindowsDeviceMixin, read_exact, select_output


class WindowsAudioTests(unittest.TestCase):
    def test_asr_recording_taps_same_pcm_before_vad_with_rejected_blocks_marked(self):
        device = WindowsDeviceMixin.__new__(WindowsDeviceMixin)
        device.cfg = {}
        device.lock = threading.RLock()
        device.cancel = threading.Event()
        device.state = dict(listening=True)
        device.turn_active = False
        device.capture_epoch = 0
        device.endpoint_until = 0
        blocks, errors = queue.Queue(), queue.Queue()
        pcms = [bytes([value, 0])*320 for value in (1, 2, 3)]
        now = time.monotonic()
        for epoch, at, pcm in [(0, now, pcms[0]), (-1, now, pcms[1]), (0, now-1, pcms[2])]:
            blocks.put((at, epoch, pcm, ''))
        captured = []
        def record(at, pcm, eligible):
            captured.append((pcm, eligible))
            if len(captured) == 3: device.cancel.set()
        device.record_input_block = record
        utterance = Mock(active=False)
        utterance.ready.return_value = False
        utterance.request.return_value = None
        with patch('windows_audio.StreamingUtterance', return_value=utterance):
            device.capture(blocks, errors)
        self.assertTrue(errors.empty())
        self.assertEqual([flag for _, flag in captured], [True, False, False])
        for (recorded, _), original in zip(captured, pcms): self.assertIs(recorded, original)
        utterance.feed.assert_called_once()
        self.assertIs(utterance.feed.call_args.args[0], pcms[0])

    def test_output_never_uses_default_or_duplicate_endpoint(self):
        sd = Mock()
        sd.query_hostapis.return_value = [{'name': 'MME'}, {'name': 'Windows WASAPI'}]
        mme = dict(name='array', max_output_channels=2, hostapi=0)
        wasapi = dict(name='array', max_output_channels=2, hostapi=1)
        sd.query_devices.return_value = [mme, dict(wasapi, name='other')]
        with self.assertRaises(ValueError): select_output(sd, 'array')
        sd.query_devices.return_value.append(wasapi)
        self.assertEqual(select_output(sd, 'array'), 2)
        sd.query_devices.return_value.append(wasapi)
        with self.assertRaises(ValueError): select_output(sd, 'array')

    def test_partial_pipe_reads_and_truncation(self):
        class ShortPipe(io.BytesIO):
            def read(self, count): return super().read(min(count, 2))
        self.assertEqual(read_exact(ShortPipe(b'abcde'), 5), b'abcde')
        self.assertEqual(read_exact(ShortPipe(b''), 5), b'')
        with self.assertRaises(RuntimeError): read_exact(ShortPipe(b'ab'), 5)

    def test_prompt_reserves_turn_before_worker_can_run(self):
        device = WindowsDeviceMixin.__new__(WindowsDeviceMixin)
        device.state = dict(ready=True, phase='paused', input_count=0)
        device.lock = threading.RLock()
        device.prompts = queue.Queue(maxsize=1)
        device.turn_active = False
        device.command('prompt hello')
        self.assertEqual(device.state['input_count'], 1)
        with self.assertRaises(ValueError): device.command('prompt again')
        with self.assertRaises(ValueError): device.listen(True)

    def test_late_pcm_gap_does_not_enable_a_second_turn(self):
        class Base:
            def snapshot(self, after=0):
                self.state['phase'] = 'paused'
                return {'state': dict(self.state)}
        class Device(WindowsDeviceMixin, Base): pass
        device = Device.__new__(Device)
        device.lock = threading.RLock()
        device.state = dict(ready=True, phase='speaking', audible_until=0)
        device.turn_active = True
        self.assertEqual(device.snapshot()['state']['phase'], 'thinking')
        with self.assertRaises(ValueError): device.command('prompt too soon')

    def test_completed_turn_budget_cannot_be_reopened_by_manual_prompt(self):
        device = WindowsDeviceMixin.__new__(WindowsDeviceMixin)
        device.lock = threading.RLock()
        device.prompts = queue.Queue(maxsize=1)
        device.turn_active = False
        device.state = dict(ready=True, phase='paused', input_count=2, turn_limit=2)
        with self.assertRaisesRegex(ValueError, 'turn limit'):
            device.command('prompt another reply')
        self.assertEqual(device.state['input_count'], 2)
        self.assertTrue(device.prompts.empty())

    def test_pcm_starts_before_reply_end_and_turn_completes_only_after_drain(self):
        device = WindowsDeviceMixin.__new__(WindowsDeviceMixin)
        device.lock = threading.RLock()
        device.cancel = threading.Event()
        device.cfg = {}
        device.state = dict(history=[], reply_count=0, synth_count=0, synth_bytes=0,
                            played_bytes=0, audible_until=0, listening=False)
        device.speech_rate = 22050
        device.capture_epoch = 0
        device.turn_active = True
        device.synth_sent = device.synth_completed = 0
        device.frames = queue.Queue()
        device.emit = Mock()
        device.terminate_owned = Mock()
        sequence = [0]
        def synthesize(line):
            sequence[0] += 1
            device.frames.put((b'P', bytes(100)))
            device.frames.put((b'D', str(sequence[0]).encode()))
        device.synth = SimpleNamespace(stdin=SimpleNamespace(write=synthesize, flush=lambda: None))
        started, draining, drained = threading.Event(), threading.Event(), threading.Event()
        stream = Mock(latency=0)
        stream.write.side_effect = lambda pcm: started.set() or False
        stream.stop.side_effect = lambda: draining.set() or drained.wait(2)
        sd = SimpleNamespace(RawOutputStream=Mock(return_value=stream), WasapiSettings=Mock())
        np = SimpleNamespace(frombuffer=lambda pcm, dtype: pcm,
                             repeat=lambda pcm, n: SimpleNamespace(tobytes=lambda: pcm))
        client = SimpleNamespace(stdin=io.BytesIO())
        replies = queue.Queue()
        errors = []
        def turn():
            try: device.turn('Question', client, replies, 0, {})
            except Exception as exc: errors.append(exc)
        with patch.dict(sys.modules, {'sounddevice': sd, 'numpy': np}):
            thread = threading.Thread(target=turn)
            thread.start()
            try:
                replies.put(b'First sentence. ')
                self.assertTrue(started.wait(2))
                self.assertEqual(device.state['reply_count'], 0)
                self.assertEqual(device.state['synth_count'], 0)
                replies.put(b'Second sentence.<turn|>\n')
                self.assertTrue(draining.wait(2))
                self.assertEqual(device.state['reply_count'], 1)
                self.assertTrue(device.turn_active)
                self.assertEqual(device.state['synth_count'], 0)
            finally:
                drained.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(device.state['synth_count'], 1)
        self.assertEqual(device.synth_sent, device.synth_completed)
        self.assertFalse(device.turn_active)


if __name__ == '__main__':
    unittest.main()

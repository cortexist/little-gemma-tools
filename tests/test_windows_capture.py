"""Capture endpointing boundaries, stale input and raw ASR spelling."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script/deviceui'))
from windows_capture import CapturePCM, Utterance, select_input, speech_text, transcribe
try:
    import numpy as np
except ImportError:
    np = None


class CaptureTests(unittest.TestCase):
    def test_endpoint_waits_for_peer_hold_and_rejects_short_noise(self):
        u = Utterance(silence=.1)
        frame = b'\0'*640
        for _ in range(15): self.assertIsNone(u.feed(frame, 300))
        for _ in range(20): self.assertIsNone(u.feed(frame, 0, held=True))
        self.assertTrue(u.feed(frame, 0))
        self.assertIsNone(u.feed(frame, 300))
        for _ in range(10): self.assertIsNone(u.feed(frame, 0))

    def test_capture_storage_is_bounded_even_while_peer_holds(self):
        u = Utterance(maximum=1)
        for _ in range(100):
            self.assertIsNone(u.feed(b'\0'*640, 300, held=True))
            self.assertLess(len(u.parts), 60)

    @unittest.skipIf(np is None, 'numpy is in the Windows/Piper runtime')
    def test_native_capture_selects_processed_channel_without_raw_mic_mix(self):
        data = np.zeros((320,6), dtype='<i2')
        data[:,0] = np.arange(320)
        data[:,1:] = -10000
        self.assertEqual(CapturePCM(16000,6).convert(data.tobytes()), data[:,0].tobytes())

    @unittest.skipIf(np is None, 'numpy is in the Windows/Piper runtime')
    def test_48k_conversion_preserves_stream_boundaries_and_rejects_aliases(self):
        def converted(hz):
            t = np.arange(4800)/48000
            x = (12000*np.sin(2*np.pi*hz*t)).astype('<i2')
            data = np.repeat(x[:,None],2,axis=1)
            c = CapturePCM(48000)
            whole = c.convert(data.tobytes())
            c = CapturePCM(48000)
            split = b''.join(c.convert(chunk.tobytes()) for chunk in np.split(data,5))
            self.assertEqual(whole,split)
            return np.sqrt(np.mean(np.frombuffer(whole,dtype='<i2')[100:].astype(float)**2))
        self.assertGreater(converted(1000), 7000)
        self.assertLess(converted(12000), 50)

    def test_input_requires_exact_wasapi_endpoint(self):
        sd = Mock()
        sd.query_hostapis.return_value = [{'name':'MME'}, {'name':'Windows WASAPI'}]
        d = dict(name='reSpeaker', hostapi=1, max_input_channels=2)
        sd.query_devices.return_value = [dict(d, hostapi=0), d]
        self.assertEqual(select_input(sd, 'reSpeaker'), 1)
        with self.assertRaises(ValueError): select_input(sd, 'default')
        sd.query_devices.return_value.append(d)
        with self.assertRaises(ValueError): select_input(sd, 'reSpeaker')

    def test_asr_keeps_raw_name_spelling_and_suppresses_non_speech(self):
        for raw, expected in [('My name is Shai.', 'My name is Shai.'), ('[BLANK_AUDIO]', '')]:
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps({'text':raw}).encode())) as request:
                self.assertEqual(speech_text(transcribe('http://localhost/inference', b'\0'*640)), expected)
                body = request.call_args.args[0].data
                self.assertIn(b'RIFF', body)
                self.assertNotIn(b'name="prompt"', body)


if __name__ == '__main__': unittest.main()

"""ASR diagnostics preserve samples and fail visibly without blocking capture."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui')]
import recording_input as source
import recording_input_tracks as tracks
import server

SPEAKER = dict(speaker_id='chey', display_name='Chey')
NODE = dict(id='cortex', speaker_id='chey', url='http://cortex:8098', key='private')


def stream(pcm, *, offset=0, end=True, session='room', eligible=True, error=''):
    meta = dict(protocol=source.FORMAT, capture_id='capture', session=session,
                speaker_id='chey', rate=16000, channels=1, sample_format='s16le')
    wire = source.frame(b'M', json.dumps(meta).encode())
    wire += source.frame(b'P', source.BLOCK.pack(offset, 42., eligible)+pcm)
    if end:
        wire += source.frame(b'E', json.dumps(dict(samples=len(pcm)//2,
                             reason='recording-stopped', error=error)).encode())
    return wire


class InputTapTests(unittest.TestCase):
    def test_exact_bytes_order_eligibility_and_graceful_drain(self):
        tap = source.InputTap('room')
        sub = tap.subscribe(SPEAKER)
        pcm = bytes(range(256))*2
        tap.feed(10., pcm, False)
        tap.feed(10.016, pcm[::-1], True)
        tap.stop(sub.id)
        wire = io.BytesIO(b''.join(sub))
        kind, payload = tracks.read_frame(wire)
        self.assertEqual(kind, b'M')
        self.assertEqual(json.loads(payload)['speaker_id'], 'chey')
        for offset, at, eligible, expected in [(0, 10., 0, pcm), (256, 10.016, 1, pcm[::-1])]:
            kind, payload = tracks.read_frame(wire)
            self.assertEqual(kind, b'P')
            self.assertEqual(source.BLOCK.unpack(payload[:source.BLOCK.size]), (offset, at, eligible))
            self.assertEqual(payload[source.BLOCK.size:], expected)
        kind, payload = tracks.read_frame(wire)
        self.assertEqual(kind, b'E')
        self.assertEqual(json.loads(payload), dict(samples=512, reason='recording-stopped', error=''))
        self.assertIsNone(tap.current)

    def test_slow_consumer_ends_only_recording_and_never_blocks_or_drops_silently(self):
        tap = source.InputTap('room')
        sub = tap.subscribe(SPEAKER)
        for _ in range(10000): tap.feed(0, bytes(640), True)
        self.assertTrue(sub.ended.is_set())
        self.assertEqual(sub.chunks.qsize(), 100)
        self.assertEqual(sub.samples, 32000)
        self.assertIn('incomplete', sub.error)
        sub.close()
        replacement = tap.subscribe(SPEAKER)
        tap.feed(2., bytes(640), True)
        self.assertEqual(replacement.samples, 320)
        replacement.close()

    def test_single_subscriber_and_stale_stop_do_not_close_other_recording(self):
        tap = source.InputTap('room')
        sub = tap.subscribe(SPEAKER)
        with self.assertRaisesRegex(ValueError, 'already'): tap.subscribe(SPEAKER)
        with self.assertRaisesRegex(ValueError, 'Stale'): tap.stop('other')
        self.assertFalse(sub.ended.is_set())
        tap.close('microphone failed')
        self.assertEqual(sub.error, 'microphone failed')
        sub.close()
        with self.assertRaisesRegex(ValueError, 'stopped'): tap.subscribe(SPEAKER)


class TrackTests(unittest.TestCase):
    def test_fragmented_stream_becomes_byte_identical_wav_and_timing(self):
        class Fragmented(io.BytesIO):
            def read(self, size=-1): return super().read(min(size, 7))
        pcm = bytes(range(256))*4
        with tempfile.TemporaryDirectory() as root:
            cancel = threading.Event(); cancel.set()
            track = tracks.InputTrack(NODE, 'room', Path(root), cancel)
            with patch.object(tracks.urllib.request, 'urlopen', return_value=Fragmented(stream(pcm, eligible=False))) as request:
                track.work()
            self.assertFalse(track.error)
            self.assertTrue(track.result['complete'])
            with wave.open(str(Path(root)/'cortex.wav')) as wav:
                self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getsampwidth()), (1, 16000, 2))
                self.assertEqual(wav.readframes(wav.getnframes()), pcm)
            timing = json.loads((Path(root)/'cortex.jsonl').read_text())
            self.assertFalse(timing['asr_eligible'])
            self.assertEqual(timing['captured_monotonic'], 42.)
            self.assertEqual(request.call_args.args[0].get_header('X-voice-key'), 'private')

    def test_missing_terminal_frame_offset_jump_and_source_overflow_leave_valid_partial_wav(self):
        cases = [(dict(end=False), 'end record'), (dict(offset=1), 'Discontinuous'),
                 (dict(error='consumer overflow'), 'overflow')]
        for options, expected in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as root:
                cancel = threading.Event(); cancel.set()
                track = tracks.InputTrack(NODE, 'room', Path(root), cancel)
                with patch.object(tracks.urllib.request, 'urlopen', return_value=io.BytesIO(stream(bytes(640), **options))):
                    track.work()
                self.assertIn(expected, track.error)
                self.assertFalse(track.result['complete'])
                with wave.open(str(Path(root)/'cortex.wav')) as wav:
                    self.assertEqual(wav.getnframes(), 0 if options.get('offset') else 320)

    def test_wrong_session_never_creates_track_and_unexpected_end_is_error(self):
        for session, expected in [('old-room', 'mismatch'), ('room', 'before recording stopped')]:
            with tempfile.TemporaryDirectory() as root:
                track = tracks.InputTrack(NODE, 'room', Path(root), threading.Event())
                with patch.object(tracks.urllib.request, 'urlopen', return_value=io.BytesIO(stream(bytes(640), session=session))):
                    track.work()
                self.assertIn(expected, track.error)
                self.assertFalse(track.result['complete'])
                if session != 'room': self.assertFalse(list(Path(root).iterdir()))

    def test_partial_start_keeps_failure_manifest_and_never_writes_credentials(self):
        def failed(track):
            track.error = 'cortex: offline'
            track.result['error'] = track.error
            track.ready.set()
        with tempfile.TemporaryDirectory() as root:
            group = tracks.InputTracks([NODE], 'room')
            with patch.object(tracks.InputTrack, 'work', failed):
                with self.assertRaisesRegex(RuntimeError, 'offline'):
                    group.start(threading.Event(), Path(root)/'video.mp4')
                with self.assertRaisesRegex(RuntimeError, 'offline'): group.close()
            manifest = (Path(root)/'video.asr/manifest.json').read_text()
            self.assertNotIn('private', manifest)
            self.assertFalse(json.loads(manifest)['tracks'][0]['complete'])


class EndpointTests(unittest.TestCase):
    def test_authorization_session_singleton_and_normal_stop(self):
        with tempfile.TemporaryDirectory() as root, patch.object(server.Device, 'audio_poll'):
            cfg = dict(key='private', name='Chey', runtime_dir=root, controls_only=True,
                       voice='/voice', profiles={}, sink='speaker', source='microphone')
            device = server.Device(cfg)
            device.speaker = SPEAKER
            device.state['ready'] = True
            tap = source.InputTap('room')
            device.room = SimpleNamespace(session='room', asr_input=tap)
            class ExistingDevice(server.Device):
                def __new__(cls, config): return device
            with patch.object(server, 'Device', ExistingDevice): app = server.create_app(cfg)
            client = app.test_client()
            headers = {'X-Voice-Key':'private'}
            self.assertEqual(client.get('/api/recording/asr?session=room').status_code, 403)
            self.assertEqual(client.get('/api/recording/asr?session=room', headers=dict(headers, Origin='http://other')).status_code, 403)
            self.assertEqual(client.get('/api/recording/asr?session=old', headers=headers).status_code, 409)
            original = tap.subscribe
            def subscribe(speaker):
                sub = original(speaker)
                tap.feed(1., bytes(640), True)
                return sub
            with patch.object(tap, 'subscribe', side_effect=subscribe):
                response = client.get('/api/recording/asr?session=room', headers=headers, buffered=False)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(client.get('/api/recording/asr?session=room', headers=headers).status_code, 409)
                capture_id = tap.current.id
                self.assertEqual(client.post('/api/recording/asr/stop', json=dict(session='room', capture_id='wrong'), headers=headers).status_code, 409)
                self.assertEqual(client.post('/api/recording/asr/stop', json=dict(session='room', capture_id=capture_id), headers=headers).status_code, 200)
                self.assertIn(bytes(640), response.data)
                response.close()
                self.assertIsNone(tap.current)
            device.state['ready'] = False
            self.assertEqual(client.get('/api/recording/asr?session=room', headers=headers).status_code, 503)


if __name__ == '__main__': unittest.main()

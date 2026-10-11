"""Digital audio ownership, backpressure, authentication and disconnects."""
import io
import socket
import struct
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui')]
import recording_audio as audio
import server


class CaptureTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'Unix playback tap')
    def test_rendered_frames_survive_fragmentation_and_disconnect_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root)/'ff.sock')
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(path); listener.listen(1)
                capture = audio.PlaybackCapture('speaker', playback_socket=path)
                capture.thread.start()
                conn, _ = listener.accept()
                with conn:
                    self.assertEqual(conn.recv(10), struct.pack('<BBHHI', 1, ord('o'), 0, 0, 0))
                    pcm = bytes(range(160))*2
                    frame = struct.pack('<BBHHI', 1, ord('P'), 0, 0, 320)+pcm
                    for part in (frame[:3], frame[3:199], frame[199:]+frame[:1], frame[1:]):
                        conn.sendall(part)
                    self.assertEqual(capture.chunks.get(timeout=2), pcm*2)
                    self.assertEqual((capture.rate, capture.channels), (16000, 1))
                capture.thread.join(timeout=2)
                self.assertFalse(capture.thread.is_alive())
                self.assertIn('capture stopped', capture.error)
                capture.close()

    def test_capture_uses_the_selected_monitor_native_rate(self):
        sources = '[{"name":"default.monitor","sample_specification":"s16le 2ch 48000Hz"},' \
                  '{"name":"respeaker.monitor","sample_specification":"s16le 2ch 16000Hz"}]'
        with patch.object(audio, 'command', return_value=sources):
            self.assertEqual(audio.monitor_rate('respeaker'), 16000)
            self.assertEqual(audio.monitor_rate('default'), 48000)
            with self.assertRaisesRegex(ValueError, 'unavailable'): audio.monitor_rate('missing')

    def test_readiness_waits_for_a_primed_recording_buffer(self):
        relay = audio.DigitalRelay(dict(id='cortex', url='http://node', key='secret'), 'mix')
        player = Mock(); player.poll.return_value = None
        class Source:
            headers = {'X-PCM-Rate':'16000', 'X-PCM-Channels':'2', 'X-PCM-Format':'s16le'}
            reads = 0
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(inner,size):
                self.assertEqual(relay.ready.is_set(), inner.reads >= 10)
                inner.reads += 1
                if inner.reads == 11: relay.cancel.set()
                return bytes(size)
        with patch.object(audio.urllib.request, 'urlopen', return_value=Source()), \
                patch.object(audio.subprocess, 'Popen', return_value=player):
            relay.work()
        self.assertFalse(relay.error)

    def test_slow_consumer_fails_instead_of_queueing_old_speech(self):
        capture = audio.PlaybackCapture('speaker')
        for _ in range(16): capture.put(bytes(3840))
        self.assertTrue(capture.cancel.is_set())
        self.assertEqual(capture.chunks.qsize(), 15)
        self.assertIn('fell behind', capture.error)
        with self.assertRaisesRegex(RuntimeError, 'fell behind'): next(iter(capture))

    def test_pcm_endpoint_requires_key_and_uses_only_configured_output(self):
        with tempfile.TemporaryDirectory() as root, patch.object(server.Device, 'audio_poll'):
            app = server.create_app(dict(key='private', name='Byers', runtime_dir=root,
                controls_only=True, voice='/voice', profiles={}, sink='configured-speaker', source='microphone'))
            client = app.test_client()
            capture = Mock(rate=16000, channels=2)
            capture.thread.is_alive.return_value = True
            capture.__iter__ = Mock(return_value=iter([b'pcm']))
            with patch.object(audio, 'PlaybackCapture', return_value=capture) as factory:
                self.assertEqual(client.get('/api/recording/pcm').status_code, 403)
                self.assertEqual(client.get('/api/recording/pcm', headers={'X-Voice-Key':'private', 'Origin':'http://other'}).status_code, 403)
                factory.assert_not_called()
                response = client.get('/api/recording/pcm?sink=microphone', headers={'X-Voice-Key':'private'})
                self.assertEqual(response.status_code, 200)
                factory.assert_called_once_with('configured-speaker')
                self.assertEqual(response.headers['X-PCM-Rate'], '16000')
                self.assertEqual(response.data, b'pcm')
                self.assertEqual(client.get('/api/recording/pcm', headers={'X-Voice-Key':'private'}).status_code, 409)
                response.close()
                capture.close.assert_called()

    def test_partial_remote_stream_is_an_error_and_closes_player(self):
        source = io.BytesIO(b'partial PCM')
        source.headers = {'X-PCM-Rate':'48000', 'X-PCM-Channels':'2', 'X-PCM-Format':'s16le'}
        player = Mock()
        player.poll.side_effect = [None, 0]
        relay = audio.DigitalRelay(dict(id='somewhere', url='http://node', key='secret'), 'isolated-mix')
        with patch.object(audio.urllib.request, 'urlopen', return_value=source) as request, \
                patch.object(audio.subprocess, 'Popen', return_value=player):
            relay.work()
        self.assertEqual(request.call_args.args[0].get_header('X-voice-key'), 'secret')
        with self.assertRaisesRegex(RuntimeError, 'somewhere.*closed'): relay.check()
        player.terminate.assert_called_once()
        player.stdin.write.assert_not_called()

    def test_failed_remote_feed_is_reported_and_owned_mix_is_removed(self):
        def failed(relay):
            relay.error = relay.node['id']+': unavailable'
            relay.ready.set()
        with patch.object(audio, 'command', side_effect=['123', '']) as command, \
                patch.object(audio.DigitalRelay, 'work', failed):
            mix = audio.DigitalMix([dict(id='somewhere')])
            try:
                with self.assertRaisesRegex(RuntimeError, 'somewhere'): mix.start(threading.Event())
            finally: mix.close()
        self.assertEqual(command.call_args.args[0], ['pactl', 'unload-module', '123'])
        self.assertIsNone(mix.module)


if __name__ == '__main__': unittest.main()

"""ASR overlap, immutable commits, window trimming, and endpoint reuse."""
import io
from pathlib import Path
import queue
import struct
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script/deviceui'))
from streaming_asr import StreamingUtterance, parse_vtt, asr_prompt
from windows_audio import WindowsDeviceMixin


FRAME = b'\1\0'*320


def feed(u, seconds, level=400):
    for _ in range(round(seconds/.02)):
        u.feed(FRAME, level)


def result(text, segments=()):
    return dict(words=text.split(), segments=list(segments), elapsed=.04)


class StreamingTests(unittest.TestCase):
    def test_name_hint_precedes_recent_transcript_without_losing_tail(self):
        hint = 'Names mentioned in this conversation: Chey, Mrs. Byers, Agent Smith.'
        self.assertEqual(asr_prompt('', 'prior words'), 'prior words')
        self.assertEqual(asr_prompt(hint, 'prior words'), hint+' prior words')
        self.assertTrue(asr_prompt(hint, 'x'*400).endswith('x'*250))

    def test_commits_while_endpoint_is_held_and_closes_with_only_uncommitted_tail(self):
        u = StreamingUtterance()
        feed(u, 1.2)
        first = u.request(True)
        self.assertEqual(u.apply(result('My name is Shai and'), first), '')
        feed(u, 1.2)
        second = u.request(True)
        prefix = u.apply(result('My name is Shai and I have a question.'), second)
        self.assertEqual(prefix, 'My name is Shai and')
        self.assertFalse(u.ready(True))
        feed(u, 1, 0)
        self.assertFalse(u.ready(True))
        self.assertFalse(u.ready(False))
        self.assertEqual(u.apply(result('My name is Shai and I have a question.'), u.request(True)), ' I have a')
        self.assertTrue(u.ready(False))
        text, tail = u.finish()
        self.assertEqual(tail, ' question.')
        self.assertEqual(prefix+' I have a'+tail, text)

    def test_asr_snapshot_does_not_block_or_erase_incoming_audio(self):
        u = StreamingUtterance()
        feed(u, 1.2)
        snapshot = u.request(True)
        before = bytes(u.audio)
        feed(u, .4)
        self.assertEqual(snapshot['pcm'], before)
        self.assertGreater(len(u.audio), len(before))
        u.apply(result('a growing utterance'), snapshot)
        self.assertLess(u.covered_voice, u.last_voice)
        feed(u, 1, 0)
        self.assertFalse(u.ready(False))
        final = u.request(False)
        self.assertEqual(final['mode'], 'final')
        u.apply(result('a growing utterance finishes.'), final)
        self.assertTrue(u.ready(False))

    def test_tentative_final_pass_is_reused_but_resumed_speech_invalidates_it(self):
        u = StreamingUtterance()
        feed(u, .6)
        feed(u, .32, 0)
        request = u.request(True)
        self.assertEqual(request['mode'], 'settle')
        u.apply(result('Hello there.'), request)
        feed(u, .6, 0)
        self.assertIsNone(u.request(True))
        self.assertTrue(u.ready(False))
        feed(u, .2)
        self.assertFalse(u.ready(False))
        feed(u, .82, 0)
        self.assertIsNotNone(u.request(False))

    def test_unstable_tail_is_not_committed_and_raw_names_are_preserved(self):
        u = StreamingUtterance()
        feed(u, 1.2)
        self.assertEqual(u.apply(result('Hello Che'), u.request(True)), '')
        feed(u, 1.2)
        self.assertEqual(u.apply(result('Hello Shai this is'), u.request(True)), 'Hello')
        feed(u, 1.2)
        self.assertEqual(u.apply(result('Hello Shai this is Smith.'), u.request(True)), ' Shai this is')
        self.assertEqual(u.finish(), ('Hello Shai this is Smith.', ' Smith.'))

    def test_confirmed_segments_trim_without_removing_new_audio_or_committed_text(self):
        u = StreamingUtterance()
        feed(u, 2)
        u.apply(result('one two three four'), u.request(True))
        feed(u, 1.2)
        request = u.request(True)
        feed(u, .4)
        total = len(u.audio)
        prefix = u.apply(result('one two three four five', [(16000, 2), (48000, 5)]), request)
        self.assertEqual(prefix, 'one two three four')
        self.assertEqual(len(u.audio), total-32000)
        self.assertEqual(u.base, 16000)
        self.assertEqual(u.local_committed, 2)
        self.assertEqual(u.finish(), ('one two three four five', ' five'))
        self.assertEqual(u.committed, ['one', 'two', 'three', 'four'])

    def test_noise_and_unbounded_unconfirmed_audio_do_not_create_partial_turns(self):
        u = StreamingUtterance(window=2)
        feed(u, .1)
        feed(u, .9, 0)
        self.assertTrue(u.ready(False))
        self.assertIsNone(u.request(False))
        self.assertEqual(u.finish(), ('', ''))
        with self.assertRaisesRegex(RuntimeError, 'unconfirmed window'):
            feed(u, 1.1)

    def test_vtt_preserves_unicode_numbers_and_cue_word_boundaries(self):
        data = parse_vtt('WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n[Music] Hello, Chéy.\n\n'
                         '00:00:01.500 --> 00:00:02.000\n42\n')
        self.assertEqual(data, dict(words=['Hello,', 'Chéy.', '42'], segments=[(24000, 2), (32000, 3)]))
        with self.assertRaises(ValueError): parse_vtt('{"error":"bad format"}')

    def test_typed_commits_do_not_consume_turn_budget_and_cannot_leak_across_pause(self):
        d = WindowsDeviceMixin.__new__(WindowsDeviceMixin)
        d.lock = threading.RLock()
        d.state = dict(ready=True, phase='hearing', listening=True, input_count=0, input_commits=0, turn_limit=1)
        d.turn_active = False
        d.capture_epoch = 1
        d.input_prefix = ''
        d.input_client = SimpleNamespace(stdin=io.BytesIO())
        d.prompts = queue.Queue(maxsize=1)
        d.emit = Mock()
        d.commit_input('Hello Chéy')
        raw = d.input_client.stdin.getvalue()
        self.assertEqual(raw[:2], b'\1T')
        self.assertEqual(struct.unpack('<HHI', raw[2:10]), (0, 0, len('Hello Chéy'.encode())))
        self.assertEqual(raw[10:].decode(), 'Hello Chéy')
        self.assertEqual(d.state['input_count'], 0)
        self.assertNotIn(b'\n', raw)
        with self.assertRaisesRegex(ValueError, 'Stop'): d.listen(False)
        with self.assertRaisesRegex(ValueError, 'committed'): d.command('prompt unrelated')
        self.assertEqual(d.capture_epoch, 1)
        d.admit('Hello Chéy, welcome.', ', welcome.')
        self.assertEqual(d.prompts.get_nowait(), ('Hello Chéy, welcome.', ', welcome.', None))
        self.assertEqual(d.state['input_count'], 1)
        self.assertFalse(d.state['listening'])
        self.assertTrue(d.state['finishing'])
        self.assertEqual(d.input_prefix, '')


if __name__ == '__main__': unittest.main()

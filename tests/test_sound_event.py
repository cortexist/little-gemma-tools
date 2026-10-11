"""Acoustic metadata must remain request-local and must not grant identity."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import struct
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui'),
               str(Path(__file__).resolve().parents[1]/'script')]
from sound_event import (CAPTURE_BOUNDARY, INPUT_FORMAT, SOUND_POLICY, align,
                         compact, speech_suffix, standalone_prompt, tag_policy, validate)
from room_direct import DirectRoomRuntime
from room_policy import DIRECT_ROOM_TAG_POLICY
from streaming_asr import StreamingUtterance
from whisper_sound_server import Adapter


def event(seconds=1):
    return dict(schema='sound-event/1', span_seconds=[0, seconds],
                audio_available_seconds=seconds, nominal_window_seconds=10,
                score_semantics='raw_uncalibrated_logits',
                candidates=[dict(label='Music', class_index=137, logit=1.23),
                            dict(label='Speech', class_index=0, logit=-2.24)])


class SoundTests(unittest.TestCase):
    def test_low_speech_is_preserved_as_an_uncertain_candidate(self):
        self.assertEqual(compact(event()), ' SOUND_EVENT {"t":[0,1],"c":[["Music",1.2],["Speech",-2.2]]}')
        self.assertEqual(speech_suffix(' ROOM_INPUT {} FIRST output [speak] or [defer].', event()),
                         compact(event())+' ROOM_INPUT {} FIRST output [speak], [defer] or [wait].')
        self.assertEqual(speech_suffix(' ROOM_INPUT {}'), ' ROOM_INPUT {}')
        self.assertIn(SOUND_POLICY, tag_policy(DIRECT_ROOM_TAG_POLICY))

    def test_span_never_includes_padding_and_labels_are_canonical(self):
        for update in [dict(span_seconds=[0, 30]), dict(score_semantics='probabilities'),
                       dict(candidates=[dict(label='Agent Smith', class_index=0, logit=1)])]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate(dict(event(), **update))

    def test_trimmed_request_retains_utterance_relative_time(self):
        request = dict(pcm=bytes(32000), base=64000, end=80000)
        self.assertEqual(align(event(), request)['span_seconds'], [4, 5])
        with self.assertRaises(ValueError): align(event(2), request)
        utterance = StreamingUtterance()
        utterance.base = 64000
        utterance.apply(dict(words=['Hello'], segments=[], sound_event=event()),
                        dict(request, mode='settle'))
        self.assertEqual(utterance.sound_event['span_seconds'], [4, 5])
        self.assertIsNone(StreamingUtterance().sound_event)

    def test_sound_only_does_not_mutate_partner_or_speaker_session(self):
        device = Mock(cfg=dict(turn_head=dict(prompt_lane='v2_sound_event', mode='control',
                                               text_decision_tags=True),
                               acoustic_input=dict(mode='whisper_at_v2')),
                      lock=threading.RLock(), cancel=threading.Event(), procs={},
                      state=dict(synth_count=1, listening=True))
        room = DirectRoomRuntime(device, dict(session='s', topic='class'))
        room.conversation = [dict(own_speech='Would you explain?')]
        room.epoch = 3
        room.identities = Mock()
        room.respond = Mock(); room.release_engine = Mock()
        live = Mock(); live.finish.return_value = dict(tracks=['t_unknown'], status='new_voice')
        room.decision(('observe', '', '', dict(sound_event=event(), admitted_at=0), b'', 99, live), None)
        room.identities.observe.assert_not_called()
        self.assertEqual(room.epoch, 3)
        self.assertEqual(room.heard, 0)
        self.assertEqual(room.conversation, [dict(own_speech='Would you explain?')])
        self.assertIsNone(room.last_observation)
        args = room.respond.call_args.args
        self.assertIsNone(args[3])
        self.assertEqual(args[1], standalone_prompt(event(), 1, True))
        self.assertNotIn('ROOM_INPUT', args[1])
        self.assertNotIn('NAME_UPDATE', args[1])

    def test_v2_head_contract_is_passed_explicitly(self):
        device = Mock(cfg=dict(turn_head=dict(prompt_lane='v2_sound_event', mode='control',
                                               text_decision_tags=True), acoustic_input=dict(mode='whisper_at_v2')),
                      lock=threading.RLock(), cancel=threading.Event(), procs={}, state={})
        room = DirectRoomRuntime(device, dict(session='s', topic='class'))
        self.assertTrue(room.sound_input)
        self.assertIn('[wait]', room.policy)
        self.assertEqual(INPUT_FORMAT, 'trio_sound_event_v2_staged_history')
        self.assertEqual(CAPTURE_BOUNDARY, 'post-output-norm final prompt token; before generated-token forward')

    def test_only_explicit_teacher_opening_uses_the_user_speak_grant(self):
        for opening, expected in [(True, 'speak'), (False, 'defer')]:
            device = Mock(cfg=dict(turn_head=dict(prompt_lane='v2_sound_event', mode='control',
                                                   text_decision_tags=True), acoustic_input=dict(mode='whisper_at_v2')),
                          lock=threading.RLock(), cancel=threading.Event(), procs={}, state={})
            room = DirectRoomRuntime(device, dict(session='s', topic='class'))
            room.client = Mock(stdin=io.BytesIO())
            room.shadow_connection = 1; room.generation_open = True
            room.current = dict(id='1', heard=None, opening=opening, needs_name=False)
            room.responses = {'1':room.current}
            raw = dict(action='defer', answer_channel_matches=True, snapshot=dict(turn=1),
                       probabilities=dict(speak=0, defer=1, wait=0), ready_monotonic=time.monotonic())
            room.apply_head_decision(raw,dict(session='s', response='1', connection=1))
            self.assertEqual(room.current['control_action'], expected)
            self.assertEqual(raw['action'], 'defer')
            self.assertEqual(room.current['control_source'], 'explicit_room_open' if opening else 'argmax')
            self.assertEqual(struct.unpack('<BIBB', room.client.stdin.getvalue())[2], 1 if opening else 2)

    def test_listener_or_sound_cannot_request_synthetic_opening(self):
        device = Mock(cfg={}, lock=threading.RLock(), cancel=threading.Event(), procs={}, state={})
        room = DirectRoomRuntime(device, dict(session='s', topic='class'))
        room.opened = True
        room.speaker = dict(role='student')
        with self.assertRaises(ValueError):
            room.respond('', 'ROOM_OPEN', dict(admitted_at=0), None, 0, opening=True)
        room.speaker = dict(role='teacher');room.opened = False
        with self.assertRaises(ValueError):
            room.respond('', 'ROOM_OPEN', dict(admitted_at=0), None, 0, opening=True)

    def test_stale_features_removed_and_concurrent_requests_serialized(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = Adapter.__new__(Adapter)
            adapter.features = Path(d)/'encoder.bin'
            adapter.features.write_bytes(b'stale')
            adapter.settings = dict(backend_url='http://127.0.0.1:1')
            adapter.serial = 0; adapter.lock = threading.Lock()
            adapter.head = Mock(sha='model')
            def predict(path, seconds):
                value = int(path.read_text())
                return dict(event(seconds), test_request=value), .001
            adapter.head.predict.side_effect = predict
            def backend(request, timeout):
                self.assertFalse(adapter.features.exists())
                value = int(request.data)
                adapter.features.write_text(str(value))
                time.sleep(.01)
                response = io.BytesIO(('VTT '+str(value)).encode())
                response.headers = {'Content-Type':'text/vtt'}
                return response
            results = {}
            with patch('whisper_sound_server.audio_seconds', return_value=1), \
                 patch('whisper_sound_server.urllib.request.urlopen', side_effect=backend):
                def run(value):
                    raw, _ = adapter.infer(str(value).encode(), 'multipart/form-data', True)
                    results[value] = json.loads(raw)
                threads = [threading.Thread(target=run, args=(n,)) for n in range(6)]
                for t in threads: t.start()
                for t in threads: t.join()
                self.assertEqual(len(results), 6)
                for n, result in results.items():
                    self.assertEqual(result['vtt'], 'VTT '+str(n))
                    self.assertEqual(result['sound_event']['test_request'], n)
                self.assertFalse(adapter.features.exists())
                adapter.head.predict.side_effect = ValueError('no usable features')
                raw, _ = adapter.infer(b'8', 'multipart/form-data', True)
                failure = json.loads(raw)
                self.assertEqual(failure['vtt'], 'VTT 8')
                self.assertIsNone(failure['sound_event'])
                self.assertEqual(failure['sound_error'], 'no usable features')


if __name__ == '__main__': unittest.main()

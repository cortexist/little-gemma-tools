"""Duplex input ordering, budget races and acoustic yielding are protocol invariants."""
import io
from pathlib import Path
import queue
import sys
import threading
import unittest
from unittest.mock import Mock
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui'), str(Path(__file__).resolve().parents[1]/'script')]
from room_direct import DirectRoomRuntime, SelfAwareVoices
from room_policy import accept_response_names, needs_response_name
from session_speakers import SpeakerSession
from trio_direct import SpokenBudget


def runtime():
    device = Mock()
    device.cfg = {}
    device.lock = threading.RLock()
    device.cancel = threading.Event()
    device.state = dict(listening=True, input_commits=0, synth_count=0)
    device.procs = {}
    r = DirectRoomRuntime(device, dict(session='session', topic='agents'))
    r.client = Mock(stdin=io.BytesIO())
    r.identities = SpeakerSession([dict(name='Mrs. Byers', embedding=[1,0], trusted_introducer=True)], canonicalize=False)
    return r


class DirectTests(unittest.TestCase):
    def test_visual_cues_wait_for_permitted_pcm_and_remain_out_of_tts(self):
        for permit, interrupted in [(True, False), (False, False), (True, True)]:
            with self.subTest(permit=permit, interrupted=interrupted):
                r = runtime(); r.text_decision_tags = True
                r.cfg = dict(room_identity={}, name_pronunciations={})
                r.state.update(input_count=0, reply_count=0, synth_bytes=0, played_bytes=0,
                               audible_until=0, history=[], capture_active=False)
                r.frames = queue.Queue(); r.replies = queue.Queue(); r.speech_rate = 16000
                r.synth_sent = r.synth_completed = 0
                spoken = []
                def synthesize(data):
                    spoken.append(data)
                    self.assertFalse({'expression', 'gesture'} &
                                     {c.args[0] for c in r.device.emit.call_args_list})
                    r.frames.put((b'P', bytes(640)))
                    r.frames.put((b'P', bytes(640)))
                    r.frames.put((b'D', str(r.synth_sent).encode()))
                    r.current['permission'] = permit
                    if interrupted: r.yielded.set()
                r.synth = Mock(stdin=Mock(write=synthesize))
                r.open_output = Mock(return_value=Mock(latency=0))
                r.replies.put(b'[speak][happy][nod]Hello.<turn|>')
                r.respond('Heard', 'Heard', dict(admitted_at=0), None, 0)
                visuals = [(c.args[0], c.args[1]) for c in r.device.emit.call_args_list
                           if c.args[0] in ('expression', 'gesture')]
                self.assertEqual(spoken, [b'Hello.\n'])
                self.assertEqual(visuals, [('expression', 'happy'), ('gesture', 'nod')]
                                 if permit and not interrupted else [])
                if permit and not interrupted:
                    self.assertEqual(r.state['expression'], 'happy')
                    self.assertGreater(r.state['gesture_until'], 0)
                else: r.open_output.assert_not_called()

    def test_one_request_streams_speech_or_defers_without_control_audio(self):
        import tempfile
        for wire, spoken in [(b'[defer]<turn|>', False),
                             (b'Hello there.<turn|>', True),
                             (b'[speak]Hello there.<turn|>', True),
                             (b'[defer] not valid speech<turn|>', False)]:
            r = runtime()
            r.text_decision_tags = True
            r.cfg = dict(room_identity={}, name_pronunciations={})
            r.state.update(input_count=0, reply_count=0, synth_bytes=0, played_bytes=0,
                           audible_until=0, history=[], capture_active=False)
            r.directory = Path(tempfile.mkdtemp())
            r.replies = queue.Queue(); r.replies.put(wire)
            r.frames = queue.Queue(); r.speech_rate = 16000
            r.synth_sent = r.synth_completed = 0
            spoken_clauses = []
            def synthesize(data):
                spoken_clauses.append(data)
                r.frames.put((b'A', b'0\t20\th\n'))
                r.frames.put((b'P', bytes(640)))
                r.frames.put((b'D', str(r.synth_sent).encode()))
            r.synth = Mock(stdin=Mock(write=synthesize))
            output = Mock(latency=0)
            r.open_output = Mock(return_value=output)
            original = r.publish
            def publish(**extra):
                if r.current and r.current['status'] == 'candidate': r.current['permission'] = True
                original(**extra)
            r.publish = publish
            r.respond('Input', 'Input', dict(admitted_at=0), None, 0)
            self.assertEqual(r.request_count, 1)
            self.assertEqual(r.client.stdin.getvalue(), b'Input\n')
            self.assertEqual(r.state['synth_count'], int(spoken))
            self.assertEqual(r.responses['1']['spoken'], spoken)
            if wire.startswith(b'[speak]'):
                self.assertEqual(r.responses['1']['text_decision']['source'], 'explicit_tag')
            self.assertEqual(spoken_clauses, [b'Hello there.\n'] if spoken else [])
            self.assertEqual(sum(c.args[0] == 'alignment' for c in r.device.emit.call_args_list), int(spoken))
            if not spoken: r.open_output.assert_not_called()

    def test_timing_join_handles_both_orders_once_without_changing_permissions(self):
        for order in ('head', 'text_decision'):
            r = runtime()
            record = dict(id='1', heard='h1', permission=None, status='generating',
                          request_sent_monotonic=9.)
            items = dict(head=dict(action='wait', ready_monotonic=10.001, ready_at=100.001,
                snapshot=dict(transport_received_monotonic=10., transport_read_serial=1, snapshot_at=100.)),
                text_decision=dict(action='speak', source='explicit_tag', metadata_before=0,
                    received_at=100.2, received_monotonic=10.2, read_serial=2,
                    recognized_monotonic=10.21, receipt_source='transport_read'))
            record[order] = items.pop(order)
            r.compare_text_timing(record)
            self.assertNotIn('decision_timing', record)
            record.update(items)
            r.compare_text_timing(record); r.compare_text_timing(record)
            self.assertAlmostEqual(record['decision_timing']['text_minus_head_s'], .199)
            self.assertEqual(record['status'], 'generating')
            self.assertIsNone(record['permission'])
            self.assertFalse(r.yielded.is_set())
            self.assertEqual(sum(c.args[0]=='turn_decision_timing' for c in r.device.emit.call_args_list), 1)

    def test_audio_barge_drains_cancelled_piper_frames_and_next_response_still_works(self):
        r=runtime(); r.cfg=dict(room_identity={}, name_pronunciations={})
        r.state.update(input_count=0,reply_count=0,synth_bytes=0,played_bytes=0,audible_until=0,
                       history=[],capture_active=False)
        r.frames=queue.Queue(); r.replies=queue.Queue(); r.speech_rate=16000
        r.synth_sent=r.synth_completed=0
        def synthesize(data):
            r.frames.put((b'A', b'0\t200\ta\n'))
            r.frames.put((b'P',bytes(6400)))
            r.frames.put((b'D',str(r.synth_sent).encode()))
        r.synth=Mock(stdin=Mock(write=synthesize))
        original=r.publish
        def publish(**extra):
            if r.current and r.current['status']=='candidate': r.current['permission']=True
            original(**extra)
        r.publish=publish
        first=Mock(latency=0)
        first.write.side_effect=lambda data:r.barge(dict(track_id='voice-1',name='Mrs. Byers'))
        second=Mock(latency=0)
        r.open_output=Mock(side_effect=[first,second])
        for wire in (b'An interrupted answer.<turn|>', b'Please go ahead.<turn|>'):
            r.replies.put(wire)
            r.respond('Heard', 'Heard', dict(admitted_at=0), None, 0)
        self.assertEqual(r.responses['1']['status'],'interrupted')
        self.assertEqual(r.responses['2']['status'],'spoken')
        self.assertEqual(r.request_count,2)
        self.assertEqual(r.state['synth_count'],2)
        self.assertEqual(r.synth_sent,r.synth_completed)
        self.assertTrue(r.frames.empty())
        first.abort.assert_called_once()
        second.stop.assert_called_once()
        r.device.terminate_owned.assert_not_called()
        events = [c.args for c in r.device.emit.call_args_list]
        self.assertEqual(sum(e[0] == 'cut' for e in events), 1)
        alignments = [e[1] for e in events if e[0] == 'alignment']
        self.assertEqual([int(a.split('\t')[0]) for a in alignments], [0,640])

    def test_visual_timing_arrives_before_the_second_blocking_pcm_write(self):
        r=runtime(); r.cfg=dict(room_identity={}, name_pronunciations={})
        r.state.update(input_count=0,reply_count=0,synth_bytes=0,played_bytes=0,audible_until=0,
                       history=[],capture_active=False)
        r.frames=queue.Queue(); r.replies=queue.Queue(); r.speech_rate=16000
        r.synth_sent=r.synth_completed=0
        def synthesize(data):
            r.frames.put((b'A', b'0\t200\tn\n200\t200\to\n'))
            r.frames.put((b'P',bytes(6400)))
            r.frames.put((b'D',str(r.synth_sent).encode()))
        r.synth=Mock(stdin=Mock(write=synthesize))
        original=r.publish
        def publish(**extra):
            if r.current and r.current['status']=='candidate': r.current['permission']=True
            original(**extra)
        r.publish=publish
        output=Mock(latency=0)
        writes=[]
        def write(data):
            kinds=[c.args[0] for c in r.device.emit.call_args_list]
            self.assertIn('alignment',kinds)
            if writes: self.assertIn('play',kinds)
            writes.append(data)
        output.write.side_effect=write
        r.open_output=Mock(return_value=output)
        r.replies.put(b'No. Never.<turn|>')
        r.respond('Heard','Heard',dict(admitted_at=0),None,0)
        events=[c.args for c in r.device.emit.call_args_list]
        self.assertEqual([e[1] for e in events if e[0]=='clause'],['No.','Never.'])
        self.assertEqual([int(e[1].split('\t')[0]) for e in events if e[0]=='alignment'],[0,6400])

    def test_budget_denied_synthesis_never_animates(self):
        r=runtime(); r.cfg=dict(room_identity={},name_pronunciations={})
        r.state.update(input_count=0,reply_count=0,synth_bytes=0,played_bytes=0,audible_until=0,
                       history=[],capture_active=False)
        r.frames=queue.Queue(); r.replies=queue.Queue(); r.speech_rate=16000
        r.synth_sent=r.synth_completed=0
        def synthesize(data):
            r.frames.put((b'A',b'0\t20\th\n'));r.frames.put((b'P',bytes(640)))
            r.frames.put((b'D',str(r.synth_sent).encode()))
            r.current['permission']=False
        r.synth=Mock(stdin=Mock(write=synthesize));r.open_output=Mock()
        r.replies.put(b'Hello.<turn|>')
        r.respond('Heard','Heard',dict(admitted_at=0),None,0)
        r.open_output.assert_not_called()
        self.assertFalse({'clause','alignment','play'} & {c.args[0] for c in r.device.emit.call_args_list})

    def test_engine_busy_buffers_commits_without_interrupting_generation(self):
        r = runtime(); r.engine_busy = True
        r.commit_input('Hello'); r.commit_input(' everyone')
        self.assertEqual(r.client.stdin.getvalue(), b'')
        self.assertEqual(r.input_prefix, 'Hello everyone')
        r.release_engine()
        self.assertEqual(r.client.stdin.getvalue(), b'\x01T\x00\x00\x00\x00\x0e\x00\x00\x00Hello everyone')
        self.assertEqual(r.capture_sent, r.input_prefix)
        r.commit_input(' today')
        self.assertEqual(r.capture_sent, 'Hello everyone today')

    def test_admission_keeps_only_unsent_text_and_does_not_close_capture(self):
        r = runtime(); r.live = Mock(); r.audio = b'x'*32000
        r.commit_input('Hello')
        r.admit('Hello everyone', ' everyone')
        job = r.prompts.get_nowait()
        self.assertEqual(job[2], ' everyone')
        self.assertTrue(r.capture_enabled())
        self.assertEqual(r.input_prefix, '')
        r.commit_input('A second voice')
        self.assertEqual(r.capture_sent, '')
        self.assertEqual(r.input_prefix, 'A second voice')

    def test_self_competes_without_entering_acquaintance_roster(self):
        s = SpeakerSession([dict(name='Teacher', embedding=[1,0,0])])
        voices = SelfAwareVoices(s, [0,1,0])
        self.assertEqual(voices.preview([0,1,0])['track_id'], voices.SELF)
        self.assertEqual(voices.preview([1,0,0])['track_id'], 'voice-1')
        self.assertEqual(voices.preview([1,1,0])['status'], 'uncertain')
        self.assertEqual(voices.preview([0,0,1])['status'], 'unknown')
        self.assertEqual(len(s.tracks), 1)

    def test_barge_sends_one_engine_interrupt_without_resetting_capture(self):
        r = runtime(); r.playing = r.generation_open = True
        r.current = dict(id='1'); r.input_prefix = 'Incoming speech'
        who = dict(track_id='voice-1', name='Mrs. Byers')
        r.barge(who); r.barge(who)
        self.assertEqual(r.client.stdin.getvalue(), b'\x02')
        self.assertEqual(r.input_prefix, 'Incoming speech')
        self.assertFalse(r.cancel.is_set())
        self.assertTrue(r.yielded.is_set())
        r.device.terminate_owned.assert_not_called()

    def test_permits_are_idempotent_and_session_scoped(self):
        r = runtime(); r.responses['1'] = dict(permission=None)
        data = dict(session='session', response='1', allowed=True)
        r.control('room-permit', data); r.control('room-permit', data)
        with self.assertRaises(ValueError): r.control('room-permit', dict(data, allowed=False))
        with self.assertRaises(ValueError): r.control('room-permit', dict(data, session='old'))

    def test_budget_allows_overlap_but_reserves_last_slot_once(self):
        b = SpokenBudget(2)
        self.assertTrue(b.reserve('teacher', '1'))
        self.assertTrue(b.reserve('student', '1'))
        self.assertFalse(b.reserve('guest', '1'))
        self.assertTrue(b.reserve('student', '1'))
        self.assertEqual(b.occupied, 2)
        self.assertFalse(b.complete)
        states = {n:dict(room=dict(responses={'1':dict(done=True, spoken=True)})) for n in ('teacher','student')}
        b.refresh(states); self.assertTrue(b.complete)
        self.assertFalse(b.reserve('student', '2'))

    def test_silent_cancel_releases_reservation_audible_barge_consumes_one(self):
        b = SpokenBudget(2); b.reserve('a','1'); b.reserve('b','1')
        b.refresh({'a':dict(room=dict(responses={'1':dict(done=True, spoken=False)})),
                   'b':dict(room=dict(responses={'1':dict(done=True, spoken=True)}))})
        self.assertEqual(b.occupied, 1)
        self.assertTrue(b.reserve('c','1'))
        self.assertFalse(b.complete)

    def test_trusted_special_guest_introduction_is_evidence_but_question_is_not(self):
        for text, accepted in [('Today we meet our special guest, Agent Smith.', True),
                               ('What did our special guest Agent Smith say?', False)]:
            s = runtime().identities
            o = s.observe('h1', [1,0], text, clean=True, duration=3)
            result = accept_response_names(s, o, ['Agent Smith'])
            self.assertEqual(bool(result['accepted']), accepted)

    def test_guest_does_not_relearn_their_own_introduction(self):
        s=runtime().identities
        o=s.observe('h1',[1,0],'Today we meet our special guest, Agent Smith.',clean=True,duration=3)
        guest=dict(display_name='Agent Smith',role='guest')
        self.assertFalse(needs_response_name(s,o,guest))
        self.assertFalse(accept_response_names(s,o,[],guest)['rejected'])
        self.assertTrue(needs_response_name(s,o,dict(display_name='Chey',role='student')))

    def test_name_suffix_uses_current_acoustic_evidence_not_model_ids(self):
        r = runtime(); s = r.identities
        o = s.observe('h1', [1,0], 'I have invited Agent Smith.', clean=True, duration=3)
        result = accept_response_names(s,o,['Agent Smith'])
        self.assertFalse(result['rejected']); self.assertIn('Agent Smith',s.pending_names)
        o = s.observe('h2', [0,1], 'Hello, I am Agent Smith.', clean=True, duration=3)
        result = accept_response_names(s,o,['Agent Smith'])
        self.assertFalse(result['rejected']); self.assertEqual(s.tracks['voice-2']['name'],'Agent Smith')
        o = s.observe('h3',[0,1], 'What do you think?', clean=True,duration=3)
        self.assertTrue(accept_response_names(s,o,['Chey'])['rejected'])
        self.assertEqual(s.tracks['voice-2']['name'],'Agent Smith')

if __name__ == '__main__': unittest.main()

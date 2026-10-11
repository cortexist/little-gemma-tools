"""Shared budgets and listener-driven routing, independent of panel order."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch, call as mock_call
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui'), str(Path(__file__).resolve().parents[1]/'script')]
from trio_session import TrioSession, TurnProgress
from room_policy import choose, parse_decision


def ready(count=0, **changes):
    result = dict(ready=True, error='', input_count=count, reply_count=count, synth_count=count,
                  synth_bytes=count*100, played_bytes=count*100, audible_until=0, phase='paused')
    result.update(changes)
    return result


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = {key:dict(id=key, speaker=dict(display_name=key, role=role)) for key, role in
                      [('teacher','teacher'), ('student','student'), ('guest','guest')]}
        self.calls = []
        self.call = lambda node, action, data=None: self.calls.append((node['id'], action, data))
        self.status = Mock(return_value=dict(capabilities=dict(pipeline=True, capture=True, room_session=True),
                                             profiles={'test':{}}, state=ready()))
        self.rec = Mock(); self.rec.snapshot.return_value = {'phase':'idle', 'error':''}
        self.session = TrioSession(self.nodes, self.call, self.status, self.rec)
        self.data = dict(topic='Agents', total_turns=6, nodes={k:dict(profile='test', system_prompt='Own knowledge') for k in self.nodes})

    def seed(self):
        self.session.update(id='12345678-1234-1234-1234-123456789012', active=True, total_turns=6,
                            turns={k:dict(accepted=0, completed=0) for k in self.nodes})

    def test_shared_budget_finishes_unequal_counts_after_last_audio(self):
        self.seed()
        states = dict(teacher=ready(1), student=ready(2), guest=ready(3, phase='speaking', played_bytes=290))
        self.assertFalse(self.session.observe(states))
        states['guest'] = ready(3, audible_until=1e12)
        self.assertFalse(self.session.observe(states))
        states['guest'] = ready(3)
        self.assertTrue(self.session.observe(states))
        self.assertEqual(self.session.snapshot()['completed'], 6)

    def test_observations_and_decisions_do_not_consume_turns(self):
        self.seed()
        states = {k:ready(1, room=dict(heard_count=19, decision=dict(action='defer'))) for k in self.nodes}
        self.assertFalse(self.session.observe(states))
        self.assertEqual(self.session.snapshot()['completed'], 3)

    def test_limits_validated_before_mutating(self):
        for bad in (0, -1, 49, 1.5, True, '4'):
            self.data['total_turns'] = bad
            with self.assertRaises(ValueError): self.session.start(self.data)
        self.assertEqual(self.calls, [])
        self.status.assert_not_called()

    def test_room_capability_required_on_every_node(self):
        self.status.return_value['capabilities']['room_session'] = False
        with self.assertRaisesRegex(ValueError, 'Cannot start'): self.session.start(self.data)
        self.assertEqual(self.calls, [])

    def test_restart_or_shared_overrun_fails(self):
        self.seed()
        self.session.observe(dict(teacher=ready(2), student=ready(), guest=ready()))
        with self.assertRaisesRegex(RuntimeError, 'restarted'):
            self.session.observe(dict(teacher=ready(1), student=ready(), guest=ready()))
        with self.assertRaisesRegex(RuntimeError, 'exceeded'):
            self.session.observe(dict(teacher=ready(7), student=ready(), guest=ready()))

    def test_every_listener_prepared_before_one_grant_without_text_relay(self):
        self.seed(); self.session.grant('student', 3)
        self.assertEqual([a for _,a,_ in self.calls], ['room-prepare']*3+['room-grant'])
        self.assertEqual([b['speaking'] for _,a,b in self.calls if a=='room-prepare'], [False,True,False])
        self.assertFalse(any('text' in b or 'speaker_id' in b for _,_,b in self.calls))

    def test_stop_cleans_every_node_despite_one_failure(self):
        def call(node, action, data=None):
            self.calls.append((node['id'],action))
            if node['id']=='student': raise OSError('offline')
        self.session.call = call; self.session.stop(); self.session.thread.join(timeout=2)
        self.assertEqual(set(self.calls), {(k,'stop') for k in self.nodes})
        self.assertIn('student: offline', self.session.snapshot()['error'])
        self.assertEqual(self.rec.stop.call_args_list, [mock_call(wait=False), mock_call(wait=True)])

    def test_digital_recording_shutdown_begins_before_its_audio_sources_stop(self):
        self.rec.snapshot.return_value = dict(phase='recording', audio_mode='digital', error='')
        order = []
        self.rec.stop.side_effect = lambda wait=False: order.append(('recording', wait))
        self.session.call = lambda node, action, data=None: order.append((node['id'], action))
        self.session.finish()
        self.assertEqual(order[0], ('recording', False))
        self.assertEqual(order[-1], ('recording', True))

    def test_room_recording_stops_asr_taps_before_pipeline_shutdown(self):
        self.rec.snapshot.return_value = dict(phase='recording', audio_mode='room', error='')
        order = []
        self.rec.stop.side_effect = lambda wait=False: order.append(('recording', wait))
        self.session.call = lambda node, action, data=None: order.append((node['id'], action))
        self.session.finish()
        self.assertEqual(order[0], ('recording', False))
        self.assertEqual(order[-1], ('recording', True))
        self.assertEqual({key for key, action in order[1:-1]}, set(self.nodes))

    def test_recorder_shutdown_failure_still_stops_all_nodes(self):
        self.rec.snapshot.return_value = dict(phase='recording', audio_mode='digital', error='')
        self.rec.stop.side_effect = OSError('recorder unavailable')
        self.session.finish()
        self.assertEqual({key for key, action, data in self.calls if action == 'stop'}, set(self.nodes))
        self.assertIn('recorder unavailable', self.session.snapshot()['error'])

    def test_progress_includes_silent_prefill_but_not_ambient_capture(self):
        progress = TurnProgress(0); state = ready()
        for second in range(0,241,20):
            state['input_commits'] = second; progress.check({'guest':state}, second, second)
        state['capture_blocks'] = 100000
        with self.assertRaisesRegex(RuntimeError, '90 seconds'): progress.check({'guest':state},331,331)

    def test_all_defer_has_no_forced_scheduled_successor(self):
        self.assertIsNone(choose({k:dict(action='defer',reason='defer',at=1) for k in self.nodes}, 'guest'))

    def test_explicit_chey_question_preempts_teacher_despite_panel_order(self):
        raw = '{"action":"speak","reason":"answer"}'
        question = 'Chey, what do you think an agent needs?'
        teacher = parse_decision(raw, question, ['Mrs. Byers'], ['Chey','Agent Smith'])
        student = parse_decision(raw, question, ['Chey','Shy'], ['Mrs. Byers'])
        teacher['at'], student['at'] = 0, 1
        self.assertEqual(teacher['action'], 'defer')
        self.assertEqual(choose(dict(teacher=teacher,student=student), 'guest'), 'student')

    def test_name_mention_is_not_address_and_malformed_decision_rejected(self):
        raw = '{"action":"defer","reason":"defer"}'
        result = parse_decision(raw, 'Agent Smith said Chey asked a good question.', ['Chey'], ['Agent Smith'])
        self.assertEqual(result['action'],'defer')
        for text in ('I should answer.', '{"action":"speak","action":"defer","reason":"answer"}'):
            with self.assertRaises(ValueError): parse_decision(text, 'Hi', ['Chey'], [])

if __name__=='__main__': unittest.main()

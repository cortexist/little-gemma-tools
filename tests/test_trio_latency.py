"""Timing boundaries, host clock skew, incomplete telemetry and durable sessions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script/deviceui'))
from trio_latency import TrioLatency


def event(i, kind, at, text='', **extra):
    return dict(id=i, kind=kind, at=at, text=text, **extra)


def status(events=(), offset=0, cursor=None):
    return dict(instance_id='node', server_time=100+offset, events=list(events),
                cursor=cursor if cursor is not None else max((e['id'] for e in events), default=0))


class LatencyTests(unittest.TestCase):
    def monitor(self, root=None):
        m = TrioLatency(root)
        m.begin(dict(id='session', profiles=dict(teacher='12b', guest='12b')),
                dict(teacher=status(), guest=status()))
        return m

    def feed(self, m):
        # Guest events arrive first. Its clock is 20s ahead of the teacher.
        guest = [event(1, 'asr', 124, elapsed=.9), event(2, 'input_done', 124.01),
                 event(3, 'model_first_token', 124.31), event(4, 'clause', 124.51),
                 event(5, 'speech_first_pcm', 124.61),
                 event(6, 'play', 124.75, '124.65\t0\t2000\t1000'), event(7, 'synth_done', 125.65)]
        teacher = [event(1, 'input_done', 100), event(2, 'clause', 100.15),
                   event(3, 'play', 100.2, '100.2\t0\t4000\t1000'), event(4, 'synth_done', 100.3)]
        for key, events, offset in [('guest', guest, 20), ('teacher', teacher, 0)]:
            for _ in range(2):  # Repeated status snapshots must not double-count.
                m.ingest(key, status(events, offset), 99.99, 100.01)
        return teacher, guest

    def test_pcm_ready_and_pipe_proxy_are_distinct_and_clock_skew_is_removed(self):
        m = self.monitor()
        self.feed(m)
        s = m.snapshot()
        teacher, guest = s['nodes']['teacher'], s['nodes']['guest']
        self.assertEqual(teacher['pcm_kinds'], ['pipe-write'])
        self.assertEqual(guest['pcm_kinds'], ['ready'])
        self.assertEqual(guest['pcm_s']['n'], 1)
        self.assertAlmostEqual(guest['pcm_s']['median'], .6)
        self.assertAlmostEqual(guest['gap_s']['median'], 2.41)
        self.assertAlmostEqual(guest['tts_s']['median'], .1)
        self.assertAlmostEqual(guest['asr_s']['median'], .9)
        self.assertIsNone(teacher['gap_s'])  # Opening is not a room handoff.
        self.assertIsNone(teacher['asr_s'])  # Unsupported is not zero.
        self.assertAlmostEqual(s['turns'][1]['gap_clock_uncertainty_s'], .02)

    def test_silent_and_pending_turns_do_not_become_zero_latency_samples(self):
        m = self.monitor()
        m.ingest('teacher', status([event(1, 'input_done', 100), event(2, 'synth_done', 101),
                                    event(3, 'input_done', 102)]), 100, 100)
        s = m.snapshot()['nodes']['teacher']
        self.assertEqual(s['observed_turns'], 2)
        self.assertIsNone(s['pcm_s'])

    def test_room_pcm_includes_silent_decision_and_floor_wait(self):
        m = self.monitor()
        m.ingest('guest', status([event(1, 'input_done', 104, heard_at=100, decision_at=103,
                                       identity_wait_s=.01, name_s=2, floor_decision_s=.98),
                                  event(2, 'speech_first_pcm', 104.4)]), 110, 110)
        s = m.snapshot()['nodes']['guest']
        self.assertAlmostEqual(s['pcm_s']['median'], 4.4)
        self.assertEqual(s['decision_s']['median'], 3)
        self.assertEqual(s['floor_wait_s']['median'], 1)
        self.assertEqual(s['identity_wait_s']['median'], .01)
        self.assertEqual(s['name_s']['median'], 2)
        self.assertEqual(s['floor_decision_s']['median'], .98)

    def test_new_session_excludes_old_events_and_stale_asr(self):
        m = self.monitor()
        m.begin(dict(id='new', profiles=dict(teacher='12b')),
                dict(teacher=status(cursor=9)))
        m.ingest('teacher', status([event(8, 'input_done', 80), event(9, 'play', 81, '81\t0\t100\t1000'),
                  event(10, 'reset', 90), event(11, 'asr', 91, elapsed=.5),
                  event(12, 'input_done', 95), event(13, 'play', 96, '96\t0\t100\t1000')]), 100, 100)
        s = m.snapshot()['nodes']['teacher']
        self.assertEqual(s['pcm_s']['n'], 1)
        self.assertEqual(s['pcm_s']['median'], 1)
        self.assertIsNone(s['asr_s'])

    def test_missing_events_prevent_false_first_pcm_and_handoff(self):
        m = self.monitor()
        m.ingest('teacher', status([event(1, 'input_done', 100), event(4, 'play', 105, '105\t0\t100\t1000')]), 100, 100)
        self.assertIsNone(m.snapshot()['nodes']['teacher']['pcm_s'])
        self.assertIn('missing events', m.snapshot()['warnings'][0])
        self.assertFalse(m.gaps_valid)

    def test_input_keeps_its_settled_asr_timing_across_endpoint_hold(self):
        m = self.monitor()
        m.ingest('guest', status([event(1, 'asr', 91, elapsed=.2),
            event(2, 'input_done', 95, asr_elapsed=.2, asr_mode='settle', asr_passes=4),
            event(3, 'speech_first_pcm', 95.4)]), 100, 100)
        s = m.snapshot()
        self.assertAlmostEqual(s['nodes']['guest']['asr_s']['median'], .2)
        self.assertEqual(s['turns'][0]['asr_mode'], 'settle')
        self.assertEqual(s['turns'][0]['asr_passes'], 4)

    def test_node_restart_and_clock_jump_are_explicit(self):
        for change in ('restart', 'clock'):
            m = self.monitor()
            self.feed(m)
            s = status(offset=30 if change == 'clock' else 20)
            if change == 'restart': s['instance_id'] = 'new-node'
            m.ingest('guest', s, 99.99, 100.01)
            self.assertTrue(m.snapshot()['warnings'])
            self.assertIsNone(m.snapshot()['nodes']['guest']['gap_s'])

    def test_partial_capture_restores_and_completed_capture_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            m = self.monitor(root)
            self.feed(m)
            m.checkpoint(force=True)
            restored = TrioLatency(root).snapshot()
            self.assertIn('incomplete', restored['warnings'][0])
            self.assertFalse(restored['active'])
            m.finish('limits')
            restored = TrioLatency(root).snapshot()
            self.assertEqual(restored['reason'], 'limits')
            self.assertEqual(restored['warnings'], [])
            self.assertEqual(restored['nodes']['guest']['pcm_s']['n'], 1)
            saved = json.loads((Path(root)/'latency/session.json').read_text())
            self.assertEqual(saved, restored)
            self.assertNotIn('text', saved['turns'][0])

    def test_save_failure_is_visible_without_stopping_conversation(self):
        with tempfile.TemporaryDirectory() as root:
            m = self.monitor(root)
            self.feed(m)
            with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
                m.finish('manual')
            self.assertFalse(m.active)
            self.assertIn('disk full', m.snapshot()['warnings'][-1])

    def test_out_of_order_batches_wait_for_previous_speech_to_finish(self):
        m = self.monitor()
        teacher, _ = self.feed(m)
        m.nodes['teacher']['turns'][0]['complete'] = False
        self.assertIsNone(m.snapshot()['nodes']['guest']['gap_s'])
        m.ingest('teacher', status(teacher+[event(5, 'synth_done', 102.2)]), 100, 100)
        self.assertAlmostEqual(m.snapshot()['nodes']['guest']['gap_s']['median'], 2.41)


if __name__ == '__main__': unittest.main()

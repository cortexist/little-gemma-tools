"""An uncertain collision holds a complete word and can resume its PCM."""
from pathlib import Path
import sys
import threading
import time
import unittest
import struct
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'script/deviceui'))
from room_hold import WindowsWordOutput, word_ends
from test_room_direct import runtime
from room_direct import DirectRoomRuntime, DuplexOutput


class HoldTests(unittest.TestCase):
    def wait_held(self, output):
        deadline = time.monotonic()+1
        while not output.held and time.monotonic() < deadline: time.sleep(.001)
        self.assertTrue(output.held)

    def test_grace_skips_early_boundary_and_repeated_requests_do_not_extend_it(self):
        for rate in (16000, 22050, 48000):
            with self.subTest(rate=rate):
                written = []
                device = Mock(latency=.1)
                device.write.side_effect = lambda data: written.append(bytes(data)) or False
                output = WindowsWordOutput(device, threading.Event(), rate=rate, grace_ms=80)
                marks = [round(rate*t) for t in (.04, .10, .20)]
                output.add_boundaries(marks)
                output.hold()
                first = round(rate*.06)
                pcm = b''.join(struct.pack('<hh', n%30000, -n%30000)
                               for n in range(round(rate*.24)))
                output.write(pcm[:first*4])
                output.add_boundaries(marks)
                output.hold()
                thread = threading.Thread(target=output.write, args=(pcm[first*4:],))
                thread.start()
                try:
                    self.wait_held(output)
                    self.assertEqual(output.held_samples, marks[1])
                    self.assertEqual(b''.join(written), pcm[:marks[1]*4])
                    output.resume()
                    thread.join(1)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(b''.join(written), pcm)
                finally:
                    output.resume(); thread.join(1)

    def test_cancel_pending_hold_keeps_exact_pcm_and_next_hold_gets_new_anchor(self):
        device = Mock(latency=.1)
        written = []
        device.write.side_effect = lambda data: written.append(bytes(data)) or False
        output = WindowsWordOutput(device, threading.Event(), rate=1000, grace_ms=80)
        output.hold()  # Before alignment: do not defer starting the grace.
        pcm = bytes(range(240))*4
        output.write(pcm[:60*4])
        output.add_boundaries([40, 100, 200])
        output.hold()
        self.assertEqual(output.hold_not_before, 80)
        output.resume()
        output.write(pcm[60*4:120*4])
        self.assertFalse(output.held)
        self.assertEqual(b''.join(written), pcm[:120*4])
        output.hold()
        self.assertEqual(output.hold_not_before, 200)
        output.resume()
        output.write(pcm[120*4:])
        self.assertEqual(b''.join(written), pcm)

    def test_request_during_in_flight_write_anchors_after_submitted_pcm(self):
        device = Mock(latency=.1)
        output = WindowsWordOutput(device, threading.Event(), rate=1000, grace_ms=80)
        device.write.side_effect = lambda data: output.hold() or False
        output.write(bytes(20*4))
        self.assertEqual(output.hold_not_before, 100)

    def test_confirmed_yield_releases_writer_even_with_pending_or_actual_hold(self):
        for actual_hold in (False, True):
            with self.subTest(actual_hold=actual_hold):
                yielded = threading.Event()
                device = Mock(latency=0)
                written = []
                device.write.side_effect = lambda data: written.append(bytes(data)) or False
                output = WindowsWordOutput(device, yielded, rate=1000, grace_ms=80)
                output.add_boundaries([100]); output.hold()
                if actual_hold:
                    thread = threading.Thread(target=output.write, args=(bytes(200*4),))
                    thread.start()
                    try:
                        self.wait_held(output)
                        yielded.set(); thread.join(1)
                        self.assertFalse(thread.is_alive())
                        self.assertEqual(sum(map(len, written)), 400)
                    finally:
                        yielded.set(); output.resume(); thread.join(1)
                else:
                    yielded.set(); output.write(bytes(200*4))
                    self.assertEqual(written, [])

    def test_linux_grace_wire_includes_empty_alignment_and_resume(self):
        output = DuplexOutput.__new__(DuplexOutput)
        output.runtime = Mock(hold_grace_ms=80)
        output.rate = 22050
        output.lock = threading.Lock()
        output.boundaries = []
        output.hold_requested = False
        output.frame = Mock()
        output.hold()
        output.frame.assert_called_with(b'G', struct.pack('<I', 1280))
        output.add_boundaries([2205, 4410])
        output.frame.assert_called_with(b'G', struct.pack('<IQQ', 1280, 1600, 3200))
        output.resume()
        output.frame.assert_called_with(b'R')
        output.runtime.hold_grace_ms = 0
        output.hold()
        output.frame.assert_called_with(b'H', struct.pack('<QQ', 1600, 3200))

    def test_grace_config_is_bounded_numeric_and_defaults_to_zero(self):
        r = runtime()
        self.assertEqual(r.hold_grace_ms, 0)
        for value in (True, '80', None, -1, 2001, float('nan'), float('inf')):
            r.device.cfg = dict(turn_head=dict(barge_hold=dict(grace_ms=value)))
            with self.subTest(value=value), self.assertRaises(ValueError):
                DirectRoomRuntime(r.device, dict(session='session', topic='agents'))
        r.device.cfg = dict(turn_head=dict(barge_hold=dict(grace_ms=80)))
        self.assertEqual(DirectRoomRuntime(r.device, dict(session='session', topic='agents')).hold_grace_ms, 80)

    def test_piper_space_marks_end_of_word_and_clause_end_is_fallback(self):
        schedule = '0.0\t100.0\th\n100.0\t40.0\t \n140.0\t200.0\tw\n'
        self.assertEqual(word_ends(schedule, 1000, 16000), [2600, 6440])

    def test_low_score_holds_once_and_false_alarm_resumes(self):
        r = runtime()
        r.hold_threshold = .7
        r.current = dict(id='1', control_action='speak', speak_score=.62)
        r.live = Mock(segment=1)
        r.playing = True
        r.active_stream = Mock()
        r.request_early_hold()
        r.request_early_hold()
        r.active_stream.hold.assert_called_once()
        r.release_early_hold('quiet_without_identification')
        r.active_stream.resume.assert_called_once()
        r.live.segment = 2
        r.request_early_hold()
        self.assertEqual(r.active_stream.hold.call_count, 2)
        r.current['speak_score'] = .85
        r.release_early_hold('quiet_without_identification')
        r.live.segment = 3
        r.request_early_hold()
        self.assertEqual(r.active_stream.hold.call_count, 2)

    def test_windows_pauses_at_boundary_and_resumes_remaining_pcm(self):
        yielded = threading.Event()
        written = []
        device = Mock(latency=0)
        device.write.side_effect = lambda data: written.append(bytes(data))
        output = WindowsWordOutput(device, yielded)
        output.add_boundaries([4, 12])
        output.hold()
        pcm = bytes(12*4)
        thread = threading.Thread(target=output.write, args=(pcm,))
        thread.start()
        try:
            deadline = time.monotonic()+1
            while sum(map(len, written)) < 16 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(sum(map(len, written)), 16)
            self.assertTrue(thread.is_alive())
            output.resume()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(sum(map(len, written)), len(pcm))
        finally:
            output.resume()
            thread.join(1)


if __name__ == '__main__': unittest.main()

"""Receiver clock boundaries survive fragments and queued/coalesced transport."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui')]
from decision_timing import TimedHiddenReplyDecoder, decision_stamp, compare_decision_timing
from test_turn_head import frame


class TimingTests(unittest.TestCase):
    def test_frame_and_tag_share_read_without_inventing_token_timestamps(self):
        snapshots = []; decoder = TimedHiddenReplyDecoder(snapshots.append)
        raw = frame()
        with patch('decision_timing.time.monotonic', return_value=10.):
            self.assertEqual(decoder.feed(raw[:-1]), [])
        with patch('decision_timing.time.monotonic', return_value=11.):
            pieces = decoder.feed(raw[-1:]+b'[speak]Hello.<turn|>')
        decoder.finish()
        s = snapshots[0]
        self.assertEqual(s['transport_received_monotonic'], 11.)
        with patch('decision_timing.time.monotonic', return_value=12.):
            text = decision_stamp('speak', 'explicit_tag', 0, pieces[0])
        self.assertEqual(text['received_monotonic'], 11.)
        self.assertEqual(text['recognized_monotonic'], 12.)
        h = dict(snapshot=s, action='defer', ready_monotonic=11.001, ready_at=text['received_at'])
        result = compare_decision_timing(h, text, 9.)
        self.assertEqual(result['text_after_frame_s'], 0.)
        self.assertTrue(result['same_transport_read'])
        self.assertLess(result['text_minus_head_s'], 0.)
        self.assertEqual(result['parser_after_receipt_s'], 1.)
        self.assertFalse(result['controlling'])

    def test_plain_bytes_are_labeled_as_parser_fallback(self):
        text = decision_stamp('defer', 'explicit_tag', 0, b'[defer]')
        self.assertEqual(text['receipt_source'], 'parser_fallback')
        self.assertIsNone(text['read_serial'])


if __name__ == '__main__': unittest.main()

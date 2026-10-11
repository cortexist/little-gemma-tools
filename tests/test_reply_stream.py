import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script'),
               str(Path(__file__).resolve().parents[1]/'script/deviceui')]
from reply_stream import ReplyStream


class ReplyStreamTests(unittest.TestCase):
    def test_fragmented_facial_cues_follow_clauses_without_entering_speech(self):
        raw = ('<|channel>thought[angry]private<channel|>[speak]'
               '<speaker_name>Chey</speaker_name>[happy][nod]Hello. '
               '[sad][shake]No.<turn|>').encode()
        for size in (1, 2, 7, 10000):
            p = ReplyStream(room=True, decision_tags=True)
            clauses = []
            for i in range(0, len(raw), size): clauses.extend(p.feed(raw[i:i+size]))
            self.assertEqual(clauses, ['Hello.', 'No.'])
            self.assertEqual(list(p.clause_visuals), [
                [('expression', 'happy'), ('gesture', 'nod')],
                [('expression', 'sad'), ('gesture', 'shake')]])
            self.assertEqual(p.metadata, ['Chey'])
            self.assertNotIn('[', p.text)

    def test_visual_only_or_trailing_tags_have_no_spoken_clause(self):
        p = ReplyStream(room=True, decision_tags=True)
        self.assertEqual(p.feed(b'[speak][happy][nod]<turn|>'), [])
        self.assertEqual(list(p.clause_visuals), [])
        for raw in (b'[happy][speak]Hello.', b'[defer][happy]<turn|>'):
            with self.assertRaises(ValueError):
                ReplyStream(room=True, decision_tags=True).feed(raw)

    def test_tag_first_measurement_never_speaks_tags_and_preserves_names(self):
        for action in ('speak', 'defer'):
            raw = ('<|channel>thought\n<channel|>['+action+']'
                   '<speaker_name>Chéy</speaker_name>'+('Hello, Chéy.' if action == 'speak' else '')+'<turn|>').encode()
            for split in range(len(raw)+1):
                observed = []
                p = ReplyStream(room=True, decision_tags=True, on_decision=lambda *v: observed.append(v))
                clauses = p.feed(raw[:split])+p.feed(raw[split:])
                self.assertEqual(observed, [(action, 'explicit_tag', 0)])
                self.assertEqual(p.metadata, ['Chéy'])
                self.assertEqual(clauses, ['Hello,', 'Chéy.'] if action == 'speak' else [])

    def test_callback_marks_closing_bracket_before_rest_of_read_is_parsed(self):
        observed = []
        p = ReplyStream(room=True, decision_tags=True,
            on_decision=lambda *v: observed.append((v, p.text, list(p.metadata), p.done)))
        p.feed(b'[spe'); self.assertEqual(observed, [])
        p.feed(b'ak'); self.assertEqual(observed, [])
        p.feed(b']<speaker_name>Chey</speaker_name>Hello.<turn|>')
        self.assertEqual(observed, [(('speak', 'explicit_tag', 0), '', [], False)])

    def test_missing_truncated_and_name_first_tags_are_distinguishable(self):
        for raw, decision, source, before in [
            (b'Hello.<turn|>', 'speak', 'implicit_speech', 0),
            (b'[spe<turn|>', None, None, None),
            (b'<speaker_name>Chey</speaker_name>[defer]<turn|>', 'defer', 'explicit_tag', 1)]:
            seen = []; p = ReplyStream(room=True, decision_tags=True, on_decision=lambda *v: seen.append(v))
            p.feed(raw)
            self.assertEqual((p.decision, p.decision_source), (decision, source))
            self.assertEqual(seen, [] if decision is None else [(decision, source, before)])

    def test_duplicate_contradictory_and_late_tags_fail_closed(self):
        for raw in (b'[speak][defer]', b'[defer][speak]', b'[speak][speak]',
                    b'Hello. [speak]', b'[defer]Hello.'):
            with self.assertRaises(ValueError):
                ReplyStream(room=True, decision_tags=True).feed(raw)

    def test_speak_control_is_opt_in(self):
        p = ReplyStream(room=True)
        self.assertEqual(p.feed(b'[speak]Hello.<turn|>'), ['[speak]Hello.'])
        with self.assertRaises(ValueError): ReplyStream(decision_tags=True)

    def test_every_byte_boundary_hides_control_and_preserves_unicode_and_titles(self):
        raw = ('<|channel>thought\nprivate deliberation<channel|>[happy]Hello Mrs. Byers, '
               'I’m here. <|tool_call>do_not_speak{}<tool_call|>Ask me!<turn|>\n').encode()
        for size in (1, 2, 7, 10000):
            parser = ReplyStream()
            clauses = []
            for i in range(0, len(raw), size):
                clauses.extend(parser.feed(raw[i:i+size]))
            self.assertEqual(clauses, ['Hello Mrs. Byers,', 'I’m here.', 'Ask me!'])
            self.assertTrue(parser.done)
            self.assertNotIn('private', parser.text)
            self.assertNotIn('do_not_speak', parser.text)

    def test_first_clause_streams_before_turn_end_and_markdown_is_not_spoken(self):
        parser = ReplyStream()
        self.assertEqual(parser.feed(b'**Hello.** More'), ['Hello.'])
        self.assertFalse(parser.done)
        self.assertEqual(parser.feed(b' words<turn|>'), ['More words'])
        self.assertTrue(parser.done)

    def test_unclosed_hidden_channel_never_leaks(self):
        parser = ReplyStream()
        self.assertEqual(parser.feed(b'<|channel>thought secret<turn|>'), [])
        self.assertTrue(parser.done)

    def test_engine_eos_also_closes_a_silent_or_spoken_reply(self):
        for text in ('NONE', '{"action":"defer","reason":"defer"}', 'Hello.'):
            parser = ReplyStream()
            for byte in (text+'<eos>').encode(): parser.feed(bytes([byte]))
            self.assertTrue(parser.done)
            self.assertEqual(parser.text, text)

    def test_room_defer_and_name_metadata_never_reach_speech_at_any_boundary(self):
        for body in ('[defer]', 'Hello Chey.'):
            raw=(body+'<speaker_name>{"name":"Chey"}</speaker_name><turn|>').encode()
            for size in (1,2,7,10000):
                parser=ReplyStream(room=True);clauses=[]
                for i in range(0,len(raw),size):clauses.extend(parser.feed(raw[i:i+size]))
                self.assertEqual(clauses,[] if body=='[defer]' else ['Hello Chey.'])
                self.assertEqual(parser.deferred,body=='[defer]')
                self.assertEqual(parser.metadata,['{"name":"Chey"}'])
                self.assertTrue(parser.done)

    def test_private_name_before_defer_or_speech_is_never_a_clause(self):
        for body in ('[defer]', 'Hello there.'):
            raw=('<speaker_name>Agent Smith</speaker_name>'+body+'<turn|>').encode()
            for size in (1,2,7,10000):
                parser=ReplyStream(room=True); clauses=[]
                for offset in range(0,len(raw),size): clauses.extend(parser.feed(raw[offset:offset+size]))
                self.assertEqual(parser.metadata, ['Agent Smith'])
                self.assertEqual(clauses, [] if body=='[defer]' else ['Hello there.'])

    def test_room_defer_cannot_be_combined_with_speech(self):
        for raw in (b'[defer]Hello.<turn|>',b'Hello. [defer]<turn|>'):
            with self.assertRaises(ValueError):ReplyStream(room=True).feed(raw)

    def test_room_truncated_name_metadata_is_not_spoken(self):
        parser=ReplyStream(room=True)
        self.assertEqual(parser.feed(b'[defer]<speaker_name>{"name":"secret"'),[])
        with self.assertRaises(ValueError):parser.feed(b'<turn|>')
        self.assertNotIn('secret',parser.text)


if __name__ == '__main__': unittest.main()

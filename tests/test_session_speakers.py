"""Listener-local learning, uncertainty and reset; no model/microphone required."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'script'), str(ROOT/'script/deviceui')]
from session_speakers import SpeakerSession, proposals, unit
from speech_names import piper_names, whisper_name_hint
from session_scenario import ClassroomScenario


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.s = SpeakerSession([dict(name='Mrs. Byers', embedding=[1, 0, 0], trusted_introducer=True)])

    def hear(self, turn, vector, text, **kw):
        return self.s.observe(turn, vector, text, clean=kw.get('clean', True), duration=kw.get('duration', 2))

    def tag(self, name, turn, *, kind='self', track='voice-2', target=None):
        return dict(session=self.s.session_id, kind=kind, name=name, track=track,
                    turn=turn, target_turn=turn if kind == 'self' else target)

    def test_teacher_names_previously_unknown_voice(self):
        self.assertNotIn('Agent Smith', self.s.prompt())
        heard = self.hear('guest', [0, 1, 0], 'Good afternoon, everyone.')
        self.assertEqual(heard['track_id'], 'voice-2')
        self.assertIsNone(self.s.tracks['voice-2']['name'])
        self.hear('teacher', [1, 0, 0], 'The person who just spoke is our guest, Agent Smith.')
        output = '<speaker_name>'+json.dumps(self.tag('Agent Smith', 'teacher', kind='introduction', target='guest'))+'</speaker_name>'
        result = self.s.accept(output)
        self.assertEqual(result['accepted'][0]['status'], 'bound')
        self.assertEqual(result['spoken'], '')
        self.assertEqual(self.s.tracks['voice-2']['name'], 'Agent Smith')
        self.assertEqual(self.hear('guest2', [0, 1, 0], 'What would you like to ask?')['track_id'], 'voice-2')

    def test_introduction_before_voice_stays_pending_until_evidence(self):
        self.hear('teacher', [1, 0, 0], 'Let me introduce our guest, Agent Smith.')
        event = self.tag('Agent Smith', 'teacher', kind='introduction', track=None)
        self.assertEqual(self.s.apply(event)['status'], 'pending_voice')
        self.hear('guest', [0, 1, 0], 'Good afternoon, everyone.')
        self.assertIsNone(self.s.tracks['voice-2']['name'])
        self.hear('guest2', [0, 1, 0], 'My name is Agent Smith.')
        self.s.apply(self.tag('Agent Smith', 'guest2'))
        self.assertFalse(self.s.pending_names)

    def test_recent_similar_introduction_does_not_bind_asr_variant(self):
        self.hear('teacher', [1, 0, 0], 'I have invited Agent Smith.')
        self.s.apply(self.tag('Agent Smith', 'teacher', kind='introduction', track=None))
        self.hear('guest', [0, 1, 0], 'Hello, I am Major Smith.')
        with self.assertRaisesRegex(ValueError, 'clarify before binding'):
            self.s.apply(self.tag('Major Smith', 'guest'))
        self.assertIsNone(self.s.tracks['voice-2']['name'])
        self.assertIn('Agent Smith', self.s.pending_names)

        # An unrelated person can still give their own name after the teacher.
        self.s.reset()
        self.hear('teacher', [1, 0, 0], 'I have invited Agent Smith.')
        self.s.apply(self.tag('Agent Smith', 'teacher', kind='introduction', track=None))
        self.hear('student', [0, 1, 0], 'My name is Chey.')
        self.assertEqual(self.s.apply(self.tag('Chey', 'student'))['status'], 'bound')

    def test_explicit_homophone_name_claim_preserves_raw_asr(self):
        self.hear('student', [0, 1, 0], 'My name is Shy. I have a question.')
        self.assertEqual(self.s.apply(self.tag('Shy', 'student'))['name'], 'Chey')
        self.assertEqual(self.s.turns['student']['raw_transcript'], 'My name is Shy. I have a question.')

    def test_measured_asr_spelling_and_biased_adjective(self):
        self.hear('student', [0, 1, 0], 'My name is Shai I have a question about language models.')
        self.assertEqual(self.s.apply(self.tag('Shai', 'student'))['name'], 'Chey')
        self.s.reset()
        self.hear('student', [0, 1, 0], 'I am Chey, speaking in front of the whole class makes me nervous.')
        with self.assertRaises(ValueError):
            self.s.apply(self.tag('Chey', 'student'))

    def test_adjective_quoted_negative_and_question_do_not_learn(self):
        for text in ['I am shy.', 'I did not say my name is Shy.', 'She said "My name is Shy."', 'My name is Shy?']:
            with self.subTest(text=text):
                self.setUp()
                self.hear('student', [0, 1, 0], text)
                with self.assertRaises(ValueError):
                    self.s.apply(self.tag('Shy', 'student'))

    def test_mention_cannot_attach_pending_name_to_next_voice(self):
        self.hear('guest', [0, 1, 0], 'Good afternoon.')
        self.hear('teacher', [1, 0, 0], 'Agent Smith can answer questions.')
        with self.assertRaises(ValueError):
            self.s.apply(self.tag('Agent Smith', 'teacher', kind='introduction', target='guest'))
        self.assertIsNone(self.s.tracks['voice-2']['name'])

    def test_teacher_question_negative_or_quotation_cannot_name_a_voice(self):
        for text in ['Was the person who just spoke Agent Smith?',
                     'The person who just spoke is not Agent Smith.',
                     'She said "The person who just spoke is Agent Smith."']:
            with self.subTest(text=text):
                self.s.reset()
                self.hear('guest', [0, 1, 0], 'Good afternoon.')
                self.hear('teacher', [1, 0, 0], text)
                with self.assertRaises(ValueError):
                    self.s.apply(self.tag('Agent Smith', 'teacher', kind='introduction', target='guest'))
                self.assertIsNone(self.s.tracks['voice-2']['name'])

    def test_reset_forgets_learned_voice_and_rejects_delayed_tag(self):
        self.hear('guest', [0, 1, 0], 'My name is Agent Smith.')
        event = self.tag('Agent Smith', 'guest')
        self.s.apply(event)
        previous = self.s.session_id
        self.s.reset()
        self.assertNotEqual(previous, self.s.session_id)
        self.assertEqual(len(self.s.tracks), 1)
        self.assertEqual(self.s.tracks['voice-1']['name'], 'Mrs. Byers')
        self.assertFalse(self.s.turns or self.s.pending_names or self.s.audit)
        with self.assertRaises(ValueError):
            self.s.apply(event)

    def test_uncertainty_overlap_short_turns_never_gain_names(self):
        for kw in [dict(clean=False), dict(duration=.2)]:
            self.s.reset()
            turn = self.hear('guest', [0, 1, 0], 'My name is Agent Smith.', **kw)
            self.assertIsNone(turn['track_id'])
            with self.assertRaises(ValueError):
                self.s.apply(self.tag('Agent Smith', 'guest'))
        self.s.reset()
        self.hear('guest', [0, 1, 0], 'Hello there.')
        self.assertEqual(self.hear('mixed', [1, 1, 0], 'My name is Agent Smith.')['status'], 'uncertain')

    def test_wrong_target_conflicting_name_and_stale_turn_are_rejected(self):
        self.hear('guest', [0, 1, 0], 'My name is Agent Smith.')
        with self.assertRaises(ValueError):
            self.s.apply(self.tag('Agent Smith', 'guest', track='voice-1'))
        self.s.apply(self.tag('Agent Smith', 'guest'))
        self.hear('renaming', [0, 1, 0], 'My name is Chey.')
        with self.assertRaises(ValueError):
            self.s.apply(self.tag('Chey', 'renaming'))
        with self.assertRaises(ValueError):
            self.s.apply(self.tag('Agent Smith', 'guest'))

    def test_prior_embedding_does_not_drift_after_self_introduction(self):
        self.hear('teacher', [1, 0, 0], 'My name is Mrs. Byers.')
        self.s.apply(self.tag('Mrs. Byers', 'teacher', track='voice-1'))
        self.hear('teacher2', [1, .1, 0], 'Hello again.')
        self.assertEqual(self.s.tracks['voice-1']['embedding'], (1., 0., 0.))

    def test_duplicate_turn_and_invalid_embeddings_fail(self):
        self.hear('guest', [0, 1, 0], 'Hello there.')
        with self.assertRaises(ValueError):
            self.hear('guest', [0, 1, 0], 'Another transcript.')
        for vector in [[], [0, 0], [float('nan'), 1], [float('inf'), 1], [1e308, 1e308]]:
            with self.assertRaises(ValueError):
                unit(vector)
        with self.assertRaises(ValueError):
            self.hear('bad_dimension', [0, 1], 'Hello there.')

    def test_metadata_never_becomes_spoken_text(self):
        for text in ['Hello.<speaker_name>{bad}</speaker_name>', 'Hello.<speaker_name>{"session":',
                     'Hello.<speaker_name>{"name":"A","name":"B"}</speaker_name>']:
            spoken, events, errors = proposals(text)
            self.assertEqual(spoken, 'Hello.')
            self.assertFalse(events)
            self.assertTrue(errors)

    def test_extraction_channel_rejects_unframed_output_and_silences_none(self):
        self.hear('student', [0, 1, 0], 'My name is Chey.')
        bare = json.dumps(self.tag('Chey', 'student'))
        result = self.s.accept(bare, metadata_only=True)
        self.assertTrue(result['rejected'])
        self.assertEqual(result['spoken'], '')
        self.assertIsNone(self.s.tracks['voice-2']['name'])
        self.assertEqual(self.s.accept('NONE', metadata_only=True), dict(spoken='', accepted=[], rejected=[]))

    def test_extraction_channel_rejects_multiple_tags(self):
        self.hear('student', [0, 1, 0], 'My name is Chey.')
        tag = '<speaker_name>'+json.dumps(self.tag('Chey', 'student'))+'</speaker_name>'
        result = self.s.accept(tag+tag, metadata_only=True)
        self.assertTrue(result['rejected'])
        self.assertFalse(result['accepted'])
        self.assertIsNone(self.s.tracks['voice-2']['name'])


class SpeechNamesTests(unittest.TestCase):
    def test_pronunciation_boundary(self):
        self.assertEqual(piper_names('Chey, is Cheyenne shy?'), '[[ʃˈaɪ]], is Cheyenne shy?')
        self.assertEqual(piper_names('[[Chey]] <|tool_call>Chey<tool_call|> CHEY'),
                         '[[Chey]] <|tool_call>Chey<tool_call|> [[ʃˈaɪ]]')
        self.assertEqual(piper_names('Chey', {}), 'Chey')

    def test_hint_is_only_supplied_known_vocabulary(self):
        self.assertEqual(whisper_name_hint([]), '')
        hint = whisper_name_hint(['Mrs. Byers', 'Mrs. Byers'])
        self.assertEqual(hint, 'Names mentioned in this conversation: Mrs. Byers.')
        self.assertNotIn('Chey', hint)


class ScenarioTests(unittest.TestCase):
    def test_acquaintances_are_asymmetric_and_do_not_leak_into_prompts(self):
        scenario = ClassroomScenario()
        self.assertEqual({speaker: policy['node'] for speaker, policy in scenario.participants.items()},
                         {'catherine-byers': 'cortex', 'chey': 'cortex1', 'agent-smith': 'nowhere'})
        self.assertNotIn('Agent Smith', scenario.prompt('chey'))
        self.assertNotIn('Chey', scenario.prompt('agent-smith'))
        self.assertIn('Chey', scenario.prompt('catherine-byers'))
        self.assertIn('Agent Smith', scenario.prompt('catherine-byers'))
        loaded = []
        def load(peer):
            loaded.append(peer)
            return [1, 0]
        priors = scenario.priors('agent-smith', load)
        self.assertEqual(loaded, ['catherine-byers'])
        self.assertEqual([p['name'] for p in priors], ['Mrs. Byers'])
        self.assertTrue(priors[0]['trusted_introducer'])


if __name__ == '__main__':
    unittest.main()

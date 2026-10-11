"""Peer tracking and dashboard failure isolation, without models or microphones."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'script/deviceui'))
from speaker_collections import SpeakerCollection, load_collection, identity_prompt
import copy

COLLECTION = json.loads((ROOT/'script/deviceui/speakers.json').read_text())['collections']['voice-classroom']

path = ROOT/'script/deviceui/speaker_identity.py'
spec = importlib.util.spec_from_file_location('speaker_identity', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.collection = SpeakerCollection('voice-classroom', copy.deepcopy(COLLECTION))
        self.names = {key: {} for key in self.collection.by_voice}
        self.t = mod.PeerIdentity(self.collection, ['catherine-byers', 'linda-johnson'])

    def confirm(self, voice, at=10):
        scores = {name: .8 if name == voice else .1 for name in self.names}
        self.t.observe(scores, at)
        return self.t.observe(scores, at+.25)

    def test_swap_requires_new_evidence_and_preserves_history_as_history(self):
        state = self.confirm('lessac')
        self.assertEqual(state['display_name'], 'Catherine Byers')
        self.assertFalse(state['allow_named_address'])
        self.t.reset('quiet')
        self.assertIsNone(self.t.snapshot()['voice_id'])
        self.assertEqual(self.t.snapshot()['last_peer']['voice_id'], 'lessac')
        self.t.reset()
        state = self.t.observe({'lessac': .1, 'ljspeech': .8, 'ozgirl': .1}, 12)
        self.assertEqual(state['status'], 'identifying')
        self.assertIsNone(state['voice_id'])
        state = self.t.observe({'lessac': .1, 'ljspeech': .8, 'ozgirl': .1}, 12.25)
        self.assertEqual(state['last_peer']['address_name'], 'Mrs. Johnson')

    def test_unknown_conflict_self_voice_and_gap_cannot_inherit_identity(self):
        self.confirm('lessac')
        for scores in [{'lessac': .2, 'ljspeech': .1, 'ozgirl': .1},
                       {'lessac': .8, 'ljspeech': .79, 'ozgirl': .1},
                       {'lessac': .1, 'ljspeech': .1, 'ozgirl': .9},
                       {'lessac': float('nan'), 'ljspeech': .1, 'ozgirl': .1}]:
            self.assertIsNone(self.t.observe(scores, 13)['voice_id'])
        self.confirm('lessac', 14)
        self.t.reset('audio_gap')
        self.assertIsNone(self.t.snapshot()['voice_id'])
        self.assertEqual(self.t.snapshot()['status'], 'audio_gap')

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/'script/deviceui/server.py'
        spec = importlib.util.spec_from_file_location('identity_dashboard', path)
        cls.ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ui)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = dict(name='test', runtime_dir=self.tmp.name, sink='sink', source='source',
                   profiles={}, whisper_model='asr', voice='tts',
                   speaker_collection_id='voice-classroom',
                   speaker_identity={'enabled': True, 'peer_speaker_ids': ['catherine-byers']})
        with patch.object(self.ui.Device, 'audio_poll'):
            self.d = self.ui.Device(cfg)
        self.d.state.update(listening=True, ready=True, phase='listening')

    def tearDown(self):
        self.tmp.cleanup()

    def event(self, at):
        return json.dumps(dict(status='identified', collection_id='voice-classroom', speaker_id='catherine-byers', voice_id='lessac', display_name='untrusted name',
                               last_peer=dict(collection_id='voice-classroom', speaker_id='catherine-byers', voice_id='lessac', display_name='untrusted name', at=at)))

    def test_local_playback_and_disabled_listening_do_not_establish_peer(self):
        self.d.state['speaker_suppress_until'] = 20
        self.d.receive_identity(self.event(10), 10)
        self.assertIsNone(self.d.state['speaker_identity']['last_peer'])
        self.d.receive_identity(self.event(10), 21)  # stale worker memory after playback
        self.assertIsNone(self.d.state['speaker_identity']['last_peer'])
        self.assertIsNone(self.d.state['speaker_identity']['voice_id'])
        self.d.receive_identity(self.event(22), 22)
        self.assertEqual(self.d.state['speaker_identity']['last_peer']['at'], 22)
        self.d.state['listening'] = False
        self.d.receive_identity(self.event(23), 23)
        self.assertIsNone(self.d.state['speaker_identity']['voice_id'])
        self.assertEqual(self.d.state['speaker_identity']['last_peer']['at'], 22)

    def test_resuming_listening_requires_post_resume_identity_evidence(self):
        self.d.state['listening'] = False
        class Events:
            def __init__(self): self.sent = False
            def recv(self, size):
                if self.sent: raise OSError('done')
                self.sent = True
                return json.dumps(dict(kind='listening', text='1', at=22)).encode()
        self.d.telemetry = Events()
        self.d.receive()
        self.d.receive_identity(self.event(21), 23)
        self.assertIsNone(self.d.state['speaker_identity']['voice_id'])
        self.assertIsNone(self.d.state['speaker_identity']['last_peer'])
        self.d.receive_identity(self.event(24), 24)
        self.assertEqual(self.d.state['speaker_identity']['voice_id'], 'lessac')

    def test_metadata_is_canonical_and_cross_collection_or_voice_mismatch_is_rejected(self):
        self.d.receive_identity(self.event(22), 22)
        state = self.d.state['speaker_identity']
        self.assertEqual(state['display_name'], 'Catherine Byers')
        self.assertEqual(state['face_color'], '#e2ebd7')
        self.assertEqual(state['speaker_id'], 'catherine-byers')
        for key, wrong in [('collection_id', 'different-room'), ('voice_id', 'ljspeech'), ('speaker_id', 'linda-johnson')]:
            incoming = json.loads(self.event(23))
            incoming[key] = wrong
            incoming['last_peer'][key] = wrong
            self.d.receive_identity(json.dumps(incoming), 23)
            self.assertIsNone(self.d.state['speaker_identity']['speaker_id'])
            self.assertEqual(self.d.state['speaker_identity']['last_peer']['at'], 22)

    def test_start_passes_configured_identity_into_engine_prompt(self):
        self.d.cfg['profiles']['test'] = {'model': '/model'}
        self.d.speaker = self.d.collection.speaker('chey')
        with patch.object(self.ui.threading, 'Thread') as thread:
            self.d.start(dict(profile='test', system_prompt='Student instructions.',
                              peer_collection_id='voice-classroom', peer_speaker_id='linda-johnson'))
            prompt = thread.call_args.kwargs['args'][1]
            self.assertIn('Chey', prompt)
            self.assertIn('Mrs. Johnson', prompt)
            self.assertNotIn('Mrs. Byers', prompt)
        with self.assertRaises(ValueError):
            self.d.start(dict(profile='test', system_prompt='Student instructions.',
                              peer_collection_id='wrong-room', peer_speaker_id='linda-johnson'))

    def test_final_drain_keeps_identity_live_without_reopening_model_input(self):
        with patch.object(self.d, 'command') as command:
            self.d.listen(False, finishing=True)
            command.assert_called_once_with('listen 0')
        self.d.state.update(listening=False, phase='finishing')
        self.d.receive_identity(self.event(22), 22)
        with patch.object(self.ui.time, 'time', return_value=22.5):
            state = self.d.snapshot()['state']
        self.assertFalse(state['listening'])
        self.assertEqual(state['phase'], 'finishing')
        self.assertEqual(state['speaker_identity']['speaker_id'], 'catherine-byers')
        self.assertEqual(state['speaker_identity']['status'], 'identified')
        with self.assertRaisesRegex(ValueError, 'busy'):
            self.d.prompt('Do not start another turn.')
        # Normal pause revokes the final-drain exception.
        with patch.object(self.d, 'command'):
            self.d.listen(False)
        self.d.receive_identity(self.event(23), 23)
        self.assertEqual(self.d.state['speaker_identity']['status'], 'paused')

    def test_final_drain_preserves_playback_and_heartbeat_guards(self):
        self.d.state.update(finishing=True, listening=False, speaker_suppress_until=20)
        self.d.receive_identity(self.event(19), 19)
        self.assertIsNone(self.d.state['speaker_identity']['speaker_id'])
        self.d.receive_identity(self.event(19), 21)
        self.assertIsNone(self.d.state['speaker_identity']['speaker_id'])
        self.d.receive_identity(self.event(22), 22)
        self.assertEqual(self.d.state['speaker_identity']['speaker_id'], 'catherine-byers')
        with patch.object(self.ui.time, 'time', return_value=26):
            self.assertEqual(self.d.snapshot()['state']['speaker_identity']['status'], 'unavailable')

    def test_failed_finish_control_does_not_leave_tracking_enabled(self):
        with patch.object(self.d, 'command', side_effect=ValueError('not ready')):
            with self.assertRaises(ValueError): self.d.listen(False, finishing=True)
        self.assertFalse(self.d.state['finishing'])

    def test_panel_identity_and_host_are_separate(self):
        cfg = dict(self.d.cfg, name='Old host label', voice='en_US-lessac-medium.onnx',
                   platform='RTX PRO 4500 Blackwell')
        with patch.object(self.ui.Device, 'audio_poll'), patch.object(self.ui.socket, 'gethostname', return_value='nowhere.local'):
            d = self.ui.Device(cfg)
            state = d.snapshot()
        self.assertEqual(state['state']['name'], 'Catherine Byers')
        self.assertEqual(state['host'], {'hostname': 'nowhere', 'platform': 'RTX PRO 4500 Blackwell'})

    def test_optional_worker_death_does_not_stop_voice_pipeline(self):
        class Exited:
            pid = 100
            def poll(self): return 1
        self.d.procs['speaker-id'] = Exited()
        self.d.receive_identity(self.event(22), 22)
        with patch.object(self.ui.time, 'time', return_value=26):
            state = self.d.snapshot()['state']
        self.assertTrue(state['ready'])
        self.assertEqual(state['speaker_identity']['status'], 'unavailable')
        self.assertIsNone(state['speaker_identity']['voice_id'])

class CollectionTests(unittest.TestCase):
    def test_duplicate_names_allowed_but_voice_ids_and_models_are_unique(self):
        data = copy.deepcopy(COLLECTION)
        data['speakers']['linda-johnson']['display_name'] = 'Catherine Byers'
        collection = SpeakerCollection('room-b', data)
        self.assertNotEqual(collection.by_voice['lessac'], collection.by_voice['ljspeech'])
        for key in ('voice_id', 'voice'):
            invalid = copy.deepcopy(data)
            invalid['speakers']['linda-johnson'][key] = invalid['speakers']['catherine-byers'][key]
            with self.assertRaises(ValueError): SpeakerCollection('room-b', invalid)
        # The same voice is permitted in another collection; IDs are scoped.
        self.assertNotEqual(collection.speaker('chey')['collection_id'],
                            SpeakerCollection('room-a', data).speaker('chey')['collection_id'])

    def test_renaming_does_not_change_identity_and_metadata_is_extensible(self):
        data = copy.deepcopy(COLLECTION)
        data['speakers']['chey'].update(display_name='Another name', age=12, gender='female')
        speaker = SpeakerCollection('room', data).for_model('/models/en_US-ozgirl_v6-step18500.onnx')
        self.assertEqual(speaker['speaker_id'], 'chey')
        self.assertEqual(speaker['age'], 12)
        self.assertEqual(speaker['display_name'], 'Another name')

    def test_invalid_color_and_missing_required_fields(self):
        for key, value in [('face_color', 'red'), ('role', ''), ('voice', '/absolute/model.onnx')]:
            data = copy.deepcopy(COLLECTION)
            data['speakers']['chey'][key] = value
            with self.assertRaises(ValueError): SpeakerCollection('room', data)

    def test_teacher_swap_and_chey_pronunciation(self):
        collection = SpeakerCollection('voice-classroom', COLLECTION)
        for peer, address, absent in [('catherine-byers', 'Mrs. Byers', 'Mrs. Johnson'),
                                      ('linda-johnson', 'Mrs. Johnson', 'Mrs. Byers')]:
            prompt = identity_prompt(collection, collection.speaker('chey'), peer)
            self.assertIn('Chey', prompt)
            self.assertIn('shy', prompt)
            self.assertIn(address, prompt)
            self.assertNotIn(absent, prompt)
        unpaired = identity_prompt(collection, collection.speaker('chey'))
        self.assertNotIn('Mrs.', unpaired)
        self.assertNotIn('Use your teacher’s address name', unpaired)
        self.assertIn('person who just answered', unpaired)
        self.assertIn('Use your teacher’s address name',
                      identity_prompt(collection, collection.speaker('chey'), 'catherine-byers'))

    def test_unassigned_enrollment_competes_without_becoming_known_speaker(self):
        collection = SpeakerCollection('voice-classroom', COLLECTION)
        tracker = mod.PeerIdentity(collection, ['catherine-byers', 'linda-johnson'])
        state = tracker.observe({'lessac': .7, 'ljspeech': .1, 'ozgirl': .1, 'hfc_male': .95}, 1)
        self.assertEqual(state['status'], 'other_voice')
        self.assertIsNone(state['speaker_id'])

    def test_duplicate_collection_ids_in_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'speakers.json'
            path.write_text('{"collections":{"room":{},"room":{}}}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                load_collection({'speaker_collection_id': 'room', 'speaker_collections': str(path)})

if __name__ == '__main__': unittest.main()

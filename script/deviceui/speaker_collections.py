"""Stable collection/speaker identities; names are editable metadata, not keys."""
import json
from pathlib import Path
import re


class SpeakerCollection:
    def __init__(self, collection_id, data):
        if not isinstance(collection_id, str) or not collection_id.strip():
            raise ValueError('A collection needs a nonempty ID')
        self.id = collection_id
        self.speakers = data.get('speakers')
        if not isinstance(self.speakers, dict) or not self.speakers:
            raise ValueError('A collection needs speakers keyed by unique ID')
        self.by_voice = {}
        self.by_model = {}
        for speaker_id, info in self.speakers.items():
            if not isinstance(speaker_id, str) or not speaker_id.strip() or not isinstance(info, dict):
                raise ValueError('Invalid speaker ID or metadata')
            for key in ('display_name', 'voice_id', 'voice', 'face_color', 'role'):
                if not isinstance(info.get(key), str) or not info[key].strip():
                    raise ValueError(f'{speaker_id}: {key} is required')
            for key in ('address_name', 'pronunciation'):
                if key in info and (not isinstance(info[key], str) or not info[key].strip()):
                    raise ValueError(f'{speaker_id}: invalid {key}')
            if not re.fullmatch(r'#[0-9a-fA-F]{6}', info['face_color']):
                raise ValueError(f'{speaker_id}: face_color must be a six-digit hex color')
            if Path(info['voice']).name != info['voice']:
                raise ValueError('Speaker voice must be a model filename, not a device path')
            if info['voice_id'] in self.by_voice or info['voice'] in self.by_model:
                raise ValueError('A voice can belong to only one speaker per collection')
            self.by_voice[info['voice_id']] = speaker_id
            self.by_model[info['voice']] = speaker_id

    def speaker(self, speaker_id):
        if speaker_id not in self.speakers:
            return None
        # Canonical IDs cannot be overridden by optional metadata.
        return dict(self.speakers[speaker_id], collection_id=self.id, speaker_id=speaker_id)

    def for_model(self, model):
        return self.speaker(self.by_model.get(Path(model).name))

    def allowed(self, cfg):
        ids = cfg.get('peer_speaker_ids', [])
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or x not in self.speakers for x in ids):
            raise ValueError('peer_speaker_ids must reference speakers in the selected collection')
        return set(ids)


def load_collection(cfg):
    collection_id = cfg.get('speaker_collection_id')
    if collection_id is None:
        if cfg.get('speaker_identity', {}).get('enabled'):
            raise ValueError('Speaker identification requires speaker_collection_id and peer_speaker_ids')
        return None
    path = Path(cfg.get('speaker_collections', Path(__file__).with_name('speakers.json'))).expanduser()
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj
    data = json.loads(path.read_text(), object_pairs_hook=unique_object)
    collections = {key: SpeakerCollection(key, value) for key, value in data['collections'].items()}
    if collection_id not in collections:
        raise ValueError('Unknown speaker_collection_id')
    return collections[collection_id]


def identity_prompt(collection, own, peer_id=None):
    """Configured participants, never an assertion about uncertain microphone audio."""
    if own is None:
        return ''
    text = '\nYour configured speaker name is ' + own['display_name'] + '.'
    if own.get('pronunciation'):
        text += ' Pronounce it like ' + own['pronunciation'] + '.'
    peer = collection.speaker(peer_id)
    if own['role'] == 'teacher' and peer and peer['role'] == 'student':
        text += ' Your configured student is ' + peer['display_name'] + '.'
        if peer.get('pronunciation'):
            text += ' Their name is pronounced like ' + peer['pronunciation'] + '.'
    if own['role'] == 'student':
        if peer and peer['role'] == 'teacher':
            text += (' Use your teacher’s address name naturally when appropriate, for example before or after a question, '
                     'as a polite form of address. Do not repeat it in every turn. If the speaker is unclear, avoid guessing a name.')
            text += ' The configured teacher for this lesson is ' + peer['display_name'] + '; address them as ' + peer.get('address_name', peer['display_name']) + '.'
        else:
            # A room participant is not necessarily talking with their teacher.
            # The paired-classroom politeness policy must not invent an addressee.
            text += (' Use a person’s name only when addressing that person. Direct follow-up questions to the person '
                     'who just answered unless you intentionally change whom you are addressing. '
                     'If the speaker is unclear, omit the name or ask for clarification.')
    return text

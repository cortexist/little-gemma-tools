"""Build one listener's prior knowledge without exposing the complete roster."""
import json
from pathlib import Path

from speaker_collections import SpeakerCollection

SHORT_TURN_POLICY = ('This is a live spoken exchange. Reply in one or two short sentences, '
                     'at most 45 words total, then yield the floor. Answer one point per turn. '
                     'Use plain speech without markdown, lists, stage directions or tool calls. '
                     'Do not invent another participant\'s reply.')

PENDING_QUESTION_POLICY = (
    'After you ask someone a question, give that person a chance to answer before '
    'asking another person a new question. A brief acknowledgment, unfinished remark, '
    'or interruption that another speaker withdraws does not answer your question '
    'or change its addressee. Keep listening for the pending answer; do not switch '
    'partners just because someone else spoke most recently. Change course when '
    'someone clearly redirects the discussion, asks you a new question, or needs '
    'urgent attention. Do not invent an answer or repeat the pending question '
    'merely to fill a pause.')

CHEY_FOLLOWUP_HINT = (
    'After an invited guest answers a question you asked, take the next available turn '
    'to ask one brief follow-up if the answer raises a specific point worth exploring. '
    'Keep the question with that guest. Yield if someone else is addressed or the topic '
    'is closed; do not speak merely to fill silence.')

TITLE_NAME_POLICY = (
    'Keep a person\'s title and name as separate spoken cues. Hear both '
    'phonetically, but allow more variation in the title than in the name. '
    'The name and acoustic voice are stronger identity evidence; a plausible '
    'sound-alike title alone does not make a new person. If a speaker briefly corrects '
    'their own title and name, attach the correction to that speaker\'s voice. '
    'It is not an address to another person or a request to answer. A brief '
    'correction is allowed after an introduction without repeating the introduction.')

ASR_INPUT_POLICY = (
    'The words you hear arrive as automatic speech recognition (ASR) text, not typed messages. '
    'Transcription errors, missing punctuation, and incorrect word boundaries are common. '
    'Use the topic and admitted conversation history to interpret a plausible sound-alike '
    'when the intended meaning is clear. If someone asks about an unclear term resembling '
    'your recent wording, briefly clarify what you meant and answer; do not invent a new term. '
    'If the meaning remains uncertain, ask one short clarification question. '
    'A noisy word before an otherwise correct title and name is not a mistaken title; '
    'do not interrupt the answer with an unnecessary name correction. '
    'This tolerance concerns meaning, not new identity evidence: preserve accepted voice '
    'bindings and copy only explicit name or spelling evidence for requested name metadata. '
    'Do not guess a speaker\'s name, change the addressee without evidence, or complete '
    'an unfinished request from imagined words.')


class ClassroomScenario:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).with_name('trio-classroom.json')
        self.data = json.loads(path.read_text(encoding='utf-8'))
        collection_id = self.data['collection_id']
        collections = json.loads(path.with_name('speakers.json').read_text(encoding='utf-8'))
        self.collection = SpeakerCollection(collection_id, collections['collections'][collection_id])
        self.participants = self.data['participants']
        for speaker_id, policy in self.participants.items():
            own = self.collection.speaker(speaker_id)
            if not own or own['role'] != policy['role']:
                raise ValueError('Scenario role must match collection identity')
            if (speaker_id in policy['known'] or len(set(policy['known'])) != len(policy['known'])
                    or any(peer not in self.participants for peer in policy['known'])):
                raise ValueError('Invalid prior acquaintances')

    def priors(self, listener, load_embedding):
        result = []
        for peer in self.participants[listener]['known']:
            identity = self.collection.speaker(peer)
            result.append(dict(name=identity.get('address_name', identity['display_name']),
                               embedding=load_embedding(peer),
                               trusted_introducer=self.participants[peer]['trusted_introducer']))
        return result

    def prompt(self, listener):
        own = self.collection.speaker(listener)
        policy = self.participants[listener]
        names = [self.collection.speaker(peer).get('address_name', self.collection.speaker(peer)['display_name'])
                 for peer in policy['known']]
        title = own.get('address_title')
        name_part = own.get('address_name_part')
        if title and name_part:
            own_address = ('In your form of address, your title is '+title+' and your name is '
                           +name_part+'. Treat these as separate spoken parts. ')
            if listener == 'agent-smith':
                own_address += ('If someone uses a plausible sound-alike title with '+name_part+', '
                                'you may briefly correct it with "Agent, Agent Smith." '
                                'That is a correction, not a second introduction. ')
        else:
            own_address = ('Your name '+own['display_name']+' has no title. ')
        prompt = ('Your name is '+own['display_name']+'. Your role is '+policy['role']+'. '
                  +own_address+
                  'You already know '+', '.join(names)+'. '+policy['instructions']+' '
                  'A voice ID is an anonymous local track, not a name. Use learned names only after an accepted '
                  'binding. If identity is uncertain, avoid named address or ask for clarification. '
                  +TITLE_NAME_POLICY+' '+ASR_INPUT_POLICY+' '+PENDING_QUESTION_POLICY+' '+SHORT_TURN_POLICY)
        return prompt + ('\n' + CHEY_FOLLOWUP_HINT if listener == 'chey' else '')

"""Silent listener decisions, explicit-address checks, and floor arbitration."""
import json
import re

PRIORITY = {'addressed': 4, 'followup': 3, 'invited': 2, 'answer': 1, 'defer': 0}
ROOM_POLICY = (
    'This is a room conversation. Everyone hears every other speaker. '
    'Messages ending in ROOM_OBSERVE require a silent JSON decision, never a spoken answer. '
    'ROOM_IDENTIFY requires silent name metadata or NONE, never speech. '
    'Decide whether YOU should answer. Defer when someone else is addressed, '
    'including a question returned to the previous conversational partner. '
    'A mention of your name is not necessarily an address. Do not alternate addressees. '
    'Only ROOM_SPEAK grants permission to speak; then answer briefly in plain speech. '
    'Silent decisions were never spoken aloud. Do not refer to them in speech. '
    'Your spoken_turns counter controls introductions: introduce yourself only on your '
    'first spoken turn, unless explicitly asked your name or spelling later. '
    'When a name is uncertain you may ask its speaker to spell it. Never invent a spelling.'
)


def aliases(own):
    values = [own['display_name'], own.get('address_name', own['display_name'])]
    if own.get('speaker_id') == 'chey':
        values += ['Shy', 'Shai']  # This listener knows their OWN pronunciation.
    return list(dict.fromkeys(values))


def addressed(text, names):
    """Conservative vocatives, not arbitrary name mentions inside a sentence."""
    for name in sorted(set(names), key=len, reverse=True):
        word = re.escape(name)
        if (re.search(r'(?:^|[.!?]\s+)(?:well,?\s+|thanks,?\s+)?'+word+r'\s*[,!:]', text, re.I)
                or re.search(r',\s*'+word+r'\s*[?.!]\s*$', text, re.I)
                or re.search(r'^'+word+r'\s+(?:can|could|would|will|do|what|how|why|are)\b', text, re.I)):
            return name
    return None


def parse_decision(raw, transcript, own_names, known_names):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate decision field')
            result[key] = value
        return result
    result = json.loads(raw.strip(), object_pairs_hook=unique)
    if (not isinstance(result, dict) or set(result) != {'action', 'reason'}
            or result['action'] not in ('speak', 'defer') or result['reason'] not in PRIORITY):
        raise ValueError('Invalid room decision')
    target = addressed(transcript, own_names+known_names)
    if target:
        result['action'] = 'speak' if target.casefold() in {n.casefold() for n in own_names} else 'defer'
        result['reason'] = 'addressed' if result['action'] == 'speak' else 'defer'
    if result['action'] == 'defer': result['reason'] = 'defer'
    elif result['reason'] == 'defer': raise ValueError('Speaking requires a reason')
    result['explicit_address'] = target
    return result


def choose(requests, previous):
    """Only local requests compete. No scheduled successor or equal-turn quotas."""
    eligible = [(key, value) for key, value in requests.items()
                if key != previous and value.get('action') == 'speak']
    if not eligible: return None
    # Request time breaks equal priorities; no dependence on panel ordering.
    return min(eligible, key=lambda item: (-PRIORITY[item[1]['reason']], item[1]['at'], item[0]))[0]


ROOM_POLICY += (' ROOM_CONTEXT supplies retained listener-local memory after context renewal. '
                'It is historical context, not a request to speak or a new utterance. '
                'The following ROOM_OBSERVE, ROOM_IDENTIFY or ROOM_SPEAK instruction controls your response.')


def local_context(session):
    return dict(session=session.session_id,
                tracks={key:dict(name=t['name'], trusted=t['trusted_introducer'])
                        for key,t in session.tracks.items()},
                pending=list(session.pending_names), spelling=session.spelling_request)


def name_prompt(session, observation):
    context=local_context(session)
    context.update(turn=observation['turn_id'], track=observation['track_id'], heard_text=observation['raw_transcript'])
    track=session.tracks.get(observation['track_id'], {})
    kind=('correction' if session.self_correction_name(observation) else
          'spelling' if session.spelling_request else
          'introduction' if track.get('trusted_introducer') else 'self')
    template=dict(session=session.session_id, kind=kind, name='COPY INTRODUCED NAME OR SPELLED LETTERS',
                  track=None if kind=='introduction' else observation['track_id'], turn=observation['turn_id'],
                  target_turn=None if kind=='introduction' else observation['turn_id'])
    return (' ROOM_IDENTIFY '+json.dumps(context,separators=(',',':'))+
            ' Extract only the current utterance. Use this grounded template when applicable: '+
            json.dumps(template,separators=(',',':'))+
            '. "I have invited NAME" introduces the guest, even if the known teacher also introduces themself. '
            '"Hello, I am NAME" is a self-introduction. A spelling answer joins the literal letters as the name. '
            'Do not invent IDs. A guest introduction never names the teacher. Otherwise output NONE.')


def needs_name(session, observation):
    track=session.tracks.get(observation['track_id'], {})
    text=observation['raw_transcript']
    if session.self_correction_name(observation):
        return True
    if session.spelling_request and session.spelling_request['track'] == observation['track_id']:
        return True
    if track.get('trusted_introducer') and re.search(r'\b(?:introduce|invited|(?:our(?: special)?|special|invited) guest|just spoke|just heard)\b', text, re.I):
        return True
    return not track.get('name') and bool(re.search(r"\b(?:my name|call me|i am|i['’]m)\b", text, re.I))


def decision_prompt(session, observation, own, spoken_turns, conversation=None):
    track=session.tracks.get(observation['track_id'], {})
    context=dict(heard_turn=observation['turn_id'], voice=observation['track_id'],
                 name=track.get('name'), heard_text=observation['raw_transcript'],
                 own=own['display_name'], role=own['role'],
                 spoken_turns=spoken_turns, conversation=conversation)
    partner = ''
    if conversation:
        previous = (conversation['own_last_spoken_epoch'] > 0 and
                    conversation['own_last_spoken_epoch'] == conversation['current_epoch']-1)
        partner = (' You spoke immediately before this person. A question returned to the previous '
                   'conversational partner is returned to YOU; request speak to answer it.' if previous else
                   ' You did not speak immediately before this person. A question returned to the '
                   'previous conversational partner belongs to someone else; defer to that partner.')
    return (' ROOM_OBSERVE. SILENT DECISION ONLY. The preceding speech was heard by everyone. '
            'Decide if YOU should respond next. Return exactly one JSON object with exactly '
            'two fields, action and reason. action is "speak" or "defer". reason is '
            '"addressed", "followup", "invited", "answer", or "defer". '
            'For example {"action":"speak","reason":"addressed"}. '
            'An explicit question to someone else means defer. A returned question belongs '
            'to the previous conversational partner. If you are the student and have not '
            'spoken yet, a guest greeting after the teacher invited questions is reason '
            '"invited" to set action="speak". After an answer to your own question, a relevant '
            'follow-up is reason "followup" to set action="speak". '
            'These are requests for permission, NOT instructions to say the answer now. '
            'Do not introduce yourself here. No spoken words, name metadata, markdown, or '
            'explanation outside the two-field JSON. Listener-local context: '+json.dumps(context)+
            partner+' END ROOM_OBSERVE. Output the two-field JSON now.')


DIRECT_ROOM_POLICY = (
    'You are one participant in a live room, listening to every speaker. '
    'For each input, produce ONE response: [defer] if another person should answer, '
    'otherwise the actual words you want to say. This ends your turn. '
    'Never answer a question addressed to someone else. An invitation or question '
    'returned to the previous conversational partner belongs to that partner. '
    'The student may introduce themself and ask a first question after the guest greeting. '
    'Answer briefly, at most 45 words. Introduce yourself only on your first spoken turn, '
    'unless someone asks your name again. A brief correction of a mistaken title or name '
    'is allowed later; it is not another introduction. Never invent someone else\'s reply. '
    'An interrupted response ends that turn. Let the interrupting speaker finish; '
    'do not restart an unheard draft or repeat a greeting automatically. '
    'ROOM_OPEN asks the teacher to introduce the guest, then invite the guest to introduce themself. '
    'ROOM_CONTEXT contains retained history, not a new utterance. '
    'ROOM_INPUT gives the current speaker and your conversation state. '
    'NAME_UPDATE requests a private name from the CURRENT utterance. '
    'When a speaker corrects their own title and name, the corrected name belongs to '
    'that acoustic voice; it is not an addressee or a request for you to answer. '
    'If NAME_UPDATE is requested, FIRST output the extracted name between <speaker_name> and </speaker_name>. Then output your response. '
    'For example: <speaker_name>Dr. Jones</speaker_name>[defer] '
    'Or: <speaker_name>Dr. Jones</speaker_name>Hello, Dr. Jones. '
    'The tag is never spoken. No JSON. If asked to extract a spelling, join only the literal letters. '
    'You may ask a person with an uncertain name how to spell it. '
    'When speaking, you may sparingly use [happy], [sad], [angry], [neutral], '
    '[nod], [shake] or [quiet] to express a suitable facial reaction. These are '
    'visual cues, never spoken words. Place them after the required decision and '
    'name metadata, alongside the words they accompany. A deferred response '
    'must not contain facial cues.'
)


DIRECT_ROOM_TAG_POLICY = DIRECT_ROOM_POLICY.replace(
    'For each input, produce ONE response: [defer] if another person should answer, '
    'otherwise the actual words you want to say. This ends your turn.',
    'For each input, FIRST output exactly one private decision tag: [speak] if YOU '
    'should answer, or [defer] if another person should answer. This tag must come '
    'before any speaker_name tag or spoken words. Then emit any requested name '
    'metadata. After [speak], say your actual response. After [defer], emit no '
    'spoken words. This is one response and ends your turn.'
).replace(
    'If NAME_UPDATE is requested, FIRST output the extracted name between <speaker_name> and </speaker_name>. Then output your response.',
    'If NAME_UPDATE is requested, AFTER the decision tag output the extracted name '
    'between <speaker_name> and </speaker_name>, before any spoken words.'
).replace(
    '<speaker_name>Dr. Jones</speaker_name>[defer]',
    '[defer]<speaker_name>Dr. Jones</speaker_name>'
).replace(
    '<speaker_name>Dr. Jones</speaker_name>Hello, Dr. Jones.',
    '[speak]<speaker_name>Dr. Jones</speaker_name>Hello, Dr. Jones.'
)


WAIT_ASSURANCE = ('A direct address identifies who may answer; it does not complete '
                  'an unfinished request. Use only admitted words. Wait if the '
                  'request is still developing, and answer a complete short '
                  'question or greeting when it is your turn.')


def response_prompt(session, observation, own, spoken_turns, conversation, *,
                    decision_tags=False, wait_assurance=False):
    track = session.tracks.get(observation['track_id'], {})
    previous = conversation.get('previous_partner_is_me', (conversation['own_last_spoken_epoch'] > 0 and
                conversation['own_last_spoken_epoch'] == conversation['current_epoch']-1))
    context = dict(turn=observation['turn_id'], voice=observation['track_id'],
                   name=track.get('name'), own=own['display_name'], role=own['role'],
                   spoken_turns=spoken_turns, previous_partner_is_me=previous,
                   last_own_speech=conversation['own_last_spoken_text'])
    prompt = ' ROOM_INPUT '+json.dumps(context, separators=(',', ':'))
    correction = session.self_correction_name(observation)
    extract_name = needs_response_name(session, observation, own)
    if extract_name:
        kind = ('correction' if correction else
                'spelling' if session.spelling_request and session.spelling_request['track'] == observation['track_id'] else
                'introduction' if track.get('trusted_introducer') else 'self')
        prompt += (' NAME_UPDATE: '+kind+'. Current utterance transcript: '
                   +json.dumps(observation['raw_transcript'], ensure_ascii=False)+'. '
                   'Copy the name from this transcript, not from your own identity, '
                   'a known speaker, or an earlier turn. The name tag is private metadata '
                   'about a voice you heard; it does not name the person who will answer.')
        if kind == 'introduction':
            prompt += (' Extract the invited GUEST name from the teacher\'s introduction. '
                       'Use the name after the invitation or guest-introduction phrase. '
                       'Ignore the teacher\'s own "I am" introduction; the teacher is already known.')
        elif kind == 'self':
            prompt += (' This acoustic voice has NO accepted name yet, even if the name appeared earlier. '
                       'Copy only the name this speaker says after "my name is", "call me", or "I am". '
                       'Its self-introduction must be tagged now to bind the name to this voice.')
        elif kind == 'correction':
            prompt += (' This speaker is correcting THEIR OWN title and name. Copy the full corrected '
                       'name after the pause into the private name tag, even if this acoustic voice '
                       'already has a name. The correction refers to the current speaker, not an addressee.')
    prompt += (' Decide whether this utterance calls for YOU to speak. Defer to a named '
               'addressee other than yourself, even on your first turn. ')
    prompt += ('FIRST output [speak] or [defer], then any requested name tag, then spoken words only for [speak].'
               if decision_tags else 'After any requested name tag, output [defer] or answer directly.')
    if spoken_turns:
        prompt += (' Do not repeat an introduction you have already given. '
                   'You may briefly correct a mistaken title or name without repeating your introduction.')
        if own['role'] == 'student' and not conversation.get('introduction_confirmed', False):
            prompt += (' If you have not yet told this new guest your name, give it briefly '
                       'before your next lesson question.')
    if extract_name:
        prompt += (' AFTER the decision tag, extract the name from THIS utterance in <speaker_name>NAME</speaker_name>. Do not skip the name tag when deferring.'
                   if decision_tags else ' FIRST extract the name from THIS utterance in <speaker_name>NAME</speaker_name>. THEN output [defer] or answer. Do not skip the name tag when deferring.')
    else:
        prompt += ' No name update for this input. Do not output a speaker_name tag.'
    if own['role'] == 'student' and not spoken_turns and not track.get('name'):
        turns = list(session.turns)
        just_introduced = any(pending['turn_id'] == turns[-2] for pending in session.pending_names.values()) if len(turns) >= 2 else False
        self_intro = re.search(r"\b(?:my name is|call me|i am|i['’]m)\b",
                               observation['raw_transcript'], re.I)
        uncertain_name = just_introduced and self_intro and not any(
            re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', observation['raw_transcript'], re.I)
            for name in session.pending_names)
        if uncertain_name:
            prompt += (' A trusted teacher introduced this guest, but the guest name in the current '
                       'transcript differs. Do not say that uncertain guest name. If you speak, '
                       'introduce yourself and ask the guest one short question about the lesson '
                       'without using a guest name.')
    known = [t['name'] for t in session.tracks.values() if t['name']]+list(session.pending_names)
    target = None if correction else addressed(observation['raw_transcript'], aliases(own)+known)
    if correction:
        prompt += (' The current speaker corrected their own name. This is identity metadata '
                   'for that voice, not an address to anyone in the room.')
    if target:
        mine = target.casefold() in {n.casefold() for n in aliases(own)}
        prompt += (' The current utterance explicitly addresses YOU ('+target+').'+
                   ('' if wait_assurance else ' Give your answer.') if mine else
                   ' The current utterance explicitly addresses '+target+', someone other than you. Output [defer].')
    elif (own['role'] == 'guest' and not spoken_turns and track.get('trusted_introducer')
          and re.search(r'\b(?:please introduce|introduce yourself|introduce themsel)', observation['raw_transcript'], re.I)):
        prompt += (' You are the invited guest. Your teacher is asking the guest to introduce themself; '
                   'this invitation is for YOU. Introduce yourself briefly now.')
    prompt += (' End of current input. FIRST output the decision tag, then any requested name tag, then your response only for [speak].'
               if decision_tags else ' End of current input. Output the required name tag first if requested, then your response.')
    if extract_name:
        prompt += (' Start with [speak] or [defer]; the <speaker_name> tag comes AFTER that decision.'
                   if decision_tags else ' Start your output with <speaker_name> and the name extracted from the current speech.')
    if target and mine and not wait_assurance:
        prompt += (' Start with [speak] and answer the question addressed to YOU now. [defer] would be incorrect here.'
                   if decision_tags else ' Answer the question addressed to YOU now. [defer] would be incorrect here.')
    if wait_assurance:
        prompt += ' ' + WAIT_ASSURANCE
    return prompt


def needs_response_name(session, observation, own=None):
    if not needs_name(session, observation): return False
    track = session.tracks.get(observation['track_id'], {})
    if own and track.get('trusted_introducer'):
        names = '|'.join(re.escape(name) for name in aliases(own))
        if re.search(r'\b(?:invited|(?:our(?: special)?|special|invited) guest)[,:\s]+(?:'+names+r')(?!\w)',
                     observation['raw_transcript'], re.I):
            # Being introduced to others is not learning a new acquaintance.
            return False
    return True


def accept_response_names(session, observation, values, own=None):
    """Attach runtime-owned references, then apply the existing evidence checks."""
    if not values:
        if observation and needs_response_name(session, observation, own):
            return dict(spoken='', accepted=[], rejected=['Missing requested name update'])
        return session.accept('NONE', metadata_only=True)
    if len(values) != 1 or not observation or not needs_response_name(session, observation, own):
        return dict(spoken='', accepted=[], rejected=['Unexpected name update'])
    track = session.tracks[observation['track_id']]
    kind = ('correction' if session.self_correction_name(observation) else
            'spelling' if session.spelling_request and session.spelling_request['track'] == observation['track_id'] else
            'introduction' if track.get('trusted_introducer') else 'self')
    proposal = dict(session=session.session_id, kind=kind, name=values[0].strip(),
                    turn=observation['turn_id'], track=None if kind=='introduction' else observation['track_id'],
                    target_turn=None if kind=='introduction' else observation['turn_id'])
    return session.accept('<speaker_name>'+json.dumps(proposal)+'</speaker_name>', metadata_only=True)

"""Session-local anonymous voice tracks and evidence-backed name proposals.

The acoustic tracker never consults the room's complete speaker roster. A
listener receives only its own prior acquaintances. Model output proposes a
name; this module checks the referenced session, turns, and acoustic target.
Clean single-speaker segments are a caller prerequisite, not inferred here.
"""
from copy import deepcopy
import json
import math
import re
import uuid

from speech_names import canonical_name

TAG = re.compile(r'<speaker_name>(.*?)</speaker_name>', re.S)
TITLES = {'agent', 'major', 'mr', 'mrs', 'miss', 'ms', 'dr', 'doctor',
          'prof', 'professor', 'captain', 'officer', 'sergeant', 'general',
          'colonel', 'sir', 'lady'}


def name_core(value):
    """Keep the name strict while allowing a separate spoken title."""
    words = value.split()
    if len(words) > 1 and words[0].rstrip('.').casefold() in TITLES:
        words = words[1:]
    return ' '.join(word.strip(".,!").casefold() for word in words)


def unit(values):
    values = tuple(float(x) for x in values)
    norm = math.sqrt(sum(x*x for x in values))
    if not values or not all(math.isfinite(x) for x in values) or not math.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid speaker embedding')
    return tuple(x/norm for x in values)


def proposals(text):
    """Separate complete identity tags from speech; malformed tags stay silent."""
    result, errors = [], []
    for match in TAG.finditer(text):
        try:
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError('Duplicate name proposal field')
                    result[key] = value
                return result
            value = json.loads(match[1], object_pairs_hook=unique)
            if not isinstance(value, dict):
                raise ValueError('name proposal must be an object')
            result.append(value)
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    spoken = TAG.sub('', text)
    # A truncated control message must never reach a speaker.
    start = spoken.find('<speaker_name')
    if start >= 0:
        errors.append('incomplete name proposal')
        spoken = spoken[:start]
    return spoken.strip(), result, errors


class SpeakerSession:
    def __init__(self, known=(), *, match_threshold=.425, new_threshold=.3,
                 margin=.1, max_tracks=8, max_turns=200, canonicalize=True):
        if not 0 <= new_threshold < match_threshold <= 1 or not 0 < margin <= 2:
            raise ValueError('Invalid voice matching thresholds')
        if not 1 <= max_tracks <= 32 or not 1 <= max_turns <= 2000:
            raise ValueError('Invalid session bounds')
        self.known = deepcopy(list(known))
        names = [p.get('name') for p in self.known]
        if (any(not isinstance(n, str) or not n.strip() for n in names)
                or len({n.casefold() for n in names}) != len(names)):
            raise ValueError('Prior acquaintances need distinct names')
        self.match_threshold, self.new_threshold, self.margin = match_threshold, new_threshold, margin
        self.max_tracks, self.max_turns = max_tracks, max_turns
        self.canonicalize = canonicalize
        self.reset()

    def reset(self):
        self.session_id = uuid.uuid4().hex[:12]
        self.tracks, self.turns, self.pending_names, self.audit = {}, {}, {}, []
        self.spelling_request = None
        self.dimension = None
        for prior in self.known:
            track = self._new(prior['embedding'])
            track.update(name=prior['name'], name_source='prior',
                         trusted_introducer=bool(prior.get('trusted_introducer', False)))
        return self.snapshot()

    def _new(self, embedding):
        vector = unit(embedding)
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError('Embedding dimension changed')
        if len(self.tracks) >= self.max_tracks:
            raise ValueError('Session voice limit reached; reset or review tracks')
        self.dimension = len(vector)
        track_id = 'voice-'+str(len(self.tracks)+1)
        track = dict(track_id=track_id, embedding=vector, observations=0,
                     name=None, name_source=None, trusted_introducer=False)
        self.tracks[track_id] = track
        return track

    def observe(self, turn_id, embedding, transcript, *, clean=False, duration=0):
        """Associate one bounded, completed utterance with an acoustic track.

        `clean` means the caller established a single non-self speaker with no
        overlap/gap. False/short input is retained without a usable voice label.
        Repeated IDs cannot reinforce or replace evidence.
        """
        if not isinstance(turn_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', turn_id):
            raise ValueError('Invalid turn ID')
        if turn_id in self.turns or len(self.turns) >= self.max_turns:
            raise ValueError('Duplicate turn or session turn limit')
        if not isinstance(transcript, str) or len(transcript) > 8000:
            raise ValueError('Invalid transcript')
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid duration')
        if not isinstance(clean, bool):
            raise ValueError('Clean-segment flag must be boolean')
        track, score, status = None, None, 'unattributed'
        if clean and duration >= .75:
            vector = unit(embedding)
            if self.dimension is not None and len(vector) != self.dimension:
                raise ValueError('Embedding dimension changed')
            scored = sorted(((sum(a*b for a, b in zip(vector, t['embedding'])), t['track_id'])
                             for t in self.tracks.values()), reverse=True)
            score = scored[0][0] if scored else None
            separated = len(scored) < 2 or score-scored[1][0] >= self.margin
            if scored and score >= self.match_threshold and separated:
                track = self.tracks[scored[0][1]]
                status = 'matched'
                # Keep prior enrollments immutable. Update anonymous centroids
                # only after a separated match, never during an ambiguous turn.
                if track['name_source'] != 'prior':
                    n = min(track['observations'], 10)
                    track['embedding'] = unit([a*n+b for a, b in zip(track['embedding'], vector)])
            elif not scored or score < self.new_threshold:
                track = self._new(vector)
                status = 'new_voice'
            else:
                status = 'uncertain'
            if track:
                track['observations'] += 1
        record = dict(turn_id=turn_id, track_id=track['track_id'] if track else None,
                      raw_transcript=transcript, status=status, similarity=score,
                      clean=bool(clean), duration=duration)
        self.turns[turn_id] = record
        return deepcopy(record)

    def preview(self, embedding):
        """Score a live window without reinforcing tracks or assigning names."""
        vector = unit(embedding)
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError('Embedding dimension changed')
        scores = sorted(((sum(a*b for a, b in zip(vector, t['embedding'])), key)
                         for key, t in self.tracks.items()), reverse=True)
        score = scores[0][0] if scores else None
        separated = len(scores) < 2 or score-scores[1][0] >= self.margin
        if scores and score >= self.match_threshold and separated:
            track = self.tracks[scores[0][1]]
            return dict(status='matched', track_id=track['track_id'], name=track['name'],
                        name_source=track['name_source'], similarity=score)
        return dict(status='unknown' if not scores or score < self.new_threshold else 'uncertain',
                    track_id=None, name=None, name_source=None, similarity=score)

    def self_correction_name(self, observation):
        """Recognize a narrow title, full-name correction by the same voice."""
        if not observation or not observation.get('track_id'):
            return None
        text = observation['raw_transcript']
        match = re.match(
            r"^\s*(?P<first>[A-Za-z]+)\.?\s*[,!]\s*"
            r"(?P<title>[A-Za-z]+)\.?\s+(?P<core>[A-Za-z][A-Za-z' -]{0,60}?)"
            r"\s*(?:[.!]|$)", text, re.I)
        if not match or any(ch in text[:match.end()] for ch in '"“”'):
            return None
        first, title = match['first'].casefold(), match['title'].casefold()
        if first not in TITLES or title not in TITLES:
            return None
        full = match['title'] + ' ' + match['core'].strip()
        core = name_core(full)
        target = self.tracks[observation['track_id']]
        if target['name']:
            return full if name_core(target['name']) == core else None
        if not any(name_core(pending) == core for pending in self.pending_names):
            return None
        # An unbound voice must already have made its own name claim. Otherwise
        # a first-time vocative could be mistaken for this speaker's correction.
        earlier = [turn['raw_transcript'] for turn in self.turns.values()
                   if turn['track_id'] == target['track_id'] and turn['turn_id'] != observation['turn_id']]
        declaration = (r"\b(?:my name is|call me|i am|i['’]m)\s+"
                       r"(?:[A-Za-z]+\.?\s+)?" + re.escape(match['core'].strip()) + r"\b")
        return full if any(re.search(declaration, utterance, re.I) for utterance in earlier) else None

    def apply(self, event):
        """Validate an LLM proposal. Learning cannot create acoustic tracks."""
        expected = {'session', 'kind', 'name', 'track', 'turn', 'target_turn'}
        if not isinstance(event, dict) or set(event) != expected:
            raise ValueError('Invalid name proposal fields')
        if event['session'] != self.session_id:
            raise ValueError('Stale name proposal from another session')
        kind, name = event['kind'], event['name']
        if kind not in ('self', 'introduction', 'spelling', 'correction') or not isinstance(name, str):
            raise ValueError('Invalid proposal kind/name')
        if not 1 <= len(name) <= 80 or any(c in name for c in '<>{}[]\n\r'):
            raise ValueError('Invalid proposed name')
        name = ' '.join(name.split())
        if not name:
            raise ValueError('Empty proposed name')
        for field in ('turn', 'track', 'target_turn'):
            if event[field] is not None and not isinstance(event[field], str):
                raise ValueError('Invalid proposal reference')
        source = self.turns.get(event['turn'])
        if not source or not source['track_id']:
            raise ValueError('Name evidence has no reliable acoustic source')
        if event['turn'] != next(reversed(self.turns)):
            raise ValueError('Name evidence is not the current completed turn')
        source_track = self.tracks[source['track_id']]
        text = source['raw_transcript']
        if kind != 'spelling' and not re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', text, re.I):
            raise ValueError('Proposed name is absent from the evidence transcript')
        target = self.tracks.get(event['track'])
        if kind == 'correction':
            if (not target or target['track_id'] != source['track_id']
                    or event['target_turn'] != event['turn']):
                raise ValueError('Correction must name the acoustic voice that spoke it')
            corrected = self.self_correction_name(source)
            if not corrected or corrected.casefold() != name.casefold():
                raise ValueError('No supported self-name correction in this utterance')
        elif kind == 'spelling':
            request = self.spelling_request
            if (not request or not target or request['track'] != source['track_id']
                    or event['track'] != source['track_id'] or event['target_turn'] != event['turn']
                    or list(self.turns)[-2:-1] != [request['after_turn']]):
                raise ValueError('Spelling must answer a recent request to the same acoustic voice')
            # Require literal, individually spoken letters. Never infer a
            # spelling from the room roster or a homophone dictionary.
            letters = re.findall(r'(?<![A-Za-z])([A-Za-z])(?:\s*[,.-]\s*|\s+|$)', text)
            if (not 2 <= len(letters) <= 30 or ''.join(letters).casefold() != name.casefold()
                    or '?' in text or re.search(r'["“”]', text)):
                raise ValueError('Spelling is not supported by the transcribed letter sequence')
            if target['name_source'] == 'prior' and target['name'].casefold() != name.casefold():
                raise ValueError('Spelling cannot overwrite a prior acquaintance')
        elif kind == 'self':
            if not target or target['track_id'] != source['track_id'] or event['target_turn'] != event['turn']:
                raise ValueError('Self-introduction must name the voice that spoke it')
            # "I am shy" is an adjective as well as a possible name. The
            # homophone spelling rule requires an explicit name declaration.
            cue = r"(?:^|[.!]\s+)(?:(?:hello|hi|good afternoon)(?:\s+everyone)?,?\s+)?(?:my name is|call me|i am|i['’]m)\s+"
            declaration = re.search(cue+re.escape(name)+r'(?!\w)', text, re.I)
            if not declaration:
                raise ValueError('No self-introduction in the evidence')
            clause_end = re.search(r'[.!?]', text[declaration.end():])
            questioning = clause_end and clause_end[0] == '?'
            if questioning or re.search(r'["“”]', text):
                raise ValueError('Quoted or questioning name claims need clarification')
            canonical = canonical_name(name, text)  # Also rejects adjective-only claims.
            name = canonical if self.canonicalize else name
            # A fresh self-introduction can be a different person. But when it
            # follows a trusted introduction and shares the distinctive final
            # name token, a different full name may be an ASR error (Agent vs
            # Major Smith). Keep both claims unbound for clarification.
            order = list(self.turns)
            previous_turn = order[-2] if len(order) > 1 else None
            final_token = lambda value: (re.findall(r'\w+', value.casefold()) or [''])[-1]
            for pending in self.pending_names.values():
                if (pending['turn_id'] == previous_turn
                        and pending['name'].casefold() != name.casefold()
                        and final_token(pending['name']) == final_token(name)
                        and len(final_token(name)) >= 3):
                    raise ValueError('Self-introduction conflicts with a just-introduced similar name; clarify before binding')
        else:
            if not source_track['trusted_introducer']:
                raise ValueError('This speaker is not a known introducer')
            # A clear introduction may be followed by a separate question to
            # the guest. Check the name's own clause instead of rejecting the
            # entire utterance just because a later sentence asks a question.
            declaration = False
            for mentioned in re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)', text, re.I):
                end = re.search(r'[.!?]', text[mentioned.end():])
                start = max(text.rfind(mark, 0, mentioned.start()) for mark in '.!?')+1
                clause = text[start:mentioned.start()]
                tail = text[mentioned.end():mentioned.end()+end.start()] if end else text[mentioned.end():]
                welcomed_guest = (re.search(r'\b(?:pleased|glad|delighted) to have\b', clause, re.I)
                                   and re.search(r'\bjoining us as (?:a|our) guest\b', tail, re.I))
                intro_cue = re.search(
                    r'\b(?:introduce|invited|(?:our(?: special)?|special|invited) guest|this is|that was|just spoke|just heard)\b',
                    clause, re.I)
                if (not end or end[0] != '?') and (intro_cue or welcomed_guest):
                    declaration = True
            if (not declaration or re.search(r'["“”]', text)
                    or re.search(r"\b(?:not|never|isn['’]t|wasn['’]t|don['’]t|didn['’]t)\b", text, re.I)):
                raise ValueError('Questioning, quoted or negative introductions need clarification')
            if event['track'] is None:
                if event['target_turn'] is not None:
                    raise ValueError('An unbound introduction cannot cite a target voice')
                self.pending_names[name] = dict(name=name, turn_id=event['turn'])
                result = dict(event, name=name, status='pending_voice')
                self.audit.append(result)
                return deepcopy(result)
            target_turn = self.turns.get(event['target_turn'])
            order = list(self.turns)
            source_index = order.index(event['turn'])
            if (not target or not target_turn or target_turn['track_id'] != target['track_id']
                    or source_index == 0 or order[source_index-1] != event['target_turn']
                    or target['track_id'] == source['track_id']):
                raise ValueError('Introduction does not reference the preceding other voice')
            if not re.search(r'\b(?:just spoke|just heard|that voice)\b', text, re.I):
                raise ValueError('Introduction has no explicit reference to that voice')
        if kind not in ('spelling', 'correction') and target['name'] and target['name'].casefold() != name.casefold():
            raise ValueError('Conflicting name; keep the existing binding for review')
        if any(t['track_id'] != target['track_id'] and t['name'] and t['name'].casefold() == name.casefold()
               for t in self.tracks.values()):
            raise ValueError('Name already belongs to another voice in this session')
        target.update(name=name, name_source='prior' if target['name_source'] == 'prior' else 'learned')
        if kind == 'correction':
            for pending in list(self.pending_names):
                if name_core(pending) == name_core(name):
                    self.pending_names.pop(pending)
        else:
            self.pending_names.pop(name, None)
        if kind == 'spelling':
            self.spelling_request = None
        result = dict(event, name=name, status='bound')
        self.audit.append(result)
        return deepcopy(result)

    def ask_spelling(self, track, spoken):
        """Open a transaction only after we actually asked the last voice."""
        latest = next(reversed(self.turns.values()), None)
        if (not latest or not track or latest['track_id'] != track
                or not re.search(r'\bspell(?:ing)?\b', spoken, re.I)
                or not re.search(r'\b(?:your name|that name|name for me)\b', spoken, re.I)):
            return False
        self.spelling_request = dict(track=track, after_turn=latest['turn_id'])
        return True

    def accept(self, output, *, metadata_only=False):
        if metadata_only:
            output = output.strip()
            if output == 'NONE':
                return dict(spoken='', accepted=[], rejected=[])
            matches = list(TAG.finditer(output))
            if len(matches) != 1 or matches[0].span() != (0, len(output)):
                return dict(spoken='', accepted=[], rejected=['Expected exactly one framed name proposal or NONE'])
        spoken, events, rejected = proposals(output)
        accepted = []
        for event in events:
            try:
                accepted.append(self.apply(event))
            except (ValueError, TypeError) as exc:
                rejected.append(str(exc))
        return dict(spoken=spoken, accepted=accepted, rejected=rejected)

    def snapshot(self):
        return dict(session=self.session_id,
                    tracks=[{k: deepcopy(v) for k, v in t.items() if k != 'embedding'}
                            for t in self.tracks.values()],
                    turns=deepcopy(list(self.turns.values())),
                    pending_names=deepcopy(list(self.pending_names.values())),
                    spelling_request=deepcopy(self.spelling_request))

    def prompt(self):
        """Only observed voices and this listener's prior/learned names reach the LLM."""
        turns = list(self.turns.values())
        context = dict(session=self.session_id,
                       tracks=[{k: t[k] for k in ('track_id', 'name', 'trusted_introducer')} for t in self.tracks.values()],
                       previous_turns=turns[-6:-1], current_turn=turns[-1] if turns else None,
                       spelling_request=self.spelling_request)
        current = turns[-1] if turns else None
        if current:
            context['unbound_guest_introduction_template'] = dict(session=self.session_id,
                kind='introduction', name='COPY THE GUEST NAME', track=None,
                turn=current['turn_id'], target_turn=None)
            context['self_introduction_template'] = dict(session=self.session_id,
                kind='self', name='COPY THE SELF-INTRODUCED NAME', track=current['track_id'],
                turn=current['turn_id'], target_turn=current['turn_id'])
        return ('Extract a name ONLY from current_turn, never from previous_turns. '
                'If current_turn neither introduces a name nor answers spelling_request, output NONE. '
                'Otherwise output one <speaker_name>JSON object</speaker_name> and nothing else. '
                'The object has exactly six fields: session, kind, name, track, turn, target_turn. '
                'session must equal the supplied session. turn must equal current_turn.turn_id. '
                'name must copy the introduced name exactly as transcribed. '
                'kind must be exactly "self", "introduction", "spelling", or "correction". '
                'For "My name is Robin", kind="self", track=current_turn.track_id, '
                'target_turn=current_turn.turn_id. '
                'A greeting such as "Hello everyone, I am Agent Robin" is also a self-introduction. '
                'If a known teacher introduces themself AND a guest, extract the guest introduction; '
                'do not rename a known teacher. "I have invited Robin" introduces the guest Robin. '
                'For an answer "R, O, B, I, N" to spelling_request, kind="spelling", name="Robin", '
                'track=current_turn.track_id and target_turn=current_turn.turn_id. '
                'For a speaker correcting their own title with "Doctor, Doctor Jones", '
                'kind="correction", name="Doctor Jones", track=current_turn.track_id, '
                'target_turn=current_turn.turn_id; this is not an address to another voice. '
                'For a trusted teacher saying "The person who just spoke is Robin", kind="introduction", '
                'track is the preceding other voice, and target_turn is that preceding turn ID. '
                'For a trusted teacher saying "Let me introduce our guest, Robin" without explicitly '
                'referencing the preceding voice, kind="introduction", track=null, target_turn=null. '
                'Use JSON null, never the string "null". An introduction never names the teacher who says it. '
                'Ordinary mentions, quotations, questions, negatives and adjectives are not introductions. '
                'In particular "I am shy" is not a name declaration; "My name is Shy" is. '
                'Never invent IDs or names. Observed data:\n'+json.dumps(context, ensure_ascii=False))

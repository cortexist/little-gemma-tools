"""Frozen v2.0 acoustic input contract and request-local alignment helpers.

Nothing here decides whether an ASR transcript is speech. Scores are raw,
uncalibrated class logits. A missing class is not a negative observation.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {r['display_name']: int(r['index']) for r in
          csv.DictReader((ROOT / 'audioset_label.csv').open())}

SOUND_POLICY = (
    ' SOUND_EVENT is private acoustic metadata attached to this input, not spoken words. '
    't is the observed audio span in seconds relative to this utterance or standalone event; '
    'c lists uncertain AudioSet labels with raw logits, not probabilities. '
    'Missing labels and low Speech scores never disprove speech, acknowledgments or a new voice. '
    'Use admitted ASR words and conversation context even when noise dominates. '
    'Labels never establish a person, name, addressee or an answer. '
    'ROOM_SOUND marks a standalone sound without admitted words or an assigned speaker; '
    'it does not advance the conversational partner or finish a pending question. '
    'ASR_REVISION replaces earlier text for the same utterance; do not count it as another speaker. '
    'A sound during an unfinished request does not finish it. '
    'Do not read metadata aloud. [wait] means the admitted request is incomplete, not a silence timer.'
)


def validate(record):
    if record['schema'] != 'sound-event/1':
        raise ValueError('unknown acoustic schema')
    a, b = record['span_seconds']
    if not all(math.isfinite(x) for x in (a, b)) or not 0 <= a < b <= record['audio_available_seconds']:
        raise ValueError('acoustic span includes unavailable/padded audio')
    if not 0 < record['nominal_window_seconds'] <= 10:
        raise ValueError('unsupported acoustic window')
    candidates = record['candidates']
    if not 1 <= len(candidates) <= 2:
        raise ValueError('compact profile requires one or two candidates')
    if len({c['label'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate')
    for c in candidates:
        if c['label'] not in LABELS or c['class_index'] != LABELS[c['label']] or not math.isfinite(c['logit']):
            raise ValueError('invalid label, index or score')
    if candidates != sorted(candidates, key=lambda c: -c['logit']):
        raise ValueError('candidates must be in score order')
    if record['score_semantics'] != 'raw_uncalibrated_logits':
        raise ValueError('scores must not be called probabilities')


def compact(record):
    validate(record)
    return ' SOUND_EVENT ' + json.dumps(
        {'t': record['span_seconds'],
         'c': [[c['label'], round(c['logit'], 1)] for c in record['candidates']]},
        separators=(',', ':'), ensure_ascii=False)


INPUT_FORMAT = 'trio_sound_event_v2_staged_history'
CAPTURE_BOUNDARY = 'post-output-norm final prompt token; before generated-token forward'


def tag_policy(policy):
    return policy.replace(
        'or [defer] if another person should answer.',
        'or [defer] if another person should answer, or [wait] if the admitted request is unfinished.'
    ).replace('After [defer], emit no spoken words.',
              'After [defer] or [wait], emit no spoken words.') + SOUND_POLICY


def speech_suffix(suffix, record=None):
    return (compact(record) if record is not None else '') + suffix.replace(
        '[speak] or [defer]', '[speak], [defer] or [wait]')


def standalone_prompt(record, spoken_turns, previous_partner_is_me):
    return compact(record) + ' ROOM_SOUND ' + json.dumps(
        {'spoken_turns': spoken_turns, 'previous_partner_is_me': previous_partner_is_me},
        separators=(',', ':')) + ' Decide [speak], [defer] or [wait].'


def align(record, request):
    """Move an ASR-window observation to its utterance-relative actual span.

    Partial text remains immutable. These observations are neither transcript
    replacements nor ASR_REVISION events. Window trimming must not restart time.
    """
    validate(record)
    duration = len(request['pcm']) / 32000
    if record['audio_available_seconds'] != duration:
        raise ValueError('Sound observation does not match this ASR request')
    base = request['base'] / 16000
    result = dict(record, span_seconds=[round(base + t, 5) for t in record['span_seconds']],
                  audio_available_seconds=round(request['end'] / 16000, 5))
    validate(result)
    return result

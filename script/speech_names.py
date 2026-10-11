"""Local name pronunciation and narrow ASR spelling hints, separate from identity."""
import re

CHEY_IPA = 'ʃˈaɪ'


def piper_names(text, pronunciations=None):
    """Replace exact written names only at the final Piper input boundary.

    Existing Piper [[phoneme]] blocks and control spans are left intact. This
    function must not modify the displayed transcript or the model's history.
    """
    mapping = {'Chey': CHEY_IPA} if pronunciations is None else pronunciations
    if not mapping:
        return text
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str)
           or not v or any(c in v for c in '[]\r\n') for k, v in mapping.items()):
        raise ValueError('Invalid name pronunciation mapping')
    pattern = re.compile(r'(?<!\w)('+ '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) +r')(?!\w)', re.I)
    lookup = {k.casefold(): v for k, v in mapping.items()}
    parts = re.split(r'(\[\[.*?\]\]|<\|tool_call>.*?<tool_call\|>)', text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(lambda m: '[['+lookup[m[0].casefold()]+']]', parts[i])
    return ''.join(parts)


def canonical_name(declared, evidence):
    """Resolve the configured name homophone only after an explicit declaration."""
    # Scenario-specific spelling lexicon, not evidence of who owns a voice.
    # base.en produced Shai on the dry Chey introduction. Biasing with Chey
    # also turned the adjective "shy" into Chey: all three need a name cue.
    if declared.casefold() in ('shy', 'shai', 'chey'):
        if not re.search(r'\b(?:my name is|call me)\s+'+re.escape(declared)+r'\b', evidence, re.I):
            raise ValueError('Chey/Shy/Shai needs an explicit name declaration or spelling clarification')
        return 'Chey'
    return declared


def whisper_name_hint(known_names):
    """Vocabulary only; callers pass names known to this listener, not a roster."""
    names = []
    for value in known_names:
        if not isinstance(value, str):
            continue
        name = value.strip()
        if (name and re.fullmatch(r"[A-Za-z0-9 .'-]+", name)
                and name.casefold() not in {item.casefold() for item in names}):
            candidate = names + [name]
            if len('Names mentioned in this conversation: '+', '.join(candidate)+'.') >= 192:
                break
            names = candidate
    if not names:
        return ''
    return 'Names mentioned in this conversation: '+', '.join(names)+'.'

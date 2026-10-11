"""Incremental Gemma framing and spoken clauses; no newline buffering."""
import codecs
import re
from collections import deque

from speech_titles import title_continues


class ReplyStream:
    def __init__(self, room=False, *, decision_tags=False, on_decision=None):
        if decision_tags and not room:
            raise ValueError('Decision tags require room mode')
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.tag = ''
        self.hidden = None
        self.bracket = ''
        self.pending = ''
        self.text = ''
        self.done = False
        self.size = 0
        self.room = room
        self.deferred = False
        self.metadata = []
        self.name_buffer = None
        self.decision_tags = decision_tags
        self.on_decision = on_decision
        self.decision = None
        self.decision_source = None
        self.clause_visuals = deque()
        self.pending_visuals = []

    def note_decision(self, action, source):
        self.decision, self.decision_source = action, source
        if self.on_decision:
            self.on_decision(action, source, len(self.metadata))

    def feed(self, data):
        self.size += len(data)
        if self.size > 128000:
            raise RuntimeError('Model reply exceeded the playback limit')
        clauses = []

        def flush():
            # Formatting belongs on screen, never spoken as "asterisk".
            value = ' '.join(re.sub(r'[*`#]', '', self.pending).split())
            self.pending = ''
            if value:
                clauses.append(value)
                self.clause_visuals.append(self.pending_visuals)
                self.pending_visuals = []

        def add(char):
            if self.room and self.deferred and not char.isspace():
                raise ValueError('A deferred response cannot also contain speech')
            if self.room and self.decision is None and not char.isspace():
                self.note_decision('speak', 'implicit_speech')
            if (self.pending and self.pending[-1] in ',;:.!?'
                    and char in ' \t\n\r*)"\']'
                    and not title_continues(self.pending, char)):
                flush()
            self.pending += char
            self.text += char

        for char in self.decoder.decode(data):
            if self.done:
                continue  # The client's trailing newline is framing, not speech.
            if self.tag:
                self.tag += char
                if char == '>':
                    tag, self.tag = self.tag, ''
                    if tag in ('<turn|>', '<eos>'):
                        if self.name_buffer is not None:
                            raise ValueError('Incomplete speaker name metadata')
                        self.done = True
                        flush()
                    elif self.hidden:
                        if tag == self.hidden:
                            self.hidden = None
                    elif tag == '<|channel>':
                        self.hidden = '<channel|>'
                    elif tag == '<|tool_call>':
                        self.hidden = '<tool_call|>'
                    elif self.room and tag == '<speaker_name>':
                        if self.name_buffer is not None:
                            raise ValueError('Nested speaker name metadata')
                        flush()
                        self.name_buffer = ''
                    elif self.room and tag == '</speaker_name>':
                        if self.name_buffer is None:
                            raise ValueError('Unexpected speaker name metadata end')
                        self.metadata.append(self.name_buffer)
                        self.name_buffer = None
                elif len(self.tag) > 32:
                    tag, self.tag = self.tag, ''
                    if not self.hidden:
                        for c in tag: add(c)
                continue
            if char == '<':
                self.tag = char
                continue
            if self.hidden:
                continue
            if self.name_buffer is not None:
                self.name_buffer += char
                if len(self.name_buffer) > 2048:
                    raise ValueError('Speaker name metadata exceeded its bound')
                continue
            if self.bracket:
                self.bracket += char
                if (char == ']' and (not self.bracket.startswith('[[') or self.bracket.endswith(']]'))
                        or len(self.bracket) > 80):
                    value, self.bracket = self.bracket, ''
                    if self.room and value.casefold() == '[defer]':
                        if self.text.strip() or self.decision is not None:
                            raise ValueError('Defer must be the entire spoken response')
                        self.deferred = True
                        self.note_decision('defer', 'explicit_tag')
                        continue
                    if self.decision_tags and value.casefold() == '[speak]':
                        if self.decision is not None or self.text.strip():
                            raise ValueError('Decision tag must occur once before speech')
                        self.note_decision('speak', 'explicit_tag')
                        continue
                    visual = re.fullmatch(r'\[(happy|sad|angry|neutral|nod|shake|quiet)\]', value)
                    if visual:
                        if self.room and self.deferred:
                            raise ValueError('A deferred response cannot contain facial cues')
                        if self.decision_tags and self.decision is None:
                            raise ValueError('Facial cues must follow the decision tag')
                        cue = visual.group(1)
                        kind = 'gesture' if cue in ('nod', 'shake', 'quiet') else 'expression'
                        self.pending_visuals.append((kind, cue))
                    elif not re.fullmatch(r'\[\[emotion:[^]]+\]\]', value):
                        for c in value: add(c)
                continue
            if char == '[':
                self.bracket = char
                continue
            add(char)
        return clauses

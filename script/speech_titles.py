"""English personal titles: preserve transcript text, normalize only TTS input."""
import re

_TITLES = {'mr': 'Mr', 'mrs': 'Mrs', 'ms': 'Miz', 'dr': 'Doctor',
           'prof': 'Professor', 'rev': 'Reverend'}
_TITLE = re.compile(r"(?<![A-Za-z0-9_\x80-\U0010ffff])(?:mr|mrs|ms|dr|prof|rev)\.", re.IGNORECASE)


def title_continues(text, next_char):
    return next_char in ' \t\r\n\f\v' and any(m.end() == len(text) for m in _TITLE.finditer(text))


def normalize_titles(text):
    def replace(match):
        tail = text[match.end():]
        # Same ASCII whitespace policy as the C streaming splitter.
        if tail and tail[0] in ' \t\r\n\f\v':
            name = tail.lstrip(' \t\r\n\f\v')
            if name and (name[0].isascii() and name[0].isalpha() or ord(name[0]) >= 128):
                return _TITLES[match[0][:-1].lower()]
        return match[0]
    return _TITLE.sub(replace, text)

"""Apply pronunciation spelling only to Piper's spoken stdin, not model history."""
import json
import os
import sys

from speech_names import piper_names


def spoken_lines(lines, names):
    for line in lines:
        yield piper_names(line, names)


if __name__ == '__main__':
    from piper.__main__ import main
    sys.stdin = spoken_lines(sys.stdin, json.loads(os.environ.get(
        'VOICE_NAME_PRONUNCIATIONS', '{"Chey":"ʃˈaɪ"}')))
    main()

"""Exercise both real C clause parsers, including byte-fragmented reply streams."""
import importlib.util
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('speech_titles', ROOT/'script/speech_titles.py')
py = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py)

CASES = [
    ('Mrs. Byers.\n', ['Mrs Byers.']),
    ('Why is that, Mrs. Byers?\n', ['Why is that,', 'Mrs Byers?']),
    ('Mrs. Johnson, may I ask? Yes.\n', ['Mrs Johnson,', 'may I ask?', 'Yes.']),
    ('Mr. Kai. Dr. Smith. Prof. Jones. Rev. Green. Ms. Lee.\n',
     ['Mr Kai.', 'Doctor Smith.', 'Professor Jones.', 'Reverend Green.', 'Miz Lee.']),
    ('mrs. byers agrees. MR. Kai agrees.\n', ['Mrs byers agrees.', 'Mr Kai agrees.']),
    ('Yes. Mrs. Byers knows.\n', ['Yes.', 'Mrs Byers knows.']),
    ('Mrs.\nMr.\n', ['Mrs.', 'Mr.']),
    ('3.14 is pi. Next: stop! Really?\n', ['3.14 is pi.', 'Next:', 'stop!', 'Really?']),
    ('NotMrs. Next. code_Mrs. Next. 1Mrs. Next.\n',
     ['NotMrs.', 'Next.', 'code_Mrs.', 'Next.', '1Mrs.', 'Next.']),
    ('Mrs.\tByers and Dr. García.\n', ['Mrs\tByers and Doctor García.']),
    ('<|channel>Mrs. hidden<channel|>Mrs. Byers.<turn|>\n', ['Mrs Byers.']),
]

class SpeechTitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != 'posix' or not shutil.which('cc'):
            raise unittest.SkipTest('C parser checks require POSIX and cc')
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        # Include the real voicecat parser, redirecting only its TTS descriptor.
        source = root/'mouth.c'
        source.write_text('#define main voicecat_program_main\n#include "voicecat.c"\n#undef main\n'
                          'int main(int argc, char **argv) {\n'
                          'g_mouth_synth="test"; m_synth_in=1; char buf[4096];\n'
                          'size_t step=argc>1?(size_t)atoi(argv[1]):1, n;\n'
                          'while ((n=fread(buf,1,step,stdin))) mouth_feed(buf,(int)n);\n'
                          'm_flush_line(); return 0; }\n')
        cls.bins = [root/'clausecat', root/'mouth']
        for src, binary in [(ROOT/'src/clausecat.c', cls.bins[0]), (source, cls.bins[1])]:
            subprocess.run(['cc', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-D_DEFAULT_SOURCE',
                            '-I'+str(ROOT/'src'), '-I'+str(ROOT/'proto'), str(src), '-lm', '-o', str(binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_streams_and_fragmented_titles(self):
        for text, expected in CASES:
            for binary in self.bins:
                for size in ([1, 2, 7, 4096] if binary.name == 'mouth' else [1]):
                    with self.subTest(text=text, parser=binary.name, chunk=size):
                        args = [str(binary)] + ([str(size)] if binary.name == 'mouth' else [])
                        out = subprocess.check_output(args, input=text.encode()).decode().splitlines()
                        self.assertEqual(out, expected)

    def test_normal_sentences_flush_without_waiting_for_more_text(self):
        for binary in self.bins:
            with subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
                proc.stdin.write(b'Yes. '); proc.stdin.flush()
                self.assertTrue(select.select([proc.stdout], [], [], 2)[0])
                self.assertEqual(proc.stdout.readline(), b'Yes.\n')
                proc.stdin.write(b'Mrs. '); proc.stdin.flush()
                self.assertFalse(select.select([proc.stdout], [], [], .05)[0])
                proc.stdin.write(b'Byers.\n'); proc.stdin.flush()
                self.assertEqual(proc.stdout.readline(), b'Mrs Byers.\n')
                proc.stdin.close(); proc.wait(timeout=2)

    def test_python_normalization_keeps_non_titles_and_terminal_periods(self):
        for text, expected in CASES:
            if '<' in text: continue
            parts = []
            line = ''
            for c in text:
                if c == '\n':
                    if line.strip(): parts.append(py.normalize_titles(line.strip()))
                    line = ''
                    continue
                if line and line[-1] in ',;:.!?' and c in ' \t\r\f\v*)"\']' and not py.title_continues(line, c):
                    if line.strip(): parts.append(py.normalize_titles(line.strip()))
                    line = ''
                line += c
            self.assertEqual(parts, expected)

if __name__ == '__main__': unittest.main()

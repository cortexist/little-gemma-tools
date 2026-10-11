"""Compile the real audio-tap sender and verify output routing/backpressure."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PlaybackTapTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'PulseAudio sender is Linux-only')
    def test_recording_backpressure_never_stalls_or_silently_drops_audio(self):
        if not shutil.which('cc') or not shutil.which('pkg-config'):
            self.skipTest('C compiler/pkg-config unavailable')
        flags = subprocess.run(['pkg-config', '--cflags', '--libs', 'libpulse-simple'], capture_output=True, text=True)
        if flags.returncode: self.skipTest('libpulse-simple development package unavailable')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory)/'playback-tap-test')
            subprocess.run(['cc', '-O2', '-I'+str(root/'proto'), '-I'+str(root/'src'),
                str(root/'tests/playback_tap.c'), str(root/'src/aec-null.c'), '-o', binary,
                *flags.stdout.split(), '-lm', '-lpthread'], check=True, capture_output=True)
            subprocess.run([binary], check=True, timeout=5, capture_output=True)

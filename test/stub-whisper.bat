@echo off
rem Windows twin of stub-whisper.sh: voicecat spawns its ASR through _popen ->
rem cmd.exe, which cannot exec a .sh - forward to Git Bash. Use with
rem   --whisper-bin test\stub-whisper.bat --whisper-model dummy
"C:\Program Files\Git\bin\bash.exe" "%~dp0stub-whisper.sh" %*

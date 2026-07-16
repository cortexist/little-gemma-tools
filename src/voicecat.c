// voicecat — a live voice turn-taker for little-gemma's -s socket. mmcat's
// sibling: mmcat is files-in-one-turn, voicecat is an open microphone and an
// endless conversation.
//
//     voicecat <socket> [--mic FFMPEG_INPUT | --stdin-pcm] [flags]
//
// It listens (energy VAD), and for each utterance either:
//   - streams a whisper transcript INTO THE OPEN TURN while the user is still
//     talking (--whisper-model): whisper runs over the growing utterance every
//     --commit-ms, and words two consecutive passes agree on (LocalAgreement-2)
//     are committed as 'T' frames — the runner's turn text is appendable by
//     design, so by end-of-speech only the unstable tail is left to send with
//     the closing line. Post-utterance latency is one whisper pass over the
//     last chunk, not the whole clip;
//   - or, with no whisper model, sends the whole utterance as ONE native audio
//     span at end-of-speech. The model's own ears work ONLY in vision-free
//     sessions (with any frames in context the 12B cannot hear — measured);
//     whisper mode is the one that composes with a camera.
//
// BARGE-IN: the mic stays open while the reply streams. If the user starts
// talking over it, voicecat sends the one-byte LG_BARGE signal — the runner
// stops decoding at the next token, closes the turn on the wire, and the
// session (cut-off reply included, so the model knows what it didn't say)
// continues. The interrupted state is remembered HERE, not in the runner: the
// next turn's text opens with --barge-note (default "(interrupting) ").
//
// Turn-taking assumes the mic doesn't hear the speakers (headset, or AEC
// upstream); an open-air mic will barge on its own TTS. Whisper must run
// faster than real time (tiny/base on a GPU) or commits fall behind capture.
// v1 shells out to ffmpeg (capture) and whisper-cli over pipes, like mmcat.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <afunix.h>
#include <io.h>
#include <fcntl.h>
typedef SOCKET sock_t;
#define sock_close closesocket
#define SHUT_WR SD_SEND
#define popen  _popen
#define pclose _pclose
static int  sock_init(void) { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w); }
static void msleep(int ms) { Sleep(ms); }
static int  wouldblock(void) { return WSAGetLastError() == WSAEWOULDBLOCK; }
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
static int  sock_init(void) { signal(SIGPIPE, SIG_IGN); return 0; }
static void msleep(int ms) { usleep(ms * 1000); }
static int  wouldblock(void) { return errno == EAGAIN || errno == EWOULDBLOCK; }
#endif

#include "lg_media_proto.h"

#define FR_SAMP 320                      // one mic tick: 20 ms @ 16 kHz
#define FR_MS   20
#define PREROLL 16                       // ticks kept before the voice onset (320 ms)
#define ONSET   3                        // consecutive voiced ticks that open an utterance (60 ms)

static const char *g_whisper_bin = NULL, *g_whisper_model = NULL;
static const char *g_whisper_url = NULL;   // resident whisper-server: no model load per pass
static const char *g_barge_note = "(interrupting) ";
// Commit cadence must comfortably exceed one ASR invocation (~0.6s for CUDA
// base.en via whisper-cli, process+model load included) or passes stack up
// and the loop falls behind real time: keep commit_ms >= 3x the pass cost.
static int g_commit_ms = 2500, g_hang_ms = 700, g_vad = 400, g_realtime = 0;
// While a reply is live (pending, or the mouth still sounding) the bar to
// open an utterance is HIGHER and LONGER — the browser demo's BARGE_THRESH
// rule: echo residuals and AEC startup transients must not cut the mouth.
static int g_barge_mult = 2, g_barge_onset = 6;
// --duck-sock: TWO-STAGE barge via far-field-service's control channel.
// Stage 1, at onset: the reply DUCKS (drops --duck-db, keeps playing) —
// instant, and cheap to undo. Stage 2, when words MATERIALIZE (first
// streamed commit, or a non-empty final transcript): the real cut + the
// barge turn. A door slam or a cough ducks the voice for a second and
// then it swells back — never a turn, never a dangling half-reply.
static const char *g_duck_sock = NULL;

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// ---- the listener (--listener) ------------------------------------------------
// While the user is STILL SPEAKING, ask the model what it would do if the turn
// ended right here: a 'P' probe frame rides after transcript commits and on
// mid-utterance pauses (the classic backchannel-inviting spots), carrying a
// steering suffix; the runner dry-runs the turn close on the live context,
// decodes a few tokens, ROLLS THE CONTEXT BACK, and answers mid-turn with
// "<|probe>text<probe|>\n" — probes never change the conversation. The
// envelope is lifted out of the reply stream before stdout (the TTS never
// sees it); the verdict's first word becomes an action line on stderr:
//   nod    -> set_facial{"expression":"nodding"}
//   mhmm   -> say_backchannel{"text":"mhmm"}    (a canned clip's cue, not TTS)
//   answer -> logged as a listener end-point: the model believes the request
//             is complete (a non-keyword verdict usually means it simply
//             STARTED answering — the same signal, logged the same way)
//   quiet  -> nothing
// Policy — the suffix wording, the cadence, what the words mean — lives HERE
// in the client; the engine only owns the dry-run + rollback mechanism. Note
// a pause probe fires on COMMITTED text only: with per-invocation whisper the
// unconfirmed tail lags the audio by up to commit-ms, so the model judges a
// slightly stale transcript — the price of slow ears, not of the probe.
static int g_listener = 0, g_probe_ms = 1500, g_probe_gen = 6;
#define PROBE_PAUSE_MS 440               // mid-utterance pause that invites a cue
// The verdict grammar is the house [[tag]] form — the same inline-annotation
// style the orchestrator will parse out of reply streams, so the listener cues
// and the speaking-time tags become ONE grammar (one parser, one finetune
// story). The parser below still falls back to a bare first word.
static const char *g_probe_suffix =
    "\n(brief listener check - I am still mid-request, keep listening; emit "
    "exactly one backchannel tag: [[nod]] if you follow me so far, [[mhmm]] to "
    "acknowledge me, [[shake]] if I have something wrong, [[answer]] if my "
    "request is already complete, [[quiet]] otherwise)";
static double g_probe_t0 = 0;            // last probe's send time (rate limit + latency)
static int    g_probe_inflight = 0;

// ---- idle compression (--idle-compress) ----------------------------------------
// A long-lived session slowly fills the runner's context. After a long QUIET
// (no speech, no reply in flight) the session compresses itself: ask the model
// for a digest of the conversation (consumed here — never spoken), reopen a
// FRESH session (the runner re-seats its -sys prefix from saved rows
// instantly), and seat the digest as the open turn's first 'T' text — the
// engine prefills it on arrival, during the idle, so the user's next words
// simply continue a conversation the model still remembers, at a fraction of
// the context positions. This is the beyond-voice REHYDRATION mechanism with
// an idle trigger; the policy (the ask, the seed wording, the timing) lives
// HERE. What a digest cannot keep: verbatim wording, and raw media spans (a
// camera-locked session comes back with its ears, per the A/V exclusion).
static int g_idle_compress = 900;        // seconds of quiet that trigger it (0 = off)
static const char *g_compress_ask =
    "(session maintenance) Summarize our conversation so far in under 120 "
    "words - the facts, names, preferences, any unfinished business, and a "
    "one-line description of anything you were shown or heard beyond my "
    "words. Anchor events to their clock times where that matters. When a "
    "fact changed during our talk, state it as a dated transition - old to "
    "new, never only the old state. Reply with only the summary.";
// --memory FILE: the dated LEDGER — every compress appends one line,
// "[Tue 2026-07-09 14:50] <digest>", and a fresh voicecat seeds its first
// turn from the ledger's tail, so memory survives process restarts and every
// entry is born knowing its date.
//
// CONSOLIDATION (runs at startup and after each compress, over throwaway
// serve sessions): the ledger ages the way human memory does — raw dated
// digests older than a week fold into "[week of YYYY-MM-DD]" lines, week
// lines older than a month into "[YYYY-MM]", months older than a year into
// "[YYYY]" one-liners under the flashbulb ask (most words on the defining
// event, its small details kept verbatim). Graphiti's bi-temporal model,
// translated to plain text: validity lives in the WORDING (changes are
// dated transitions, old to new — never only the old state), contradiction
// resolves by the seed's newest-wins rule instead of edge invalidation, and
// nothing is ever lost except through a successful, budgeted merge (a
// failed ask copies the originals through untouched).
static const char *g_memory = NULL;
static const char *g_merge_week =
    "(memory maintenance) Merge these notes from one week into a single line "
    "of at most 18 words. Keep names, dates, unfinished business, and any "
    "change as a dated transition - old to new, never only the old state. "
    "Drop the routine. Reply with only the line.";
static const char *g_merge_month =
    "(memory maintenance) Merge these notes from one month into a single "
    "line of at most 14 words. Keep only what will still matter in a year; "
    "keep changes as dated transitions. Reply with only the line.";
static const char *g_merge_year =
    "(memory maintenance) Merge these notes from one year into one short "
    "passage of at most 35 words. Spend nearly all of it on the single most "
    "defining event, keeping its small details - the little things. "
    "Everything else gets a few words at most, or nothing at all. "
    "Reply with only the passage.";

// ---- time awareness (--clock) ---------------------------------------------------
// GPT-Live-style temporal awareness as plain turn text — the engine never
// knows. Every turn opens with the real clock, "[14:32] " (weekday on the
// session's first), and once the quiet since the last exchange passes the
// threshold, the client does the arithmetic the model is bad at and appends
// "(27 minutes of quiet pass) ". MEASURED on E4B: a plain bracket reads back
// correctly when asked the time but carries no sense of gap length; the gap
// parenthetical fixes elapsed judgments and return-greetings but corrupts an
// absolute read in that same turn (the model computes instead of reading) —
// hence plain always, annotation only past the threshold, both tunable.
// docs/voice-sys.txt teaches the model to use the clock silently.
static int    g_clock = 300;             // gap-annotation threshold, s (0 = no markers)
static int    g_first_turn = 1;
static double g_last_close = 0;          // when the previous exchange ended

static const char *turn_clock(void) {
    static char buf[96];
    if (g_clock <= 0) return "";
    time_t t = time(NULL);
    size_t n = strftime(buf, sizeof buf,                 // the first marker carries the
                        g_first_turn ? "[%a %Y-%m-%d %H:%M] " : "[%H:%M] ",   // date, so "Friday"
                        localtime(&t));                  // can be stored as an absolute day
    double gap = g_last_close > 0 ? now_sec() - g_last_close : 0;
    if (!g_first_turn && gap >= (double)g_clock) {
        int mins = (int)(gap / 60.0 + 0.5);
        if (mins < 1) mins = 1;
        if (mins >= 60)
            snprintf(buf + n, sizeof buf - n, "(%d hour%s and %d minute%s of quiet pass) ",
                     mins / 60, mins / 60 == 1 ? "" : "s", mins % 60, mins % 60 == 1 ? "" : "s");
        else
            snprintf(buf + n, sizeof buf - n, "(%d minute%s of quiet pass) ", mins, mins == 1 ? "" : "s");
    }
    g_first_turn = 0;
    return buf;
}

// ---- the wire (mmcat's idioms; the socket here is NON-BLOCKING) --------------
static int send_all(sock_t s, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        int k = (int)send(s, p, (int)(n > (1 << 20) ? (1 << 20) : n), 0);
        if (k <= 0) {
            if (k < 0 && wouldblock()) { msleep(1); continue; }
            return -1;
        }
        p += k; n -= (size_t)k;
    }
    return 0;
}
static int send_frame(sock_t s, uint8_t type, const void *payload, uint32_t len) {
    uint8_t hdr[LG_FRAME_HDR] = { LG_FRAME_MAGIC, type };   // w = h = 0 ('A'/'T' only here)
    memcpy(hdr + 6, &len, 4);
    if (send_all(s, hdr, sizeof hdr) != 0) return -1;
    return send_all(s, payload, len);
}
static FILE *run_pipe(const char *cmd) {
#ifdef _WIN32
    return popen(cmd, "rb");
#else
    return popen(cmd, "r");
#endif
}

// ---- whisper over a temp wav (mmcat's pattern), words only -------------------
static void temp_wav_path(char *buf, size_t n) {
#ifdef _WIN32
    char dir[MAX_PATH]; DWORD d = GetTempPathA((DWORD)sizeof dir, dir);
    if (!d || d > sizeof dir) strcpy(dir, ".\\");
    snprintf(buf, n, "%svoicecat_%lu.wav", dir, (unsigned long)GetCurrentProcessId());
#else
    const char *t = getenv("TMPDIR"); if (!t || !*t) t = "/tmp";
    snprintf(buf, n, "%s/voicecat_%ld.wav", t, (long)getpid());
#endif
}
static int write_wav(const char *path, const int16_t *pcm, size_t nsamp) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t datalen = (uint32_t)(nsamp * 2), srate = LG_RATE, byterate = srate * 2, riff = 36 + datalen, fmtlen = 16;
    uint16_t fmt = 1, ch = 1, ba = 2, bps = 16;
    uint8_t h[44];
    memcpy(h, "RIFF", 4);      memcpy(h + 4, &riff, 4);   memcpy(h + 8, "WAVE", 4);
    memcpy(h + 12, "fmt ", 4); memcpy(h + 16, &fmtlen, 4);
    memcpy(h + 20, &fmt, 2);   memcpy(h + 22, &ch, 2);    memcpy(h + 24, &srate, 4);
    memcpy(h + 28, &byterate, 4); memcpy(h + 32, &ba, 2); memcpy(h + 34, &bps, 2);
    memcpy(h + 36, "data", 4); memcpy(h + 40, &datalen, 4);
    int ok = fwrite(h, 1, 44, f) == 44 && fwrite(pcm, 2, nsamp, f) == nsamp;
    fclose(f);
    return ok ? 0 : -1;
}

// One whisper pass over the audio window, WITH segment timestamps and prior-
// text context. The timestamps are what make streaming real: a segment whose
// words are all confirmed can be TRIMMED from the window afterwards, so pass
// cost tracks the unconfirmed tail, not the whole utterance — and `prompt`
// (the trimmed text) keeps whisper's continuation coherent across the cut.
// whisper-cli's default output is one "[hh:mm:ss.mmm --> hh:mm:ss.mmm] text"
// line per segment; output without headers falls back to one whole-window
// segment. Collects tag-stripped, space-normalized words in `out`, plus per
// segment its end (samples) and the CUMULATIVE word count.
#define WSEG_MAX 64
struct wseg { size_t end_samp; int words; };
static size_t parse_ts(const char *p) {                  // "hh:mm:ss.mmm" -> samples
    int h = 0, m = 0; double sec = 0;
    if (sscanf(p, "%d:%d:%lf", &h, &m, &sec) != 3) return 0;
    return (size_t)(((h * 60 + m) * 60 + sec) * LG_RATE);
}

// Append one transcript line/segment to `out`: [..] (..) tags stripped whole,
// whitespace folded, a trailing space closes the last word. `w` = bytes in
// out, `words` = completed words — both run cumulatively across calls.
static void words_add(const char *text, char *out, size_t *w, size_t cap, int *words) {
    for (size_t i = 0; text[i]; ) {
        char c = text[i];
        if (c == '[' || c == '(') {
            char close = c == '[' ? ']' : ')';
            while (text[i] && text[i] != close) i++;
            if (text[i]) i++;
            continue;
        }
        char cc = (c == '\r' || c == '\t' || c == '\n') ? ' ' : c;
        if (cc == ' ' && (*w == 0 || out[*w - 1] == ' ')) { i++; continue; }
        if (*w + 1 < cap) { out[(*w)++] = cc; out[*w] = 0; }
        if (cc == ' ') (*words)++;
        i++;
    }
    if (*w && out[*w - 1] != ' ' && *w + 1 < cap) { out[(*w)++] = ' '; out[*w] = 0; (*words)++; }
}

// whisper-server's vtt: "WEBVTT", then cues of "t0 --> t1" on one line and
// the text on the following line(s) — the cue's end closes the segment when
// the NEXT cue begins (or the input ends). Fills out/segs like the cli parse.
static void whisper_vtt(char *raw, char *out, size_t cap, struct wseg *segs, int *nseg) {
    size_t w = 0, pend = 0;
    int words = 0, seg_open = 0;
    for (char *line = raw; line && *line; ) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *arrow = strstr(line, "-->");
        if (arrow) {
            if (seg_open && *nseg < WSEG_MAX) { segs[*nseg].end_samp = pend; segs[*nseg].words = words; (*nseg)++; }
            pend = parse_ts(arrow + 3 + strspn(arrow + 3, " "));
            seg_open = 1;
        } else if (seg_open && *line)
            words_add(line, out, &w, cap, &words);
        line = nl ? nl + 1 : NULL;
    }
    if (seg_open && *nseg < WSEG_MAX) { segs[*nseg].end_samp = pend; segs[*nseg].words = words; (*nseg)++; }
}
static int whisper_pass(const int16_t *pcm, size_t nsamp, const char *prompt,
                        char *out, size_t cap, struct wseg *segs, int *nseg) {
    if (cap) out[0] = 0;
    *nseg = 0;
    char wav[1024]; temp_wav_path(wav, sizeof wav);
    if (write_wav(wav, pcm, nsamp) != 0) { fprintf(stderr, "voicecat: temp wav write failed (%s)\n", wav); return -1; }
    const char *bin = g_whisper_bin ? g_whisper_bin : "whisper-cli";
    char parg[512] = "";
    if (prompt && *prompt) {
        size_t k = 0;
        for (const char *p = prompt; *p && k < sizeof parg - 32; p++)
            if (*p != '"' && *p != '\\' && *p != '`' && *p != '$') parg[k++] = *p;
        parg[k] = 0;
    }
    char cmd[4096];
    // the extra outer quotes are for cmd.exe: a /c string that STARTS with a
    // quote gets its first and last quote stripped, mangling every path inside
    if (g_whisper_url)
        // resident server: the pass pays inference only, no model load —
        // vtt keeps the reply small and carries the segment timestamps
        snprintf(cmd, sizeof cmd,
#ifdef _WIN32
                 "\"curl -s --max-time 30 -F file=@\"%s\" -F response_format=vtt -F prompt=\"%s\" \"%s\" 2>NUL\"",
#else
                 "curl -s --max-time 30 -F file=@\"%s\" -F response_format=vtt -F prompt=\"%s\" \"%s\" 2>/dev/null",
#endif
                 wav, parg, g_whisper_url);
    else
        snprintf(cmd, sizeof cmd,
#ifdef _WIN32
                 "\"\"%s\" -m \"%s\" -f \"%s\" -np --prompt \"%s\" 2>NUL\"",
#else
                 "\"%s\" -m \"%s\" -f \"%s\" -np --prompt \"%s\" 2>/dev/null",
#endif
                 bin, g_whisper_model, wav, parg);
    FILE *f = run_pipe(cmd);
    if (!f) { remove(wav); fprintf(stderr, "voicecat: whisper spawn failed\n"); return -1; }
    char raw[16384]; size_t n = fread(raw, 1, sizeof raw - 1, f); raw[n] = 0;
    int rc = pclose(f);
    remove(wav);
    if (n == 0 && rc != 0) { fprintf(stderr, "voicecat: whisper failed (rc %d) — is it on PATH?\n", rc); return -1; }

    if (g_whisper_url)
        whisper_vtt(raw, out, cap, segs, nseg);
    else {
        size_t w = 0; int words = 0;
        char *line = raw;
        while (line && *line) {
            char *nl = strchr(line, '\n');
            if (nl) *nl = 0;
            char *text = line;
            size_t seg_end = 0;
            char *arrow = strstr(line, "-->");
            if (line[0] == '[' && arrow) {               // "[t0 --> t1]  text"
                seg_end = parse_ts(arrow + 3 + strspn(arrow + 3, " "));
                char *close = strchr(arrow, ']');
                text = close ? close + 1 : arrow + 3;
            }
            words_add(text, out, &w, cap, &words);
            if (seg_end && *nseg < WSEG_MAX) { segs[*nseg].end_samp = seg_end; segs[*nseg].words = words; (*nseg)++; }
            line = nl ? nl + 1 : NULL;
        }
    }
    size_t w = strlen(out);
    while (w && out[w - 1] == ' ') out[--w] = 0;
    if (*nseg == 0 && w) {                               // no cues: one whole-window segment
        int words = 1;
        for (size_t i = 0; out[i]; i++) words += out[i] == ' ';
        segs[0].end_samp = nsamp; segs[0].words = words; *nseg = 1;
    }
    return 0;
}

// Incremental "<turn|>" matcher: feed reply bytes as they arrive, get the count
// of completed turns (handles the terminator split across reads, and two turn
// ends in one read — which barge-in produces back to back).
static int turn_ends(const char *data, int n, int *state) {
    static const char T[] = "<turn|>";
    int hits = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] == T[*state]) { if (!T[++*state]) { hits++; *state = 0; } }
        else *state = data[i] == T[0] ? 1 : 0;
    }
    return hits;
}

static void send_probe(sock_t s) {
    uint8_t hdr[LG_FRAME_HDR] = { LG_FRAME_MAGIC, LG_FRAME_PROBE };
    uint16_t mg = (uint16_t)g_probe_gen;
    uint32_t len = (uint32_t)strlen(g_probe_suffix);
    memcpy(hdr + 2, &mg, 2);
    memcpy(hdr + 6, &len, 4);
    if (send_all(s, hdr, sizeof hdr) == 0 && send_all(s, g_probe_suffix, len) == 0) {
        g_probe_inflight = 1;
        g_probe_t0 = now_sec();
    }
}

// A completed probe verdict: take the [[tag]]'s word (or the bare first word
// when no tag came back), classify, act on stderr (stdout belongs to the
// reply/TTS).
static void listener_verdict(const char *v) {
    double lat = g_probe_t0 ? now_sec() - g_probe_t0 : 0;
    g_probe_inflight = 0;
    char w[32];
    int n = 0;
    const char *p = strstr(v, "[[");
    for (p = p ? p + 2 : v; *p; p++) {                   // tag / first word, lowercased
        char c = *p;
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        if ((c >= 'a' && c <= 'z') || c == '-') { if (n < (int)sizeof w - 1) w[n++] = c; }
        else if (n) break;
    }
    w[n] = 0;
    int nod  = !strcmp(w, "nod") || !strcmp(w, "nodding");
    int shake = !strcmp(w, "shake") || !strcmp(w, "no");
    int mhmm = !strcmp(w, "mhmm") || !strcmp(w, "mm-hmm") || !strcmp(w, "mmhmm") ||
               !strcmp(w, "uh-huh") || !strcmp(w, "hmm") || !strcmp(w, "hm");
    int quiet = !v[0] || !strcmp(w, "quiet") || !strcmp(w, "none") ||
                !strcmp(w, "nothing") || !strcmp(w, "wait") || !strcmp(w, "silence");
    if (nod)   fprintf(stderr, "set_facial{\"expression\":\"nodding\"}\n");
    if (shake) fprintf(stderr, "set_facial{\"expression\":\"shaking\"}\n");
    if (mhmm)  fprintf(stderr, "say_backchannel{\"text\":\"mhmm\"}\n");
    if (!nod && !shake && !mhmm && !quiet)               // 'answer', or it started answering
        fprintf(stderr, "voicecat: listener end-point (the model would answer now)\n");
    fprintf(stderr, "voicecat: probe %.2fs -> '%s'\n", lat, v);
}

// Incremental "<|probe>verdict<probe|>\n" extractor: envelope bytes never
// reach `out`, completed verdicts go to listener_verdict. A held partial
// marker carries across reads; neither marker contains '<' past its first
// byte, so flush-then-rematch-current is exact.
static struct { int m, cap, cm, vn; char v[512]; } g_pf;
static int pf_feed(const char *in, int n, char *out) {
    static const char PO[] = "<|probe>", PC[] = "<probe|>\n";
    int on = 0;
    for (int i = 0; i < n; i++) {
        char c = in[i];
        if (!g_pf.cap) {
            if (c == PO[g_pf.m]) {
                if (!PO[++g_pf.m]) { g_pf.m = 0; g_pf.cap = 1; g_pf.cm = 0; g_pf.vn = 0; }
            } else {
                for (int j = 0; j < g_pf.m; j++) out[on++] = PO[j];
                g_pf.m = c == PO[0] ? 1 : 0;
                if (!g_pf.m) out[on++] = c;
            }
        } else {
            if (c == PC[g_pf.cm]) {
                if (!PC[++g_pf.cm]) {
                    g_pf.v[g_pf.vn] = 0;
                    listener_verdict(g_pf.v);
                    g_pf.cap = 0; g_pf.cm = 0; g_pf.vn = 0;
                }
            } else {
                for (int j = 0; j < g_pf.cm && g_pf.vn < (int)sizeof g_pf.v - 1; j++) g_pf.v[g_pf.vn++] = PC[j];
                g_pf.cm = c == PC[0] ? 1 : 0;
                if (!g_pf.cm && g_pf.vn < (int)sizeof g_pf.v - 1) g_pf.v[g_pf.vn++] = c;
            }
        }
    }
    return on;
}

// ---- the mouth (--mouth-synth / --mouth-play) ----------------------------------
// voicecat OWNS the TTS when these are set: reply text runs through clausecat's
// split policy IN-PROCESS (the long-planned absorb — its trigger was exactly
// this: a shell pipeline's buffered clauses keep playing after a barge), clause
// lines feed a synth spawned ONCE (piper's cold start is ~4s on the Orin — it
// must stay warm) speaking piper's mux framing, and live PCM is pumped through
// a ring into a small, cheaply-respawned player. A barge then kills ONLY the
// player (sound stops within its latency buffer) and marks every clause sent
// so far stale — their frames drain unheard, counted exactly, no timing
// heuristics. POSIX only; the flags refuse politely on Windows.
static const char *g_mouth_synth = NULL, *g_mouth_play = NULL;
static int g_hush_tail = 0;                  // --hush-tail: cut on speech over the reply's tail
#ifndef _WIN32
static long  m_synth_pid = 0, m_play_pid = 0;
static int   m_synth_in = -1, m_synth_out = -1, m_play_in = -1;
static char *m_ring = NULL;                  // PCM waiting for the player
static size_t m_rn = 0, m_rcap = 0;
static double m_last_pcm = 0;                // last byte from the synth (drain/escape clock)
// The speaking-state is an AUDIBLE-HORIZON clock: bytes handed to the player
// divided by the stream rate say exactly when the sound RUNS OUT — pipe
// activity says nothing (the synth outruns real time ~8x, so the pump goes
// idle seconds into a long reply while half a minute still plays downstream;
// measured: barge never fired and every reply queued behind the backlog).
static double m_audible_until = 0;
static int    m_rate = 22050;                // from the synth's mux 'C' frame
// The synth speaks piper's mux framing ([kind][u32 len][payload]), which
// makes the barge discard EXACT: a cut drops a set_voice control line with
// an unresolvable sentinel value down the synth's stdin — piper consumes it
// in order, echoes it back as an 'M' frame (the audio never switches), and
// every 'P' frame BEFORE that echo is stale by construction. No timing
// heuristics: a synth mid-cold-start can't leak stale audio past a gap
// timer that isn't there. ('A' schedule frames exist only on alignment-
// enabled voices — never counted on.)
#define M_MARK "vc-cut"
static long m_marks_sent = 0, m_marks_seen = 0;
static int  m_mute_turn = 0;                 // post-barge: eat the dying reply's tail clauses
static struct { uint8_t head[5]; uint32_t hn, len, got; uint8_t kind; char meta[64]; uint32_t mn; } m_fr;

// clausecat's machine, verbatim policy (no --allow-control-token, no
// --route-emotion here yet): thought spans and <tokens> dropped, [[tags]]
// dropped whole, a clause line flushes at punctuation + space.
static struct {
    char line[8192]; size_t ln;
    char tag[27];    size_t tn;
    int thought;     size_t cn;
    int tcall;       size_t tcn;
    char tsp[64];    size_t bn;
    int brk, pb;
} m_cl;

static int spawn_cmd(const char *cmd, int *in_fd, int *out_fd) {
    int pi[2] = { -1, -1 }, po[2] = { -1, -1 };
    if ((in_fd && pipe(pi) != 0) || (out_fd && pipe(po) != 0)) return -1;
    long pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        setpgid(0, 0);                       // own group: a kill(-pid) gets the whole sh -c tree
        if (in_fd) { dup2(pi[0], 0); close(pi[0]); close(pi[1]); }
        if (out_fd) { dup2(po[1], 1); close(po[0]); close(po[1]); }
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    setpgid((pid_t)pid, (pid_t)pid);         // both sides set it: no kill-before-setpgid race
    if (in_fd)  { close(pi[0]); *in_fd = pi[1]; }
    if (out_fd) { close(po[1]); *out_fd = po[0]; fcntl(*out_fd, F_SETFL, O_NONBLOCK); }
    return (int)pid;
}

static void mouth_start(void) {
    if (!g_mouth_synth) return;
    m_synth_pid = spawn_cmd(g_mouth_synth, &m_synth_in, &m_synth_out);
    if (m_synth_pid < 0) { fprintf(stderr, "voicecat: mouth synth spawn failed\n"); g_mouth_synth = NULL; }
}

static void ring_add(const uint8_t *b, size_t n) {
    if (m_rn + n > m_rcap) {
        m_rcap = (m_rn + n) * 2 + 65536;
        m_ring = realloc(m_ring, m_rcap);
    }
    memcpy(m_ring + m_rn, b, n);
    m_rn += n;
}

// One chunk of the synth's mux stream through the frame parser: live 'P'
// payload lands in the ring, an 'M' echo of our cut marker clears one mark,
// all else ('C' config, 'A' schedules, stale 'P') falls through. Parser
// state survives a cut — the stream's framing must stay intact whatever
// the policy does.
static void mux_feed(const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; ) {
        if (m_fr.hn < 5) {
            size_t take = 5 - m_fr.hn;
            if (take > n - i) take = n - i;
            memcpy(m_fr.head + m_fr.hn, b + i, take);
            m_fr.hn += (uint32_t)take;
            i += take;
            if (m_fr.hn == 5) {
                m_fr.kind = m_fr.head[0];
                memcpy(&m_fr.len, m_fr.head + 1, 4);
                m_fr.got = 0;
                m_fr.mn = 0;
                if (m_fr.len == 0) m_fr.hn = 0;
            }
            continue;
        }
        size_t take = m_fr.len - m_fr.got;
        if (take > n - i) take = n - i;
        if (m_fr.kind == 'P' && m_marks_seen == m_marks_sent) ring_add(b + i, take);
        if (m_fr.kind == 'M' || m_fr.kind == 'C')
            for (size_t j = 0; j < take && m_fr.mn < sizeof m_fr.meta - 1; j++)
                m_fr.meta[m_fr.mn++] = (char)b[i + j];
        m_fr.got += (uint32_t)take;
        i += take;
        if (m_fr.got == m_fr.len) {
            if (m_fr.kind == 'M') {
                m_fr.meta[m_fr.mn] = 0;
                if (strstr(m_fr.meta, M_MARK) && m_marks_seen < m_marks_sent) m_marks_seen++;
            }
            if (m_fr.kind == 'C') {                      // "rate=NNNNN\n": the horizon clock's unit
                m_fr.meta[m_fr.mn] = 0;
                const char *r = strstr(m_fr.meta, "rate=");
                if (r && atoi(r + 5) > 0) m_rate = atoi(r + 5);
            }
            m_fr.hn = 0;
        }
    }
}

// Pump synth mux -> ring -> player. Called every mic tick: reads never block,
// the player's pipe is written non-blocking with the ring holding the rest.
static void mouth_pump(void) {
    if (m_synth_out < 0) return;
    uint8_t buf[8192];
    int k;
    while ((k = (int)read(m_synth_out, buf, sizeof buf)) > 0) {
        m_last_pcm = now_sec();
        mux_feed(buf, (size_t)k);
    }
    // escape hatch: a lost marker echo would mute the mouth for good; a
    // long-idle synth means the stale tail has drained either way
    if (m_marks_seen < m_marks_sent && now_sec() - m_last_pcm > 3.0) m_marks_seen = m_marks_sent;
    if (m_rn == 0) return;
    if (m_play_in < 0) {
        m_play_pid = spawn_cmd(g_mouth_play, &m_play_in, NULL);
        if (m_play_pid < 0) { fprintf(stderr, "voicecat: mouth player spawn failed\n"); m_rn = 0; return; }
        fcntl(m_play_in, F_SETFL, O_NONBLOCK);
    }
    int k2 = (int)write(m_play_in, m_ring, m_rn > 65536 ? 65536 : m_rn);
    if (k2 > 0) {
        memmove(m_ring, m_ring + k2, m_rn - (size_t)k2);
        m_rn -= (size_t)k2;
        double base = m_audible_until > now_sec() ? m_audible_until : now_sec();
        m_audible_until = base + (double)k2 / (double)(m_rate * 2);
    }
    else if (k2 < 0 && !wouldblock()) {      // player died underneath us
        close(m_play_in); m_play_in = -1;
        if (m_play_pid > 0) { kill(-(int)m_play_pid, SIGKILL); waitpid((int)m_play_pid, NULL, 0); m_play_pid = 0; }
    }
}

// Barge: silence within the player's latency buffer. The synth lives on —
// the marker line makes everything queued before it stale, drained unheard.
// `mute` (a barged reply is still streaming) also eats the dying reply's
// remaining clauses until its turn-end newline arrives.
static void mouth_cut(int mute) {
    if (!g_mouth_synth) return;
    if (m_play_pid > 0) { kill(-(int)m_play_pid, SIGKILL); waitpid((int)m_play_pid, NULL, 0); m_play_pid = 0; }
    if (m_play_in >= 0) { close(m_play_in); m_play_in = -1; }
    m_rn = 0;
    m_last_pcm = now_sec();
    m_audible_until = now_sec() + 0.3;       // the cut lands within the sink's tail
    memset(&m_cl, 0, sizeof m_cl);           // a half-built clause dies with the turn
    m_mute_turn = mute;
    if (m_synth_in >= 0) {
        static const char mark[] =
            "<|tool_call>call:set_voice{speaker_id:<|\"|>" M_MARK "<|\"|>}<tool_call|>\n";
        if (write(m_synth_in, mark, sizeof mark - 1) == (int)(sizeof mark - 1))
            m_marks_sent++;
    }
}

static void mouth_close(void) {
    if (!g_mouth_synth) return;
    if (m_synth_in >= 0) { close(m_synth_in); m_synth_in = -1; }
    double t0 = now_sec();                   // let the tail finish speaking, capped
    while (now_sec() - t0 < 3.0) {
        mouth_pump();
        if (m_rn == 0 && now_sec() - m_last_pcm > 0.5) break;
        msleep(20);
    }
    if (m_synth_pid > 0) { kill(-(int)m_synth_pid, SIGKILL); waitpid((int)m_synth_pid, NULL, 0); m_synth_pid = 0; }
    if (m_play_pid > 0)  { kill(-(int)m_play_pid, SIGKILL); waitpid((int)m_play_pid, NULL, 0); m_play_pid = 0; }
}

static void m_flush_line(void) {
    size_t a = 0, b = m_cl.ln;
    while (a < b && (m_cl.line[a] == ' ' || m_cl.line[a] == '\t')) a++;
    while (b > a && (m_cl.line[b - 1] == ' ' || m_cl.line[b - 1] == '\t')) b--;
    if (b > a && m_synth_in >= 0) {
        m_cl.line[b] = '\n';
        if (write(m_synth_in, m_cl.line + a, b - a + 1) < 0 && !wouldblock())
            fprintf(stderr, "voicecat: mouth synth pipe broke\n");
    }
    m_cl.ln = 0;
}
static int m_is_punct(char c) { return strchr(",;:.!?", c) != NULL; }
static int m_is_after(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' ||
           c == '*' || c == ')' || c == '"' || c == '\'' || c == ']';
}
static void m_emit(char c) {
    if (m_cl.ln > 0 && m_is_punct(m_cl.line[m_cl.ln - 1]) && m_is_after(c))
        m_flush_line();
    if (m_cl.ln < sizeof m_cl.line - 2) m_cl.line[m_cl.ln++] = c;
}

static void mouth_feed(const char *in, int n) {
    if (!g_mouth_synth) return;
    static const char CLOSE[] = "<channel|>", TCLOSE[] = "<tool_call|>";
    for (int i = 0; i < n; i++) {
        char c = in[i];
        if (m_mute_turn) {                   // the barged reply's tail: unheard by design
            if (c == '\n') { m_mute_turn = 0; memset(&m_cl, 0, sizeof m_cl); }
            continue;
        }
        if (c == '\n') {                     // end of turn: pending '<...'/'[' was literal
            if (!m_cl.thought && !m_cl.tcall) {
                if (m_cl.pb) m_emit('[');
                for (size_t j = 0; j < m_cl.tn; j++) m_emit(m_cl.tag[j]);
            }
            m_flush_line();
            memset(&m_cl, 0, sizeof m_cl);
            continue;
        }
        if (m_cl.thought) {
            m_cl.cn = (c == CLOSE[m_cl.cn]) ? m_cl.cn + 1 : (c == CLOSE[0] ? 1 : 0);
            if (m_cl.cn == sizeof CLOSE - 1) { m_cl.thought = 0; m_cl.cn = 0; }
            continue;
        }
        if (m_cl.tcall) {                    // control spans drop whole here
            m_cl.tcn = (c == TCLOSE[m_cl.tcn]) ? m_cl.tcn + 1 : (c == TCLOSE[0] ? 1 : 0);
            if (m_cl.tcn == sizeof TCLOSE - 1) { m_cl.tcall = 0; m_cl.tcn = 0; }
            continue;
        }
        if (m_cl.tn > 0) {                   // inside a potential <token>
            if (c == '>') {
                if (m_cl.tn == 1) { m_emit('<'); m_emit('>'); m_cl.tn = 0; continue; }
                m_cl.tag[m_cl.tn] = 0;
                if (!strcmp(m_cl.tag, "<|channel"))        m_cl.thought = 1;
                else if (!strcmp(m_cl.tag, "<|tool_call")) m_cl.tcall = 1;
                m_cl.tn = 0;
                continue;
            }
            if (c != '<' && m_cl.tn < sizeof m_cl.tag - 2) { m_cl.tag[m_cl.tn++] = c; continue; }
            for (size_t j = 0; j < m_cl.tn; j++) m_emit(m_cl.tag[j]);
            m_cl.tn = 0;
        }
        if (m_cl.brk) {                      // [[...]] inline tags are control, not speech
            if (c == ']' && m_cl.bn > 0 && m_cl.tsp[m_cl.bn - 1] == ']') { m_cl.brk = 0; m_cl.bn = 0; continue; }
            if (m_cl.bn < sizeof m_cl.tsp - 1) { m_cl.tsp[m_cl.bn++] = c; continue; }
            m_emit('['); m_emit('[');
            for (size_t j = 0; j < m_cl.bn; j++) m_emit(m_cl.tsp[j]);
            m_cl.brk = 0; m_cl.bn = 0;
        }
        if (m_cl.pb) {
            m_cl.pb = 0;
            if (c == '[') { m_cl.brk = 1; m_cl.bn = 0; continue; }
            m_emit('[');
        }
        if (c == '[') { m_cl.pb = 1; continue; }
        if (c == '<') { m_cl.tag[m_cl.tn++] = c; continue; }
        m_emit(c);
    }
}
// "Speaking" = the audible horizon hasn't passed (plus anything still in the
// ring). NOT "a player process exists" (persistent player = deaf forever) and
// NOT "the pump was recently active" (the synth outruns real time — the pump
// idles while the downstream queue still holds half a minute). Both measured.
static int mouth_speaking(void) {
    return m_rn > 0 || now_sec() < m_audible_until;
}
#else
static void mouth_start(void) {}
static void mouth_pump(void)  {}
static void mouth_cut(int mute) { (void)mute; }
static void mouth_close(void) {}
static void mouth_feed(const char *in, int n) { (void)in; (void)n; }
static int  mouth_speaking(void) { return 0; }
#endif

// One received chunk: probe envelopes lifted out, the rest to stdout and the
// turn matcher. A finished turn also clears the probe-inflight latch — the
// socket is FIFO, so any probe answer would have arrived before the turn end.
static void deliver(const char *in, int k, int *pending, int *tstate) {
    char out[4096 + 16];
    int on = pf_feed(in, k, out);
    fwrite(out, 1, (size_t)on, stdout);
    mouth_feed(out, on);
    int done = turn_ends(out, on, tstate);
    while (done-- > 0) { (*pending)--; putchar('\n'); mouth_feed("\n", 1); g_probe_inflight = 0; }
    fflush(stdout);
}

// The duck line to far-field-service: hello 'c' once, then 'd'/'u' frames.
static sock_t g_duck_fd = INVALID_SOCKET;
static void duck_hdr(uint8_t type) {
    if (g_duck_fd == INVALID_SOCKET) return;
    uint8_t hdr[LG_FRAME_HDR] = { LG_FRAME_MAGIC, type };
    if (send_all(g_duck_fd, hdr, sizeof hdr) != 0) {
        sock_close(g_duck_fd);                       // service gone: fall back to
        g_duck_fd = INVALID_SOCKET;                  // single-stage cuts
    }
}
static void duck_set(int on) { duck_hdr(on ? 'd' : 'u'); }

static sock_t sock_connect(const char *spath) {
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    if (strlen(spath) >= sizeof sa.sun_path) return INVALID_SOCKET;
    strcpy(sa.sun_path, spath);
    sock_t s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return s;
    if (connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) { sock_close(s); return INVALID_SOCKET; }
#ifdef _WIN32
    u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
    return s;
}

// Collect one full reply from the socket (until "<turn|>", 120 s cap).
// Returns the byte count, or -1 when the turn never finished.
static int read_turn(sock_t s, char *raw, size_t cap) {
    size_t rn = 0;
    int st = 0, done = 0;
    double t0 = now_sec();
    while (!done && now_sec() - t0 < 120.0) {
        char buf[1024];
        int k = (int)recv(s, buf, sizeof buf, 0);
        if (k <= 0) {
            if (k < 0 && wouldblock()) { msleep(20); continue; }
            break;
        }
        done = turn_ends(buf, k, &st) > 0;
        for (int i = 0; i < k && rn < cap - 1; i++) raw[rn++] = buf[i];
    }
    raw[rn] = 0;
    return done ? (int)rn : -1;
}

// A model reply -> one plain line: thought spans, <tokens> and [[tags]]
// dropped whole, whitespace folded. Returns the length.
static size_t strip_reply(const char *raw, size_t rn, char *dig, size_t cap) {
    size_t dn = 0;
    for (size_t i = 0; i < rn && dn < cap - 1; ) {
        if (!strncmp(raw + i, "<|channel>", 10)) {
            const char *e = strstr(raw + i + 10, "<channel|>");
            i = e ? (size_t)(e - raw) + 10 : rn;
            continue;
        }
        if (raw[i] == '<') {
            size_t j = i + 1;
            while (j < rn && raw[j] != '<' && raw[j] != '>' && j - i <= 24) j++;
            if (j < rn && raw[j] == '>' && j - i >= 2) { i = j + 1; continue; }
        }
        if (raw[i] == '[' && i + 1 < rn && raw[i + 1] == '[') {   // [[tags]] drop whole
            const char *e = strstr(raw + i + 2, "]]");
            if (e) { i = (size_t)(e - raw) + 2; continue; }
        }
        char c = raw[i++];
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (c == ' ' && (dn == 0 || dig[dn - 1] == ' ')) continue;
        dig[dn++] = c;
    }
    while (dn && dig[dn - 1] == ' ') dn--;
    dig[dn] = 0;
    return dn;
}

// One question, one throwaway session: ask + material as a single turn,
// stripped reply into `out`. -1 on any trouble (caller keeps its originals).
static int ask_once(const char *spath, const char *ask, const char *material, char *out, size_t cap) {
    sock_t s = sock_connect(spath);
    if (s == INVALID_SOCKET) return -1;
    int bad = send_all(s, ask, strlen(ask)) != 0 ||
              (material && (send_all(s, " ", 1) != 0 || send_all(s, material, strlen(material)) != 0)) ||
              send_all(s, "\n", 1) != 0;
    char raw[16384];
    int rn = bad ? -1 : read_turn(s, raw, sizeof raw);
    sock_close(s);
    if (rn < 0) return -1;
    return strip_reply(raw, (size_t)rn, out, cap) > 0 ? 0 : -1;
}

// Parse a ledger line's period: 0 = raw "[Thu 2026-07-09 14:31]", 1 = week
// "[week of 2026-06-29]", 2 = month "[2026-06]", 3 = year "[2026]"; -1 = no
// date. Fills `start` with the period's first day (noon, DST-safe).
static int ledger_period(const char *line, struct tm *start) {
    int y = 0, m = 0, d = 0, hh, mm;
    memset(start, 0, sizeof *start);
    start->tm_mday = 1; start->tm_hour = 12; start->tm_isdst = -1;
    if (sscanf(line, "[%*3s %d-%d-%d %d:%d]", &y, &m, &d, &hh, &mm) == 5 && m >= 1 && m <= 12) {
        start->tm_year = y - 1900; start->tm_mon = m - 1; start->tm_mday = d;
        return 0;
    }
    if (sscanf(line, "[week of %d-%d-%d]", &y, &m, &d) == 3 && m >= 1 && m <= 12) {
        start->tm_year = y - 1900; start->tm_mon = m - 1; start->tm_mday = d;
        return 1;
    }
    if (sscanf(line, "[%d-%d]", &y, &m) == 2 && y >= 1970 && m >= 1 && m <= 12) {
        start->tm_year = y - 1900; start->tm_mon = m - 1;
        return 2;
    }
    if (sscanf(line, "[%d]", &y) == 1 && y >= 1970) {
        start->tm_year = y - 1900;
        return 3;
    }
    return -1;
}

// Is this line old enough to fold into the next level? Raw lines age out
// after a week, week lines after ~a month, month lines after ~a year.
static int ledger_due(const char *line, time_t now, int *kind, time_t *t0) {
    struct tm st;
    *kind = ledger_period(line, &st);
    if (*kind < 0 || *kind > 2) return 0;
    *t0 = mktime(&st);
    if (*t0 <= 0) return 0;
    double age = difftime(now, *t0);
    return (*kind == 0 && age > 7 * 86400.0) ||
           (*kind == 1 && age > 35 * 86400.0) ||
           (*kind == 2 && age > 400 * 86400.0);
}

// The merged line's prefix: a raw line folds to its week's Monday, a week
// line to its month, a month line to its year.
static void ledger_prefix(int kind, time_t t0, char *prefix, size_t cap) {
    struct tm lt = *localtime(&t0);
    if (kind == 0) {
        time_t w = t0 - (time_t)((lt.tm_wday + 6) % 7) * 86400;
        lt = *localtime(&w);
        strftime(prefix, cap, "[week of %Y-%m-%d]", &lt);
    } else if (kind == 1)
        strftime(prefix, cap, "[%Y-%m]", &lt);
    else
        strftime(prefix, cap, "[%Y]", &lt);
}

// Age the ledger (see the --memory comment): fold every due group through
// its merge ask over a THROWAWAY session — call only while no main session
// occupies the server. A failed ask copies its lines through untouched;
// oversized groups merge in batches whose partial gists share a prefix and
// unify at the next level up.
static void consolidate_ledger(const char *spath) {
    FILE *f = fopen(g_memory, "rb");
    if (!f) return;
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    if (!n) return;
    static char *lines[8192];
    int nlines = 0;
    for (char *q = buf; *q && nlines < 8192; ) {
        lines[nlines++] = q;
        char *e = strchr(q, '\n');
        if (!e) break;
        *e = 0;
        q = e + 1;
    }
    char tmp[1024];
    if (snprintf(tmp, sizeof tmp, "%s.tmp", g_memory) >= (int)sizeof tmp) return;
    FILE *out = fopen(tmp, "wb");
    if (!out) return;
    time_t now = time(NULL);
    int merges = 0, fails = 0;
    for (int i = 0; i < nlines; ) {
        int kind;
        time_t t0;
        if (!*lines[i]) { i++; continue; }
        if (!ledger_due(lines[i], now, &kind, &t0)) {
            fprintf(out, "%s\n", lines[i++]);
            continue;
        }
        char prefix[32], pf2[32];
        ledger_prefix(kind, t0, prefix, sizeof prefix);
        char material[6144];
        size_t mlen = 0;
        int j = i;
        while (j < nlines) {                             // same kind, same target, still due
            size_t L = strlen(lines[j]);
            if (j > i) {
                int k2; time_t t2;
                if (!ledger_due(lines[j], now, &k2, &t2) || k2 != kind) break;
                ledger_prefix(kind, t2, pf2, sizeof pf2);
                if (strcmp(pf2, prefix)) break;
                if (mlen + L + 2 > sizeof material) break;
            }
            memcpy(material + mlen, lines[j], L);
            mlen += L;
            material[mlen++] = ' ';
            j++;
        }
        material[mlen] = 0;
        const char *ask = kind == 0 ? g_merge_week : kind == 1 ? g_merge_month : g_merge_year;
        char merged[2048];
        if (ask_once(spath, ask, material, merged, sizeof merged) == 0) {
            fprintf(out, "%s %s\n", prefix, merged);
            merges++;
        } else {
            for (int k = i; k < j; k++) fprintf(out, "%s\n", lines[k]);
            fails++;
        }
        i = j;
    }
    fclose(out);
    if (merges > 0) {
        remove(g_memory);
        if (rename(tmp, g_memory) != 0)
            fprintf(stderr, "voicecat: ledger rewrite failed — %s.tmp holds the update\n", g_memory);
        else
            fprintf(stderr, "voicecat: ledger consolidated — %d merge(s)%s\n",
                    merges, fails ? " (some kept raw after a failed ask)" : "");
    } else
        remove(tmp);
}

// The idle-compression cycle (see the flag comment above). Blocks the mic loop
// for one digest generation — the user has been quiet for many minutes, and a
// live mic's queued ticks catch up afterwards exactly like a whisper pass.
// On any trouble the old session stays; on success returns the fresh socket.
static sock_t compress_session(sock_t s, const char *spath) {
    fprintf(stderr, "voicecat: idle — compressing the session\n");
    char ask[640];
    int an = snprintf(ask, sizeof ask, "%s\n", g_compress_ask);
    if (an >= (int)sizeof ask || send_all(s, ask, (size_t)an) != 0) return s;
    char raw[8192];                     // the digest reply: consumed, never spoken
    int rn = read_turn(s, raw, sizeof raw);
    if (rn < 0) {
        fprintf(stderr, "voicecat: digest never finished — keeping the session as it is\n");
        return s;
    }
    char dig[4096];
    size_t dn = strip_reply(raw, (size_t)rn, dig, sizeof dig);
    sock_close(s);
    if (g_memory && dn) {               // the ledger: one dated line per compress
        FILE *mf = fopen(g_memory, "ab");
        if (mf) {
            char stamp[64];
            time_t t = time(NULL);
            strftime(stamp, sizeof stamp, "[%a %Y-%m-%d %H:%M] ", localtime(&t));
            fprintf(mf, "%s%s\n", stamp, dig);
            fclose(mf);
        } else
            fprintf(stderr, "voicecat: cannot append to %s\n", g_memory);
    }
    if (g_memory)                       // age the ledger while no session is open
        consolidate_ledger(spath);
    sock_t ns = sock_connect(spath);
    if (ns == INVALID_SOCKET) { fprintf(stderr, "voicecat: reconnect failed after compress\n"); exit(1); }
    char seed[4608];
    char when[48] = "";
    if (g_clock > 0) {                  // the fresh session opens knowing the hour
        time_t t = time(NULL);
        strftime(when, sizeof when, "it is now %a %H:%M; ", localtime(&t));
    }
    int sl = snprintf(seed, sizeof seed, "(%smemory of our conversation before this pause: %s) ", when, dig);
    if (sl >= (int)sizeof seed) sl = (int)sizeof seed - 1;
    send_frame(ns, LG_FRAME_TEXT, seed, (uint32_t)sl);
    fprintf(stderr, "voicecat: session compressed — %d-char digest seated in a fresh context\n", (int)dn);
    return ns;
}

// Seed a fresh voicecat from the ledger's tail: the last ~1800 chars of whole
// dated lines ride in as the open turn's first text, so the conversation
// resumes across process restarts already knowing when everything happened.
static void seed_memory(sock_t s) {
    FILE *mf = fopen(g_memory, "rb");
    if (!mf) return;
    char led[2048];
    fseek(mf, 0, SEEK_END);
    long sz = ftell(mf);
    long off = sz > 1800 ? sz - 1800 : 0;
    fseek(mf, off, SEEK_SET);
    size_t n = fread(led, 1, sizeof led - 1, mf);
    fclose(mf);
    led[n] = 0;
    char *p = led;
    if (off > 0) { p = strchr(led, '\n'); p = p ? p + 1 : led; }   // whole lines only
    for (char *q = p; *q; q++)
        if (*q == '\n' || *q == '\r') *q = ' ';
    while (*p == ' ') p++;
    if (!*p) return;
    char seed[2304];
    // newest-wins is the read-side of Graphiti's invalidation: entries are
    // chronological and never rewritten, so contradictions resolve at recall
    int sl = snprintf(seed, sizeof seed,
                      "(memory from our earlier conversations, oldest first - "
                      "where entries disagree, the newest is current: %s) ", p);
    if (sl >= (int)sizeof seed) sl = (int)sizeof seed - 1;
    send_frame(s, LG_FRAME_TEXT, seed, (uint32_t)sl);
    fprintf(stderr, "voicecat: %d chars of dated memory seeded from %s\n", sl, g_memory);
}

// Byte offset just past word `k` (0 = start of word 1).
static size_t after_word(const char *s, int k) {
    const char *p = s;
    for (int i = 0; i < k; i++) {
        while (*p == ' ') p++;
        while (*p && *p != ' ') p++;
    }
    return (size_t)(p - s);
}
// Words agreeing between two passes, counted from the start (LocalAgreement).
static int agree_words(const char *a, const char *b) {
    int n = 0;
    for (;;) {
        while (*a == ' ') a++;
        while (*b == ' ') b++;
        if (!*a || !*b) return n;
        const char *ea = a, *eb = b;
        while (*ea && *ea != ' ') ea++;
        while (*eb && *eb != ' ') eb++;
        if (ea - a != eb - b || memcmp(a, b, (size_t)(ea - a)) != 0) return n;
        n++; a = ea; b = eb;
    }
}

// ---- the conversation ---------------------------------------------------------
int main(int argc, char **argv) {
    const char *spath = NULL, *mic = NULL;
    int stdin_pcm = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--mic") && i + 1 < argc)           mic = argv[++i];
        else if (!strcmp(argv[i], "--stdin-pcm"))                     stdin_pcm = 1;
        else if (!strcmp(argv[i], "--realtime"))                      g_realtime = 1;
        else if (!strcmp(argv[i], "--whisper-model") && i + 1 < argc) g_whisper_model = argv[++i];
        else if (!strcmp(argv[i], "--whisper-bin") && i + 1 < argc)   g_whisper_bin = argv[++i];
        else if (!strcmp(argv[i], "--whisper-url") && i + 1 < argc)   g_whisper_url = argv[++i];
        else if (!strcmp(argv[i], "--mouth-synth") && i + 1 < argc)   g_mouth_synth = argv[++i];
        else if (!strcmp(argv[i], "--mouth-play") && i + 1 < argc)    g_mouth_play = argv[++i];
        else if (!strcmp(argv[i], "--hush-tail"))                     g_hush_tail = 1;
        else if (!strcmp(argv[i], "--barge-mult") && i + 1 < argc)    g_barge_mult = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--barge-onset") && i + 1 < argc)   g_barge_onset = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duck-sock") && i + 1 < argc)     g_duck_sock = argv[++i];
        else if (!strcmp(argv[i], "--barge-note") && i + 1 < argc)    g_barge_note = argv[++i];
        else if (!strcmp(argv[i], "--commit-ms") && i + 1 < argc)     g_commit_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hang-ms") && i + 1 < argc)       g_hang_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vad-level") && i + 1 < argc)     g_vad = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--listener"))                      g_listener = 1;
        else if (!strcmp(argv[i], "--probe-ms") && i + 1 < argc)      g_probe_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-gen") && i + 1 < argc)     g_probe_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-suffix") && i + 1 < argc)  g_probe_suffix = argv[++i];
        else if (!strcmp(argv[i], "--idle-compress") && i + 1 < argc) g_idle_compress = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--clock") && i + 1 < argc)         g_clock = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--memory") && i + 1 < argc)        g_memory = argv[++i];
        else if (!spath)                                              spath = argv[i];
        else { fprintf(stderr, "voicecat: unknown argument %s\n", argv[i]); return 1; }
    }
    if (!spath) {
        fprintf(stderr,
            "usage: voicecat <socket> [--mic FFMPEG_INPUT | --stdin-pcm [--realtime]]\n"
            "                [--whisper-model FILE] [--whisper-bin PATH] [--whisper-url URL]\n"
            "                [--barge-note TEXT] [--mouth-synth CMD --mouth-play CMD]\n"
            "                [--commit-ms N=2500] [--hang-ms N=700] [--vad-level N=400]\n"
            "                [--listener [--probe-ms N=1500] [--probe-gen N=6] [--probe-suffix TEXT]]\n"
            "  --mic        ffmpeg input spec (default \"-f alsa -i default\" on Linux)\n"
            "  --stdin-pcm  mono 16 kHz s16 PCM on stdin instead of a mic\n"
            "  --whisper-url  POST passes to a resident whisper-server /inference instead\n"
            "               of spawning whisper-cli — no model load per pass; with fast\n"
            "               passes --commit-ms can drop to ~3x the measured pass cost\n"
            "  --mouth-synth CMD   voicecat OWNS the TTS (POSIX): clause lines (clausecat's\n"
            "  --mouth-play  CMD   policy, in-process) feed CMD's stdin (spawned once, kept\n"
            "               warm); its PCM pumps into --mouth-play (respawned cheaply). A\n"
            "               barge or speech over the reply's tail kills ONLY the player —\n"
            "               the sound stops within its latency buffer\n"
            "  --duck-sock PATH    two-stage barge via far-field-service: onset DUCKS the\n"
            "               reply (it keeps playing, quieter); the cut waits for words to\n"
            "               materialize; a false alarm swells the reply back unharmed\n"
            "  --realtime   pace stdin PCM at wall-clock rate, as a live mic would deliver it\n"
            "               (replay recordings realistically; a real mic paces itself)\n"
            "  --listener   probe the model while the user speaks (after commits and on\n"
            "               pauses): nod/backchannel/end-point actions on stderr; needs whisper\n"
            "  --idle-compress N   after N seconds of quiet, digest the conversation into a\n"
            "               fresh session (default 900; 0 = off) — long sessions stay young\n"
            "  --clock N    open every turn with the real time, \"[14:32] \"; a quiet longer\n"
            "               than N seconds also gets \"(27 minutes of quiet pass)\" so the\n"
            "               model senses the gap (default 300; 0 = no time markers)\n"
            "  --memory FILE   the dated ledger: each compress appends one \"[date] digest\"\n"
            "               line, and a fresh voicecat seeds itself from the tail — memory\n"
            "               that survives restarts and knows when things happened\n"
            "  no whisper model -> whole utterances as native audio spans (vision-free sessions only)\n");
        return 1;
    }
    if (!g_whisper_model) g_whisper_model = getenv("LG_WHISPER_MODEL");
    if (!g_whisper_bin)   g_whisper_bin   = getenv("LG_WHISPER_BIN");
    if (!g_whisper_url)   g_whisper_url   = getenv("LG_WHISPER_URL");
    if (g_whisper_url && !*g_whisper_url) g_whisper_url = NULL;
    if ((g_mouth_synth != NULL) != (g_mouth_play != NULL)) {
        fprintf(stderr, "voicecat: --mouth-synth and --mouth-play come as a pair\n");
        return 1;
    }
#ifdef _WIN32
    if (g_mouth_synth) { fprintf(stderr, "voicecat: --mouth-* is POSIX-only for now\n"); return 1; }
#endif
    int whisper = (g_whisper_model && *g_whisper_model) || g_whisper_url;
    if (!whisper)
        fprintf(stderr, "voicecat: no whisper model — native audio spans "
                        "(the model's ears work only while the session is vision-free)\n");
    if (g_listener && !whisper) {
        fprintf(stderr, "voicecat: --listener needs streaming transcripts (a whisper model) — "
                        "a native-span turn has no open text to probe; listener off\n");
        g_listener = 0;
    }

    // audio source
    FILE *src;
    if (stdin_pcm) {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        src = stdin;
    } else {
#ifdef _WIN32
        if (!mic) { fprintf(stderr, "voicecat: --mic \"-f dshow -i audio=...\" is required on Windows\n"); return 1; }
#else
        if (!mic) mic = "-f alsa -i default";
#endif
        char cmd[2048];
        snprintf(cmd, sizeof cmd, "ffmpeg -hide_banner -loglevel error %s -f s16le -ac 1 -ar %d -", mic, LG_RATE);
        src = run_pipe(cmd);
        if (!src) { fprintf(stderr, "voicecat: ffmpeg capture failed\n"); return 1; }
    }

    // the session socket, non-blocking so reply bytes drain between mic ticks
    if (sock_init() != 0) { fprintf(stderr, "voicecat: socket setup failed\n"); return 1; }
    if (g_memory) consolidate_ledger(spath);   // throwaway sessions — must run
    sock_t s = sock_connect(spath);            // before the main session occupies
    if (s == INVALID_SOCKET) { fprintf(stderr, "voicecat: connect to %s failed\n", spath); return 1; }
    if (g_memory) seed_memory(s);
    mouth_start();
    if (g_duck_sock) {
        g_duck_fd = sock_connect(g_duck_sock);
        if (g_duck_fd == INVALID_SOCKET)
            fprintf(stderr, "voicecat: --duck-sock %s unreachable — single-stage barge\n", g_duck_sock);
        else
            duck_hdr('c');                           // the control hello
    }

    int16_t frame[FR_SAMP], ring[PREROLL * FR_SAMP];
    int ring_n = 0;                                  // valid ticks in the preroll ring (rolls)
    int16_t *ub = NULL; size_t ub_cap = 0, ub_n = 0; // the audio window (trimmed as words confirm)
    char prev[8192] = "", cur[8192];                 // last two whisper passes over the window
    struct wseg segs[WSEG_MAX]; int nseg = 0;
    int committed = 0;                               // words sent as 'T' this utterance
    int trimmed = 0;                                 // of those, words whose audio left the window
    char ptail[400] = "";                            // trimmed text tail -> whisper --prompt
    size_t last_pass = 0;                            // ub_n at the previous whisper pass
    size_t last_voice = 0;                           // ub_n at the last voiced frame
    int in_utt = 0, onset = 0, sil_ms = 0, turn_open = 0, barged = 0, pause_probed = 0;
    int duck_pending = 0;                            // a stage-1 duck awaiting words
    int pending = 0, barge_armed = 1, tstate = 0;    // replies awaited; one barge per utterance
    char rbuf[4096];
    double last_rx = now_sec();                      // last reply byte (pending-turn watchdog)
    double rt0 = 0; long rtn = 0;                    // --realtime deadline pacing
    double last_activity = now_sec();                // the idle clock (--idle-compress)
    int dirty = 0;                                   // turns exchanged since the last compress

    fprintf(stderr, "voicecat: %s, listening\n", whisper ? "streaming transcripts" : "native audio spans");
    for (;;) {
        size_t got = fread(frame, 2, FR_SAMP, src);
        int eof = got < FR_SAMP;
        if (g_realtime && !eof) {
            // pace to the frame's DEADLINE, not a flat sleep: a live mic keeps
            // capturing while a whisper pass runs and the loop catches up from
            // the buffer — this models that, so passes overlap "capture"
            struct timespec ts; timespec_get(&ts, TIME_UTC);
            double now = (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
            if (!rt0) rt0 = now;
            rtn++;
            double target = rt0 + rtn * (FR_MS / 1000.0);
            if (target > now) msleep((int)((target - now) * 1000) + 1);
        }
        // drain any reply bytes; each completed turn drops `pending`
        for (;;) {
            int k = (int)recv(s, rbuf, sizeof rbuf, 0);
            if (k <= 0) break;
            last_rx = now_sec();
            deliver(rbuf, k, &pending, &tstate);
        }
        // a server that never answers must not deafen the ears forever: with
        // pending stuck, the raised in-reply barge bar gates ALL speech — so
        // give up on the turn, re-arm, and let the mouth machine reset
        if (pending > 0 && now_sec() - last_rx > 60.0) {
            fprintf(stderr, "voicecat: no reply in 60s — abandoning %d pending turn(s)\n", pending);
            pending = 0;
            barge_armed = 1;
            mouth_feed("\n", 1);
        }
        mouth_pump();                                    // synth PCM -> ring -> player

        // the idle clock: any speech or reply in flight resets it; a long
        // enough quiet (and something worth remembering) compresses the session
        if (in_utt || pending > 0) last_activity = now_sec();
        else if (g_idle_compress > 0 && dirty && !g_probe_inflight &&
                 now_sec() - last_activity > (double)g_idle_compress) {
            s = compress_session(s, spath);
            dirty = 0;
            last_activity = now_sec();
        }

        if (!eof) {
            long long ss = 0;
            for (int i = 0; i < FR_SAMP; i++) ss += (long long)frame[i] * frame[i];
            int rms = (int)sqrt((double)ss / FR_SAMP);
            int voiced = rms >= g_vad;

            if (!in_utt) {
                memmove(ring, ring + FR_SAMP, sizeof ring - sizeof frame);   // roll the preroll
                memcpy(ring + (PREROLL - 1) * FR_SAMP, frame, sizeof frame);
                if (ring_n < PREROLL) ring_n++;
                int busy = pending > 0 || mouth_speaking();
                onset = (rms >= (busy ? g_vad * g_barge_mult : g_vad)) ? onset + 1 : 0;
                if (onset >= (busy ? g_barge_onset : ONSET)) {   // an utterance begins
                    if (busy && g_duck_fd != INVALID_SOCKET) {   // stage 1: duck, keep
                        duck_set(1);                     // talking — the cut waits for
                        duck_pending = 1;                // words to materialize
                        fprintf(stderr, "voicecat: barge? ducked\n");
                    } else if (pending > 0 && barge_armed) {   // ...over the reply: barge
                        uint8_t b = LG_BARGE;
                        send_all(s, &b, 1);
                        mouth_cut(1);                    // the sound stops NOW, not when
                        barged = 1; barge_armed = 0;     // the buffered clauses run out
                        fprintf(stderr, "voicecat: barge\n");
                    } else if (g_hush_tail && pending == 0 && mouth_speaking())
                        mouth_cut(0);                    // speech over the reply's tail:
                                                         // nothing to barge, still hush.
                                                         // OPT-IN: without real AEC the
                                                         // reply's own echo triggers it
                    ub_n = 0;
                    size_t pre = (size_t)ring_n * FR_SAMP;
                    if (ub_cap < pre + FR_SAMP) { ub_cap = 1 << 20; ub = realloc(ub, ub_cap * 2); }
                    memcpy(ub, ring + (PREROLL - ring_n) * FR_SAMP, pre * 2);
                    ub_n = pre;
                    in_utt = 1; sil_ms = 0; onset = 0; pause_probed = 0;
                    committed = 0; trimmed = 0; ptail[0] = 0;
                    prev[0] = 0; last_pass = 0; last_voice = ub_n; turn_open = 0;
                }
            } else {
                if (ub_n + FR_SAMP > ub_cap) { ub_cap *= 2; ub = realloc(ub, ub_cap * 2); }
                memcpy(ub + ub_n, frame, sizeof frame); ub_n += FR_SAMP;
                sil_ms = voiced ? 0 : sil_ms + FR_MS;
                if (voiced) { last_voice = ub_n; pause_probed = 0; }
                // a beat in the speech — the classic spot for a listener cue:
                // probe once per pause, on the committed context, rate-limited
                if (g_listener && turn_open && !voiced && sil_ms >= PROBE_PAUSE_MS && !pause_probed &&
                    pending == 0 && !g_probe_inflight &&
                    now_sec() - g_probe_t0 >= g_probe_ms / 1000.0) {
                    send_probe(s);
                    pause_probed = 1;
                }
                // mid-utterance whisper pass over the (trimmed) window: commit
                // words two consecutive passes agree on, then trim the window
                // past fully-confirmed segments — pass cost tracks the tail.
                // Skipped once trailing silence accumulates: the utterance is
                // probably ending, and with a per-invocation ASR (whisper-cli
                // reloads its model every pass) a late mid-pass only delays
                // the final one that supersedes it.
                if (whisper && sil_ms < 300 && ub_n - last_pass >= (size_t)g_commit_ms * (LG_RATE / 1000)) {
                    last_pass = ub_n;
                    if (whisper_pass(ub, ub_n, ptail, cur, sizeof cur, segs, &nseg) == 0) {
                        int agree = agree_words(prev, cur);
                        int local = committed - trimmed;             // sent words still in the window
                        if (agree > local) {
                            if (duck_pending) {          // stage 2: words materialized —
                                if (pending > 0 && barge_armed) {   // commit the barge for real
                                    uint8_t bb = LG_BARGE;
                                    send_all(s, &bb, 1);
                                    barge_armed = 0;
                                }
                                mouth_cut(pending > 0);
                                duck_set(0);             // post-cut audio plays at full level
                                barged = 1; duck_pending = 0;
                                fprintf(stderr, "voicecat: barge confirmed\n");
                            }
                            size_t a = after_word(cur, local), b = after_word(cur, agree);
                            while (cur[a] == ' ') a++;
                            char piece[8192];
                            int m = snprintf(piece, sizeof piece, "%s%s%.*s ",
                                             !turn_open ? turn_clock() : "",
                                             !turn_open && barged ? g_barge_note : "", (int)(b - a), cur + a);
                            if (m > (int)sizeof piece - 1) m = (int)sizeof piece - 1;
                            send_frame(s, LG_FRAME_TEXT, piece, (uint32_t)m);
                            if (!turn_open && barged) barged = 0;
                            turn_open = 1;
                            fprintf(stderr, "voicecat: +%d words committed\n", agree - local);
                            committed += agree - local;
                            // fresh words just landed in the open turn: a
                            // natural moment to ask the listener for a cue
                            if (g_listener && pending == 0 && !g_probe_inflight &&
                                now_sec() - g_probe_t0 >= g_probe_ms / 1000.0)
                                send_probe(s);
                        }
                        strcpy(prev, cur);
                        size_t trim = 0; int twords = 0;             // whole segments, all confirmed
                        for (int k = 0; k < nseg; k++) {
                            if (segs[k].words > committed - trimmed) break;
                            trim = segs[k].end_samp; twords = segs[k].words;
                        }
                        if (trim > ub_n) trim = ub_n;
                        if (ub_n - trim < (size_t)LG_RATE / 4)       // keep 0.25s of continuity
                            trim = ub_n > (size_t)LG_RATE / 4 ? ub_n - LG_RATE / 4 : 0;
                        if (trim > 0 && twords > 0) {
                            size_t cut = after_word(cur, twords);    // their text -> the prompt tail
                            size_t pl = strlen(ptail), add = cut < sizeof ptail - 1 ? cut : sizeof ptail - 1;
                            if (pl + add >= sizeof ptail - 1) {      // keep the tail, drop the head
                                size_t keep = sizeof ptail - 1 - add;
                                memmove(ptail, ptail + pl - keep, keep + 1); pl = keep;
                            }
                            memcpy(ptail + pl, cur, add); ptail[pl + add] = 0;
                            memmove(ub, ub + trim, (ub_n - trim) * 2);
                            ub_n -= trim;
                            last_pass = last_pass > trim ? last_pass - trim : 0;
                            last_voice = last_voice > trim ? last_voice - trim : 0;
                            trimmed += twords;
                            prev[0] = 0;                             // window moved; agreement restarts
                            fprintf(stderr, "voicecat: window -%0.1fs (%d words confirmed away)\n",
                                    (double)trim / LG_RATE, twords);
                        }
                    }
                }
            }
        }

        // end of the utterance: trailing silence, or the source ran dry
        if (in_utt && (sil_ms >= g_hang_ms || eof)) {
            in_utt = 0;
            if (whisper) {
                char line[8192] = "";
                // The final pass runs over the TRIMMED window — the unconfirmed
                // tail, not the whole utterance. And when no voiced frame
                // arrived after the last mid-pass began, that pass already
                // heard every spoken word: reuse its transcript instead of
                // paying a whole ASR invocation to re-transcribe silence.
                int heard = last_pass > 0 && last_voice <= last_pass;
                int ok = heard || whisper_pass(ub, ub_n, ptail, cur, sizeof cur, segs, &nseg) == 0;
                if (ok) {
                    size_t a = after_word(cur, committed - trimmed);
                    while (cur[a] == ' ') a++;
                    if (duck_pending && (turn_open || cur[a])) {     // stage 2 at the close:
                        if (pending > 0 && barge_armed) {            // short real interruptions
                            uint8_t bb = LG_BARGE;                   // materialize here, not
                            send_all(s, &bb, 1);                     // at a streamed commit
                            barge_armed = 0;
                        }
                        mouth_cut(pending > 0);
                        duck_set(0);
                        barged = 1; duck_pending = 0;
                        fprintf(stderr, "voicecat: barge confirmed\n");
                    }
                    if (turn_open || cur[a] || barged)   // never a clock-only turn
                        snprintf(line, sizeof line, "%s%s%s", !turn_open ? turn_clock() : "",
                                 !turn_open && barged ? g_barge_note : "", cur + a);
                }
                if (turn_open || line[0]) {              // silence/false trigger sends nothing
                    size_t L = strlen(line);
                    if (L > sizeof line - 2) L = sizeof line - 2;
                    line[L] = '\n'; line[L + 1] = 0;
                    send_all(s, line, L + 1);
                    pending++; barge_armed = 1; barged = 0; dirty = 1; g_last_close = now_sec();
                    fprintf(stderr, "voicecat: turn closed (%d+ words)\n", committed);
                } else if (barged || duck_pending) {     // the barge came to nothing (a door,
                    if (duck_pending) duck_set(0);       // a cough): un-duck — the reply SWELLS
                    duck_pending = 0;                    // BACK and keeps talking; a hard-cut
                    barged = 0; barge_armed = 1;         // false alarm just stays quiet. Either
                    fprintf(stderr, "voicecat: barge came to nothing\n");   // way the next real
                }                                        // turn is NOT an interruption
            } else if (ub_n >= LG_RATE / 2) {            // native span; drop sub-0.5s blips
                if (duck_pending) {                      // a real half-second of audio IS
                    if (pending > 0 && barge_armed) {    // materialization on this path
                        uint8_t bb = LG_BARGE;
                        send_all(s, &bb, 1);
                        barge_armed = 0;
                    }
                    mouth_cut(pending > 0);
                    duck_set(0);
                    barged = 1; duck_pending = 0;
                    fprintf(stderr, "voicecat: barge confirmed\n");
                }
                size_t pad = (LG_FRAME - ub_n % LG_FRAME) % LG_FRAME;
                if (ub_n + pad > ub_cap) { ub_cap = ub_n + pad; ub = realloc(ub, ub_cap * 2); }
                memset(ub + ub_n, 0, pad * 2);
                const char *ck = turn_clock();
                if (*ck) send_frame(s, LG_FRAME_TEXT, ck, (uint32_t)strlen(ck));
                if (barged) { send_frame(s, LG_FRAME_TEXT, g_barge_note, (uint32_t)strlen(g_barge_note)); barged = 0; }
                send_frame(s, LG_FRAME_AUDIO, ub, (uint32_t)((ub_n + pad) * 2));
                send_all(s, "\n", 1);
                pending++; barge_armed = 1; dirty = 1; g_last_close = now_sec();
                fprintf(stderr, "voicecat: turn closed (%.1fs audio)\n", (double)ub_n / LG_RATE);
            } else if (barged || duck_pending) {         // a sub-0.5s barge blip: same false alarm
                if (duck_pending) duck_set(0);
                duck_pending = 0;
                barged = 0; barge_armed = 1;
                fprintf(stderr, "voicecat: barge came to nothing\n");
            }
        }

        if (eof) {
            while (pending > 0) {                        // drain the last replies, then leave
                int k = (int)recv(s, rbuf, sizeof rbuf, 0);
                if (k > 0) deliver(rbuf, k, &pending, &tstate);
                else if (k == 0) break;
                else if (wouldblock()) { mouth_pump(); msleep(10); }
                else break;
            }
            break;
        }
    }
    mouth_close();
    free(ub);
    sock_close(s);
    return 0;
}

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
static const char *g_barge_note = "(interrupting) ";
// Commit cadence must comfortably exceed one ASR invocation (~0.6s for CUDA
// base.en via whisper-cli, process+model load included) or passes stack up
// and the loop falls behind real time: keep commit_ms >= 3x the pass cost.
static int g_commit_ms = 2500, g_hang_ms = 700, g_vad = 400, g_realtime = 0;

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
    "words. Anchor events to their clock times where that matters. "
    "Reply with only the summary.";
// --memory FILE: the dated LEDGER — every compress appends one line,
// "[Tue 2026-07-09 14:50] <digest>", and a fresh voicecat seeds its first
// turn from the ledger's tail, so memory survives process restarts and every
// entry is born knowing its date. This is the substrate for age-graded
// consolidation (an idle-cycle job, not yet built): entries older than a
// day/week/month/year merge into ever-shorter gists — a year, eventually,
// is one sentence. The merge mechanics are more digest asks with shrinking
// budgets; WHAT survives each merge (salience) is finetune territory —
// measured: E4B's digest kept "pasta, Ana, vegetarian" but dropped "Friday".
static const char *g_memory = NULL;

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

    size_t w = 0; int words = 0;
    char *line = raw;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *text = line;
        size_t seg_end = 0;
        char *arrow = strstr(line, "-->");
        if (line[0] == '[' && arrow) {                   // "[t0 --> t1]  text"
            seg_end = parse_ts(arrow + 3 + strspn(arrow + 3, " "));
            char *close = strchr(arrow, ']');
            text = close ? close + 1 : arrow + 3;
        }
        for (size_t i = 0; text[i]; ) {                  // strip [..] (..) tags, keep words
            char c = text[i];
            if (c == '[' || c == '(') {
                char close = c == '[' ? ']' : ')';
                while (text[i] && text[i] != close) i++;
                if (text[i]) i++;
                continue;
            }
            char cc = (c == '\r' || c == '\t') ? ' ' : c;
            if (cc == ' ' && (w == 0 || out[w - 1] == ' ')) { i++; continue; }
            if (w + 1 < cap) { out[w++] = cc; out[w] = 0; }
            if (cc == ' ') words++;                      // completed a word
            i++;
        }
        if (w && out[w - 1] != ' ' && w + 1 < cap) { out[w++] = ' '; out[w] = 0; words++; }
        if (seg_end && *nseg < WSEG_MAX) { segs[*nseg].end_samp = seg_end; segs[*nseg].words = words; (*nseg)++; }
        line = nl ? nl + 1 : NULL;
    }
    while (w && out[w - 1] == ' ') out[--w] = 0;
    if (*nseg == 0 && w) { segs[0].end_samp = nsamp; segs[0].words = words; *nseg = 1; }
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

// One received chunk: probe envelopes lifted out, the rest to stdout and the
// turn matcher. A finished turn also clears the probe-inflight latch — the
// socket is FIFO, so any probe answer would have arrived before the turn end.
static void deliver(const char *in, int k, int *pending, int *tstate) {
    char out[4096 + 16];
    int on = pf_feed(in, k, out);
    fwrite(out, 1, (size_t)on, stdout);
    int done = turn_ends(out, on, tstate);
    while (done-- > 0) { (*pending)--; putchar('\n'); g_probe_inflight = 0; }
    fflush(stdout);
}

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

// The idle-compression cycle (see the flag comment above). Blocks the mic loop
// for one digest generation — the user has been quiet for many minutes, and a
// live mic's queued ticks catch up afterwards exactly like a whisper pass.
// On any trouble the old session stays; on success returns the fresh socket.
static sock_t compress_session(sock_t s, const char *spath) {
    fprintf(stderr, "voicecat: idle — compressing the session\n");
    char ask[512];
    int an = snprintf(ask, sizeof ask, "%s\n", g_compress_ask);
    if (an >= (int)sizeof ask || send_all(s, ask, (size_t)an) != 0) return s;
    char raw[8192];                     // the digest reply: consumed, never spoken
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
        for (int i = 0; i < k && rn < sizeof raw - 1; i++) raw[rn++] = buf[i];
    }
    raw[rn] = 0;
    if (!done) {
        fprintf(stderr, "voicecat: digest never finished — keeping the session as it is\n");
        return s;
    }
    char dig[4096];                     // thought spans and <tokens> stripped
    size_t dn = 0;
    for (size_t i = 0; i < rn && dn < sizeof dig - 1; ) {
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
    sock_close(s);
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
    int sl = snprintf(seed, sizeof seed, "(memory from our earlier conversations: %s) ", p);
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
            "                [--whisper-model FILE] [--whisper-bin PATH] [--barge-note TEXT]\n"
            "                [--commit-ms N=2500] [--hang-ms N=700] [--vad-level N=400]\n"
            "                [--listener [--probe-ms N=1500] [--probe-gen N=6] [--probe-suffix TEXT]]\n"
            "  --mic        ffmpeg input spec (default \"-f alsa -i default\" on Linux)\n"
            "  --stdin-pcm  mono 16 kHz s16 PCM on stdin instead of a mic\n"
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
    int whisper = g_whisper_model && *g_whisper_model;
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
    sock_t s = sock_connect(spath);
    if (s == INVALID_SOCKET) { fprintf(stderr, "voicecat: connect to %s failed\n", spath); return 1; }
    if (g_memory) seed_memory(s);

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
    int pending = 0, barge_armed = 1, tstate = 0;    // replies awaited; one barge per utterance
    char rbuf[4096];
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
            deliver(rbuf, k, &pending, &tstate);
        }

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
            int voiced = (int)sqrt((double)ss / FR_SAMP) >= g_vad;

            if (!in_utt) {
                memmove(ring, ring + FR_SAMP, sizeof ring - sizeof frame);   // roll the preroll
                memcpy(ring + (PREROLL - 1) * FR_SAMP, frame, sizeof frame);
                if (ring_n < PREROLL) ring_n++;
                onset = voiced ? onset + 1 : 0;
                if (onset >= ONSET) {                    // an utterance begins
                    if (pending > 0 && barge_armed) {    // ...over the reply: barge
                        uint8_t b = LG_BARGE;
                        send_all(s, &b, 1);
                        barged = 1; barge_armed = 0;
                        fprintf(stderr, "voicecat: barge\n");
                    }
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
                }
            } else if (ub_n >= LG_RATE / 2) {            // native span; drop sub-0.5s blips
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
            }
        }

        if (eof) {
            while (pending > 0) {                        // drain the last replies, then leave
                int k = (int)recv(s, rbuf, sizeof rbuf, 0);
                if (k > 0) deliver(rbuf, k, &pending, &tstate);
                else if (k == 0) break;
                else if (wouldblock()) msleep(10);
                else break;
            }
            break;
        }
    }
    free(ub);
    sock_close(s);
    return 0;
}

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
static int g_commit_ms = 1000, g_hang_ms = 700, g_vad = 400, g_rt = 0;

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

// Transcribe: whisper's plain text minus bracketed tags, whitespace-normalized.
// Returns 0 with the words in `out`, -1 if whisper is missing/failed.
static int whisper_words(const int16_t *pcm, size_t nsamp, char *out, size_t cap) {
    if (cap) out[0] = 0;
    char wav[1024]; temp_wav_path(wav, sizeof wav);
    if (write_wav(wav, pcm, nsamp) != 0) { fprintf(stderr, "voicecat: temp wav write failed (%s)\n", wav); return -1; }
    const char *bin = g_whisper_bin ? g_whisper_bin : "whisper-cli";
    char cmd[4096];
    // the extra outer quotes are for cmd.exe: a /c string that STARTS with a
    // quote gets its first and last quote stripped, mangling every path inside
    snprintf(cmd, sizeof cmd,
#ifdef _WIN32
             "\"\"%s\" -m \"%s\" -f \"%s\" -nt -np 2>NUL\"",
#else
             "\"%s\" -m \"%s\" -f \"%s\" -nt -np 2>/dev/null",
#endif
             bin, g_whisper_model, wav);
    FILE *f = run_pipe(cmd);
    if (!f) { remove(wav); fprintf(stderr, "voicecat: whisper spawn failed\n"); return -1; }
    char raw[8192]; size_t n = fread(raw, 1, sizeof raw - 1, f); raw[n] = 0;
    int rc = pclose(f);
    remove(wav);
    if (n == 0 && rc != 0) { fprintf(stderr, "voicecat: whisper failed (rc %d) — is it on PATH?\n", rc); return -1; }
    size_t w = 0;
    for (size_t i = 0; raw[i]; ) {                       // strip [..] (..) groups, keep the words
        char c = raw[i];
        if (c == '[' || c == '(') {
            char close = c == '[' ? ']' : ')';
            while (raw[i] && raw[i] != close) i++;
            if (raw[i]) i++;
            continue;
        }
        char cc = (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
        if (!(cc == ' ' && (w == 0 || out[w - 1] == ' ')) && w + 1 < cap) { out[w++] = cc; out[w] = 0; }
        i++;
    }
    while (w && out[w - 1] == ' ') out[--w] = 0;
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
        else if (!strcmp(argv[i], "--rt"))                            g_rt = 1;
        else if (!strcmp(argv[i], "--whisper-model") && i + 1 < argc) g_whisper_model = argv[++i];
        else if (!strcmp(argv[i], "--whisper-bin") && i + 1 < argc)   g_whisper_bin = argv[++i];
        else if (!strcmp(argv[i], "--barge-note") && i + 1 < argc)    g_barge_note = argv[++i];
        else if (!strcmp(argv[i], "--commit-ms") && i + 1 < argc)     g_commit_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hang-ms") && i + 1 < argc)       g_hang_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vad-level") && i + 1 < argc)     g_vad = atoi(argv[++i]);
        else if (!spath)                                              spath = argv[i];
        else { fprintf(stderr, "voicecat: unknown argument %s\n", argv[i]); return 1; }
    }
    if (!spath) {
        fprintf(stderr,
            "usage: voicecat <socket> [--mic FFMPEG_INPUT | --stdin-pcm] [--rt]\n"
            "                [--whisper-model FILE] [--whisper-bin PATH] [--barge-note TEXT]\n"
            "                [--commit-ms N=1000] [--hang-ms N=700] [--vad-level N=400]\n"
            "  --mic        ffmpeg input spec (default \"-f alsa -i default\" on Linux)\n"
            "  --stdin-pcm  mono 16 kHz s16 PCM on stdin instead of a mic (--rt paces it live)\n"
            "  no whisper model -> whole utterances as native audio spans (vision-free sessions only)\n");
        return 1;
    }
    if (!g_whisper_model) g_whisper_model = getenv("LG_WHISPER_MODEL");
    if (!g_whisper_bin)   g_whisper_bin   = getenv("LG_WHISPER_BIN");
    int whisper = g_whisper_model && *g_whisper_model;
    if (!whisper)
        fprintf(stderr, "voicecat: no whisper model — native audio spans "
                        "(the model's ears work only while the session is vision-free)\n");

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
    struct sockaddr_un sa; memset(&sa, 0, sizeof sa); sa.sun_family = AF_UNIX;
    if (sock_init() != 0 || strlen(spath) >= sizeof sa.sun_path) { fprintf(stderr, "voicecat: socket setup failed\n"); return 1; }
    strcpy(sa.sun_path, spath);
    sock_t s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET || connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "voicecat: connect to %s failed\n", spath); return 1;
    }
#ifdef _WIN32
    u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif

    int16_t frame[FR_SAMP], ring[PREROLL * FR_SAMP];
    int ring_n = 0;                                  // valid ticks in the preroll ring (rolls)
    int16_t *ub = NULL; size_t ub_cap = 0, ub_n = 0; // the growing utterance
    char prev[8192] = "", cur[8192];                 // last two whisper passes
    int committed = 0;                               // words already sent as 'T'
    size_t last_pass = 0;                            // ub_n at the previous whisper pass
    int in_utt = 0, onset = 0, sil_ms = 0, turn_open = 0, barged = 0;
    int pending = 0, barge_armed = 1, tstate = 0;    // replies awaited; one barge per utterance
    char rbuf[4096];

    fprintf(stderr, "voicecat: %s, listening\n", whisper ? "streaming transcripts" : "native audio spans");
    for (;;) {
        size_t got = fread(frame, 2, FR_SAMP, src);
        int eof = got < FR_SAMP;
        if (g_rt && !eof) msleep(FR_MS);
        // drain any reply bytes; each completed turn drops `pending`
        for (;;) {
            int k = (int)recv(s, rbuf, sizeof rbuf, 0);
            if (k <= 0) break;
            fwrite(rbuf, 1, (size_t)k, stdout);
            int done = turn_ends(rbuf, k, &tstate);
            while (done-- > 0) { pending--; putchar('\n'); }
            fflush(stdout);
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
                    in_utt = 1; sil_ms = 0; onset = 0;
                    committed = 0; prev[0] = 0; last_pass = 0; turn_open = 0;
                }
            } else {
                if (ub_n + FR_SAMP > ub_cap) { ub_cap *= 2; ub = realloc(ub, ub_cap * 2); }
                memcpy(ub + ub_n, frame, sizeof frame); ub_n += FR_SAMP;
                sil_ms = voiced ? 0 : sil_ms + FR_MS;
                // mid-utterance whisper pass: commit what two passes agree on
                if (whisper && ub_n - last_pass >= (size_t)g_commit_ms * (LG_RATE / 1000)) {
                    last_pass = ub_n;
                    if (whisper_words(ub, ub_n, cur, sizeof cur) == 0) {
                        int agree = agree_words(prev, cur);
                        if (agree > committed) {
                            size_t a = after_word(cur, committed), b = after_word(cur, agree);
                            while (cur[a] == ' ') a++;
                            char piece[8192];
                            int m = snprintf(piece, sizeof piece, "%s%.*s ",
                                             !turn_open && barged ? g_barge_note : "", (int)(b - a), cur + a);
                            if (m > (int)sizeof piece - 1) m = (int)sizeof piece - 1;
                            send_frame(s, LG_FRAME_TEXT, piece, (uint32_t)m);
                            if (!turn_open && barged) barged = 0;
                            turn_open = 1;
                            fprintf(stderr, "voicecat: +%d words committed\n", agree - committed);
                            committed = agree;
                        }
                        strcpy(prev, cur);
                    }
                }
            }
        }

        // end of the utterance: trailing silence, or the source ran dry
        if (in_utt && (sil_ms >= g_hang_ms || eof)) {
            in_utt = 0;
            if (whisper) {
                char line[8192] = "";
                if (whisper_words(ub, ub_n, cur, sizeof cur) == 0) {
                    size_t a = after_word(cur, committed);
                    while (cur[a] == ' ') a++;
                    snprintf(line, sizeof line, "%s%s", !turn_open && barged ? g_barge_note : "", cur + a);
                }
                if (turn_open || line[0]) {              // silence/false trigger sends nothing
                    size_t L = strlen(line);
                    if (L > sizeof line - 2) L = sizeof line - 2;
                    line[L] = '\n'; line[L + 1] = 0;
                    send_all(s, line, L + 1);
                    pending++; barge_armed = 1; barged = 0;
                    fprintf(stderr, "voicecat: turn closed (%d+ words)\n", committed);
                }
            } else if (ub_n >= LG_RATE / 2) {            // native span; drop sub-0.5s blips
                size_t pad = (LG_FRAME - ub_n % LG_FRAME) % LG_FRAME;
                if (ub_n + pad > ub_cap) { ub_cap = ub_n + pad; ub = realloc(ub, ub_cap * 2); }
                memset(ub + ub_n, 0, pad * 2);
                if (barged) { send_frame(s, LG_FRAME_TEXT, g_barge_note, (uint32_t)strlen(g_barge_note)); barged = 0; }
                send_frame(s, LG_FRAME_AUDIO, ub, (uint32_t)((ub_n + pad) * 2));
                send_all(s, "\n", 1);
                pending++; barge_armed = 1;
                fprintf(stderr, "voicecat: turn closed (%.1fs audio)\n", (double)ub_n / LG_RATE);
            }
        }

        if (eof) {
            while (pending > 0) {                        // drain the last replies, then leave
                int k = (int)recv(s, rbuf, sizeof rbuf, 0);
                if (k > 0) {
                    fwrite(rbuf, 1, (size_t)k, stdout);
                    int done = turn_ends(rbuf, k, &tstate);
                    while (done-- > 0) { pending--; putchar('\n'); }
                    fflush(stdout);
                } else if (k == 0) break;
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

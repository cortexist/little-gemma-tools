// mmcat — multimodal cat for little-gemma. The successor to the runner's media_cat:
// instead of per-modality switches (-i image, -a audio, -f frame), you just give it
// files and it figures out what each one is. It probes every file with ffprobe,
// decodes with ffmpeg (so it handles whatever ffmpeg handles — mp4, mkv, webm, jpg,
// png, wav, m4a, …), and streams little-gemma's media frames over the runner socket:
//
//     mmcat <socket> [flags] file... ["question"]
//
//   - a still image          -> one image frame (resized to the model's patch grid)
//   - a video (incl. its mp4 soundtrack) -> timestamped frames at --fps + an audio span
//   - an audio file          -> one audio span (mono 16 kHz)
//   - any non-file argument   -> the question (the turn's text)
//
// mmcat owns ADAPTING the input to fit the downstream: it auto-picks a low per-frame
// token budget for video, and uniformly subsamples frames so a turn never exceeds the
// runner's LG_MAX_SEG span cap (a video frame costs 2 spans: its timestamp + image).
// It throttles to fit; making long video / live streams span multiple turns is the
// application's job (stitched as one conversation via skills).
//
// Flags: -nv/--no-video, -na/--no-audio, --fps N (default 1), -n BUDGET (tokens/frame),
//        --whisper-model FILE (enable non-speech captions; see below),
//        --pre TEXT (a preamble before the frames — e.g. metadata a skill wants the model to see),
//        --no-meta (suppress the auto " downsampled video" tag mmcat prepends to video).
//
// For VIDEO, mmcat auto-prepends a ~4-token " downsampled video" tag so the model frames the
// input as a (downsampled) video rather than disconnected stills — Gemma only ever sees sampled
// frames, so this is honest, not a trick. Suppressed by --no-meta or by giving your own --pre.
//
// AUDIO is hybrid. Gemma's audio tower is a SPEECH (ASR) encoder: it transcribes spoken
// words, but it is DEAF to music/ambient sound — fed those, it confabulates (describes
// whatever instrument the prompt names). So when --whisper-model is set, mmcat runs
// whisper (which DOES tag non-speech, e.g. "[MUSIC PLAYING]") and: for a clip with
// speech, still sends the raw audio (the model's own ears) plus a caption for any music
// under it; for a pure non-speech clip, sends ONLY the caption (no audio embedding — it
// would just make the model hallucinate). With no whisper model it sends raw audio as before.
//
// v1 shells out to the ffmpeg/ffprobe/whisper-cli CLIs over pipes — no temp files for
// media; whisper needs a small temp wav (whisper-cli can't read stdin). The planned
// upgrade links libavformat/libavcodec to demux+decode in-process — see README.
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
typedef SOCKET sock_t;
#define sock_close closesocket
#define SHUT_WR SD_SEND
#define popen  _popen
#define pclose _pclose
#define file_ok(p) (_access((p), 4) == 0)
static int sock_init(void) { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w); }
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
#define file_ok(p) (access((p), R_OK) == 0)
static int sock_init(void) { signal(SIGPIPE, SIG_IGN); return 0; }
#endif

#include "lg_media_proto.h"

#define VIDEO_TOK 70   // default per-frame budget for video (frames multiply; keep it small)
static int g_budget = LG_MAX_TOK, g_budget_set = 0;
static double g_fps = 1.0;
static int g_no_video = 0, g_no_audio = 0;
static int g_no_whisper = 0;
static const char *g_whisper_bin = NULL;     // whisper-cli to shell to (default: "whisper-cli" on PATH)
static const char *g_whisper_model = NULL;   // setting a model (flag or LG_WHISPER_MODEL) turns captions ON
static char g_audio_note[4096] = "";         // transcript notes, fused onto the question (not a span)
static int g_turn_video = 0;                 // any file this turn sends frames (see the audio policy)
static const char *g_pre = NULL;             // optional preamble text, sent as a span BEFORE the frames (--pre)
static int g_no_meta = 0;                     // auto " downsampled video" tag before video frames (on unless --no-meta / --pre)
static int g_segs = 0;                       // spans sent this turn; must stay <= LG_MAX_SEG
static int room_for(int n) { return g_segs + n <= LG_MAX_SEG; }
static int whisper_on(void) { return g_whisper_model && *g_whisper_model && !g_no_whisper; }

// ---- geometry: resized (W,H), each side a PATCH multiple, <= budget patches,
//      aspect preserved. Byte-identical to the runner's media_cat target_size. ----
static void target_size(int w, int h, int budget, int *ow, int *oh) {
    const float fa = (float)LG_PATCH;
    const long long max_px = (long long)budget * LG_PATCH * LG_PATCH;
    int wb = (int)roundf((float)w / fa) * LG_PATCH, hb = (int)roundf((float)h / fa) * LG_PATCH;
    if (wb < LG_PATCH) wb = LG_PATCH;
    if (hb < LG_PATCH) hb = LG_PATCH;
    if ((long long)wb * hb > max_px) {
        float beta = sqrtf((float)h * (float)w / (float)max_px);
        wb = (int)floorf((float)w / beta / fa) * LG_PATCH;
        hb = (int)floorf((float)h / beta / fa) * LG_PATCH;
        if (wb < LG_PATCH) wb = LG_PATCH;
        if (hb < LG_PATCH) hb = LG_PATCH;
    }
    *ow = wb; *oh = hb;
}

// ---- the wire ----------------------------------------------------------------
static int send_all(sock_t s, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        int k = (int)send(s, p, (int)(n > (1 << 20) ? (1 << 20) : n), 0);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}
static int send_frame(sock_t s, uint8_t type, int w, int h, const void *payload, uint32_t len) {
    uint8_t hdr[LG_FRAME_HDR] = { LG_FRAME_MAGIC, type };
    uint16_t w16 = (uint16_t)w, h16 = (uint16_t)h;
    memcpy(hdr + 2, &w16, 2); memcpy(hdr + 4, &h16, 2); memcpy(hdr + 6, &len, 4);
    if (send_all(s, hdr, sizeof hdr) != 0) return -1;
    if (len && send_all(s, payload, len) != 0) return -1;
    g_segs++;
    return 0;
}

// ---- ffmpeg/ffprobe plumbing -------------------------------------------------
static FILE *run_pipe(const char *cmd) {
#ifdef _WIN32
    return popen(cmd, "rb");
#else
    return popen(cmd, "r");
#endif
}
static size_t read_full(FILE *f, void *buf, size_t n) {
    size_t got = 0, k;
    while (got < n && (k = fread((char *)buf + got, 1, n - got, f)) > 0) got += k;
    return got;
}

struct probe { int has_video, has_audio, vw, vh; double duration; };

static int probe_file(const char *path, struct probe *p) {
    memset(p, 0, sizeof *p);
    char cmd[4096], line[256];
    snprintf(cmd, sizeof cmd,
        "ffprobe -v error -show_entries stream=codec_type,width,height -of csv=p=0 \"%s\"", path);
    FILE *f = run_pipe(cmd);
    if (!f) { fprintf(stderr, "mmcat: ffprobe failed (is ffmpeg on PATH?)\n"); return -1; }
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "video", 5)) {
            p->has_video = 1;
            char *c1 = strchr(line, ','); if (c1) { p->vw = atoi(c1 + 1); char *c2 = strchr(c1 + 1, ','); if (c2) p->vh = atoi(c2 + 1); }
        } else if (!strncmp(line, "audio", 5)) {
            p->has_audio = 1;
        }
    }
    pclose(f);
    snprintf(cmd, sizeof cmd, "ffprobe -v error -show_entries format=duration -of csv=p=0 \"%s\"", path);
    f = run_pipe(cmd);
    if (f) { if (fgets(line, sizeof line, f)) p->duration = atof(line); pclose(f); }
    return 0;
}

static void timestamp(int sec, char *out, size_t n) {
    if (sec >= 3600) snprintf(out, n, " %d:%02d:%02d ", sec / 3600, (sec / 60) % 60, sec % 60);
    else             snprintf(out, n, " %02d:%02d ", sec / 60, sec % 60);
}

// video / still image -> image frames. is_video: emit a per-frame timestamp. Stops
// early if the turn's span cap is reached (each video frame needs 2 free spans).
static int send_visual(sock_t s, const char *path, int vw, int vh, int is_video, double fps, int budget) {
    int dw, dh; target_size(vw, vh, budget, &dw, &dh);
    size_t fbytes = (size_t)3 * dw * dh;
    uint8_t *rgb = malloc(fbytes);
    if (!rgb) return -1;
    char cmd[4096];
    if (is_video)
        snprintf(cmd, sizeof cmd, "ffmpeg -v error -i \"%s\" -vf fps=%g,scale=%d:%d -f rawvideo -pix_fmt rgb24 -",
                 path, fps, dw, dh);
    else
        snprintf(cmd, sizeof cmd, "ffmpeg -v error -i \"%s\" -vf scale=%d:%d -frames:v 1 -f rawvideo -pix_fmt rgb24 -",
                 path, dw, dh);
    FILE *f = run_pipe(cmd);
    if (!f) { free(rgb); fprintf(stderr, "mmcat: ffmpeg decode failed for %s\n", path); return -1; }
    int idx = 0, rc = 0, capped = 0;
    while (read_full(f, rgb, fbytes) == fbytes) {
        if (!room_for(is_video ? 2 : 1)) { capped = 1; break; }   // never overflow the runner's cap
        if (is_video) {
            char ts[32]; timestamp((int)lround(idx / fps), ts, sizeof ts);
            if (send_frame(s, LG_FRAME_TEXT, 0, 0, ts, (uint32_t)strlen(ts)) != 0) { rc = -1; break; }
        }
        if (send_frame(s, LG_FRAME_IMAGE, dw, dh, rgb, (uint32_t)fbytes) != 0) { rc = -1; break; }
        idx++;
    }
    pclose(f); free(rgb);   // pclose first: closing our read end SIGPIPEs ffmpeg cleanly if we stopped early
    fprintf(stderr, "mmcat: %s -> %d frame(s) %dx%d (%d tok/frame)%s\n",
            path, idx, dw, dh, (dw / LG_PATCH) * (dh / LG_PATCH), capped ? " [turn span cap reached]" : "");
    return rc;
}

// decode any audio file -> padded mono 16 kHz s16 PCM (caller frees *pcm_out).
static int decode_audio(const char *path, int16_t **pcm_out, size_t *nsamp_out) {
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "ffmpeg -v error -i \"%s\" -ac 1 -ar %d -f s16le -", path, LG_RATE);
    FILE *f = run_pipe(cmd);
    if (!f) { fprintf(stderr, "mmcat: ffmpeg audio decode failed for %s\n", path); return -1; }
    size_t cap = 1 << 20, n = 0, k; int16_t *pcm = malloc(cap * 2);
    if (!pcm) { pclose(f); return -1; }
    int16_t chunk[4096];
    while ((k = fread(chunk, 2, 4096, f)) > 0) {
        if (n + k > cap) { cap *= 2; int16_t *t = realloc(pcm, cap * 2); if (!t) { free(pcm); pclose(f); return -1; } pcm = t; }
        memcpy(pcm + n, chunk, k * 2); n += k;
    }
    pclose(f);
    size_t npad = (n + LG_FRAME - 1) / LG_FRAME * LG_FRAME;   // whole frames; runner never sees a tail
    if (npad > n) { int16_t *t = realloc(pcm, npad * 2); if (t) { pcm = t; memset(pcm + n, 0, (npad - n) * 2); } else npad = n; }
    *pcm_out = pcm; *nsamp_out = npad;
    return 0;
}

static int send_audio_frame(sock_t s, const char *path, const int16_t *pcm, size_t npad) {
    int rc = send_frame(s, LG_FRAME_AUDIO, 0, 0, pcm, (uint32_t)(2 * npad));
    fprintf(stderr, "mmcat: %s -> %.2fs audio (%zu tok)\n", path, npad / (double)LG_RATE, npad / LG_FRAME);
    return rc;
}

// ---- whisper: caption the non-speech audio Gemma is deaf to --------------------
// Gemma's audio tower transcribes SPEECH but cannot perceive music/ambient (it
// confabulates on them). whisper DOES tag non-speech ("[MUSIC PLAYING]", "(soft
// piano music)"), so we run it and inject the tag as a text caption — speech still
// goes through the model's own ears; whisper only fills the gap.
static int tag_is_noise(const char *s) {   // whisper's "no useful content" markers
    char low[64]; size_t n = 0;
    for (size_t i = 0; s[i] && n < sizeof low - 1; i++)
        if (isalpha((unsigned char)s[i])) low[n++] = (char)tolower((unsigned char)s[i]);
    low[n] = 0;
    return !n || strstr(low, "blank") || strstr(low, "silence") || strstr(low, "inaudible");
}

static void temp_wav_path(char *buf, size_t n) {   // whisper-cli reads files only, not stdin
#ifdef _WIN32
    char dir[MAX_PATH]; DWORD d = GetTempPathA((DWORD)sizeof dir, dir);
    if (!d || d > sizeof dir) strcpy(dir, ".\\");
    snprintf(buf, n, "%smmcat_w_%lu.wav", dir, (unsigned long)GetCurrentProcessId());
#else
    const char *t = getenv("TMPDIR"); if (!t || !*t) t = "/tmp";
    snprintf(buf, n, "%s/mmcat_w_%ld.wav", t, (long)getpid());
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

// run whisper on the PCM; collect non-speech tags into `tags` and the spoken words
// into `words` (whitespace-normalized), set *has_speech when real words remain.
// Returns 0 on success, -1 if whisper is missing/failed.
static int whisper_caption(const int16_t *pcm, size_t nsamp, char *tags, size_t gcap,
                           char *words, size_t wcap, int *has_speech) {
    *has_speech = 0; if (gcap) tags[0] = 0; if (wcap) words[0] = 0;
    char wav[1024]; temp_wav_path(wav, sizeof wav);
    if (write_wav(wav, pcm, nsamp) != 0) return -1;
    const char *bin = g_whisper_bin ? g_whisper_bin : "whisper-cli";
    char cmd[8192];
    snprintf(cmd, sizeof cmd, "\"%s\" -m \"%s\" -f \"%s\" -nt -np 2>%s", bin, g_whisper_model, wav,
#ifdef _WIN32
             "NUL"
#else
             "/dev/null"
#endif
             );
    FILE *f = run_pipe(cmd);
    if (!f) { remove(wav); return -1; }
    char out[8192]; size_t n = fread(out, 1, sizeof out - 1, f); out[n] = 0;
    int rc = pclose(f);
    remove(wav);
    if (n == 0 && rc != 0) return -1;   // command-not-found / failure -> caller falls back to native
    size_t g = 0, w = 0; int speech = 0;   // split bracketed/parenthesized groups (tags) from bare words (speech)
    for (size_t i = 0; out[i]; ) {
        char c = out[i];
        if (c == '[' || c == '(') {
            char close = c == '[' ? ']' : ')', tag[256]; size_t j = i + 1, t = 0;
            while (out[j] && out[j] != close && t < sizeof tag - 1) tag[t++] = out[j++];
            tag[t] = 0; if (out[j]) j++; i = j;
            if (!tag_is_noise(tag)) {
                if (g && g + 3 < gcap) { memcpy(tags + g, " | ", 3); g += 3; }
                for (size_t k = 0; tag[k] && g + 1 < gcap; k++) tags[g++] = tag[k];
                if (g < gcap) tags[g] = 0;
            }
        } else {
            if (isalnum((unsigned char)c)) speech = 1;
            char cc = (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
            if (!(cc == ' ' && (w == 0 || words[w - 1] == ' ')) && w + 1 < wcap)
                { words[w++] = cc; words[w] = 0; }
            i++;
        }
    }
    while (w && words[w - 1] == ' ') words[--w] = 0;
    *has_speech = speech;
    return 0;
}

// Two things make a non-speech caption actually land, both proven the hard way:
//  (1) frame it as a "transcript" — the model trusts an audio transcript as provided
//      ground truth and relays it, even against a yes/no "can you hear?"; a bare
//      "(background audio: ...)" reads like a claim it should verify by hearing and
//      its "I'm a text AI, I can't hear" prior dismisses it.
//  (2) FUSE it onto the question, not a separate span — as its own media span the
//      model treats "can you hear anything" as a standalone capability question and
//      denies, ignoring the span; in one text blob with the question it relays it.
static void note_caption(const char *tag) {
    char low[256]; size_t i = 0;
    for (; tag[i] && i < sizeof low - 1; i++) low[i] = (char)tolower((unsigned char)tag[i]);
    low[i] = 0;
    size_t n = strlen(g_audio_note);
    snprintf(g_audio_note + n, sizeof g_audio_note - n, "Audio transcript of the provided clip: [%s]. ", low);
}

// Speech heard under video: the model cannot use its own ears once frames share
// the context — measured 2026-07-02 on the 12B, every arrangement (audio before,
// after, interleaved, the model card's frames/text/audio order, a framing tag,
// even black frames or frames in an EARLIER turn): the audio confabulates into
// the visual narrative, or the model denies hearing outright. Audio-only turns
// hear fine. So under video the soundtrack's SPEECH rides the same transcript
// trick as non-speech tags — whisper's words, fused onto the question.
static void note_speech(const char *words) {
    size_t n = strlen(g_audio_note);
    snprintf(g_audio_note + n, sizeof g_audio_note - n, "Audio transcript of the provided clip: \"%s\". ", words);
}

static int process_file(sock_t s, const char *path) {
    struct probe p;
    if (probe_file(path, &p) != 0) return -1;
    if (!p.has_video && !p.has_audio) { fprintf(stderr, "mmcat: %s: no decodable media stream\n", path); return -1; }
    int is_video = p.has_video && p.duration > 0.0;   // a still image reports no/zero duration

    if (p.has_video && !g_no_video) {
        int budget = (is_video && !g_budget_set) ? VIDEO_TOK : g_budget;   // auto-low for video
        double fps = g_fps;
        if (is_video && !g_pre && !g_no_meta) {        // ~4-token honest tag: model frames it as (downsampled) video, not stills
            static const char meta[] = " downsampled video\n";
            send_frame(s, LG_FRAME_TEXT, 0, 0, meta, (uint32_t)(sizeof meta - 1));
        }
        if (is_video) {
            // subsample frames (2 spans each) uniformly across the whole clip to
            // fit the turn (no span reserved for audio: with frames in the turn
            // the soundtrack travels as a transcript note, never as a span).
            int max_frames = (LG_MAX_SEG - g_segs) / 2;
            if (max_frames < 1) max_frames = 1;
            int n_req = (int)(p.duration * fps + 0.5);
            if (n_req > max_frames && p.duration > 0.0) {
                fps = (double)max_frames / p.duration;
                fprintf(stderr, "mmcat: %s is %.0fs — %d frames @ %g fps over the %d-span turn limit; "
                                "sampling ~%d frames (%.3g fps). [longer media -> split across turns]\n",
                        path, p.duration, n_req, g_fps, LG_MAX_SEG, max_frames, fps);
            }
        }
        if (send_visual(s, path, p.vw ? p.vw : LG_PATCH, p.vh ? p.vh : LG_PATCH, is_video, fps, budget) != 0) return -1;
    }
    if (p.has_audio && !g_no_audio) {
        int16_t *pcm = NULL; size_t npad = 0;
        if (decode_audio(path, &pcm, &npad) != 0) return -1;
        char tags[1024] = "", words[4096] = ""; int has_speech = 1;
        if (whisper_on() && whisper_caption(pcm, npad, tags, sizeof tags, words, sizeof words, &has_speech) == 0) {
            if (has_speech && !g_turn_video) {                 // audio-only turn: native ears, note any music under it
                if (room_for(1)) send_audio_frame(s, path, pcm, npad);
                if (tags[0]) { note_caption(tags); fprintf(stderr, "mmcat: %s -> speech + background [%s]\n", path, tags); }
            } else if (has_speech) {                           // frames in the turn: transcript, never a span (see note_speech)
                note_speech(words);
                if (tags[0]) note_caption(tags);
                fprintf(stderr, "mmcat: %s -> video+speech; soundtrack fused as a transcript "
                                "(the model cannot hear once frames share the context)\n", path);
            } else if (tags[0]) {                              // non-speech only: caption, no embedding (model can't hear music)
                note_caption(tags);
                fprintf(stderr, "mmcat: %s -> non-speech, captioned as transcript [%s]; no embedding\n", path, tags);
            } else {
                fprintf(stderr, "mmcat: %s -> audio silent/blank, skipped\n", path);
            }
        } else if (g_turn_video) {                             // no whisper: a poisoned span helps nobody
            fprintf(stderr, "mmcat: %s: video+audio in one turn, and the model cannot hear once frames "
                            "share the context (measured; any arrangement) — soundtrack DROPPED. Configure "
                            "whisper (--whisper-model / LG_WHISPER_MODEL) to fuse it as a transcript.\n", path);
        } else if (room_for(1)) {
            send_audio_frame(s, path, pcm, npad);              // audio-only, whisper off: native ears as before
        }
        free(pcm);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <socket> [-nv|--no-video] [-na|--no-audio] [--fps N] [-n budget]\n"
            "             [--whisper-model FILE] [--whisper-bin PATH] [-nw|--no-whisper] file... [\"question\"]\n"
            "  modality is auto-detected per file (ffprobe); no -i/-a switches.\n"
            "  --fps           video frame sample rate (default 1; auto-reduced to fit the turn span cap)\n"
            "  -n              vision tokens per frame (default %d; video auto-defaults to %d)\n"
            "  --whisper-model whisper.cpp model file; ENABLES non-speech captions (or set LG_WHISPER_MODEL).\n"
            "                  Gemma's ears are speech-only; whisper captions music/ambient so the model knows.\n"
            "  --whisper-bin   whisper-cli to run (default: whisper-cli on PATH; or LG_WHISPER_BIN)\n"
            "  -nw             disable whisper even if a model is configured\n"
            "  --pre TEXT      preamble text sent BEFORE the frames in the turn (e.g. metadata:\n"
            "                  format/fps/sample-rate/file-vs-stream); a skill builds it, mmcat just carries it\n"
            "  --no-meta       suppress the automatic ' downsampled video' tag mmcat adds before video frames\n",
            argv[0], LG_MAX_TOK, VIDEO_TOK);
        return 1;
    }
    const char *spath = argv[1], *question = NULL;
    const char *files[256]; int nf = 0;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "-nv") || !strcmp(argv[i], "--no-video")) g_no_video = 1;
        else if (!strcmp(argv[i], "-na") || !strcmp(argv[i], "--no-audio")) g_no_audio = 1;
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) { g_fps = atof(argv[++i]); if (g_fps <= 0) g_fps = 1.0; }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)    { g_budget = atoi(argv[++i]); if (g_budget < 1) g_budget = 1; g_budget_set = 1; }
        else if (!strcmp(argv[i], "-nw") || !strcmp(argv[i], "--no-whisper")) g_no_whisper = 1;
        else if (!strcmp(argv[i], "--whisper-model") && i + 1 < argc) g_whisper_model = argv[++i];
        else if (!strcmp(argv[i], "--whisper-bin") && i + 1 < argc)   g_whisper_bin = argv[++i];
        else if (!strcmp(argv[i], "--pre") && i + 1 < argc)           g_pre = argv[++i];
        else if (!strcmp(argv[i], "--no-meta"))                       g_no_meta = 1;
        else if (file_ok(argv[i])) { if (nf < 256) files[nf++] = argv[i]; }
        else question = argv[i];   // last non-file arg is the turn's text
    }
    if (!g_whisper_model) g_whisper_model = getenv("LG_WHISPER_MODEL");
    if (!g_whisper_bin)   g_whisper_bin   = getenv("LG_WHISPER_BIN");
    if (!nf && !question) { fprintf(stderr, "mmcat: no files and no question given\n"); return 1; }

    struct sockaddr_un sa; memset(&sa, 0, sizeof sa); sa.sun_family = AF_UNIX;
    if (sock_init() != 0 || strlen(spath) >= sizeof sa.sun_path) { fprintf(stderr, "mmcat: socket setup failed\n"); return 1; }
    strcpy(sa.sun_path, spath);
    sock_t s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET || connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "mmcat: connect to %s failed\n", spath); return 1;
    }
    if (g_pre && *g_pre) {                                  // preamble before the media (e.g. metadata: format/fps/rate/file-or-stream)
        char pre[4096];
        snprintf(pre, sizeof pre, "%s\n", g_pre);
        send_frame(s, LG_FRAME_TEXT, 0, 0, pre, (uint32_t)strlen(pre));
    }
    // The audio policy is TURN-scoped (frames poison hearing wherever they sit in
    // the context, even sent after the audio), so scan all files before any is sent.
    if (!g_no_video)
        for (int i = 0; i < nf; i++) {
            struct probe p;
            if (probe_file(files[i], &p) == 0 && p.has_video) { g_turn_video = 1; break; }
        }
    for (int i = 0; i < nf; i++) if (process_file(s, files[i]) != 0) { sock_close(s); return 1; }

    char line[8192];   // fuse any non-speech transcript note in front of the question
    snprintf(line, sizeof line, "%s%s\n", g_audio_note, question ? question : "");
    if (send_all(s, line, strlen(line)) != 0) { fprintf(stderr, "mmcat: send failed\n"); sock_close(s); return 1; }
    shutdown(s, SHUT_WR);

    char buf[4096 + 8]; int hn = 0, k;
    while ((k = (int)recv(s, buf + hn, 4096, 0)) > 0) {
        fwrite(buf + hn, 1, (size_t)k, stdout); fflush(stdout);
        buf[hn + k] = 0;
        if (strstr(buf, "<turn|>")) break;
        int keep = hn + k < 7 ? hn + k : 7;
        memmove(buf, buf + hn + k - keep, (size_t)keep); hn = keep;
    }
    putchar('\n'); sock_close(s);
    return 0;
}

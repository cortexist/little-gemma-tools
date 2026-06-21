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
// Flags: -nv/--no-video, -na/--no-audio, --fps N (default 1), -n BUDGET (tokens/frame).
//
// v1 shells out to the ffmpeg/ffprobe CLI over pipes (no temp files). The planned
// upgrade links libavformat/libavcodec to demux+decode in-process — see README.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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
static int g_segs = 0;                       // spans sent this turn; must stay <= LG_MAX_SEG
static int room_for(int n) { return g_segs + n <= LG_MAX_SEG; }

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

static int send_audio(sock_t s, const char *path) {
    if (!room_for(1)) { fprintf(stderr, "mmcat: no span left in this turn for %s audio — skipped\n", path); return 0; }
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "ffmpeg -v error -i \"%s\" -ac 1 -ar %d -f s16le -", path, LG_RATE);
    FILE *f = run_pipe(cmd);
    if (!f) { fprintf(stderr, "mmcat: ffmpeg audio decode failed for %s\n", path); return -1; }
    size_t cap = 1 << 20, n = 0; int16_t *pcm = malloc(cap * 2); size_t k;
    if (!pcm) { pclose(f); return -1; }
    int16_t chunk[4096];
    while ((k = fread(chunk, 2, 4096, f)) > 0) {
        if (n + k > cap) { cap *= 2; int16_t *t = realloc(pcm, cap * 2); if (!t) { free(pcm); pclose(f); return -1; } pcm = t; }
        memcpy(pcm + n, chunk, k * 2); n += k;
    }
    pclose(f);
    size_t npad = (n + LG_FRAME - 1) / LG_FRAME * LG_FRAME;   // whole frames; runner never sees a tail
    if (npad > n) { int16_t *t = realloc(pcm, npad * 2); if (t) { pcm = t; memset(pcm + n, 0, (npad - n) * 2); } }
    int rc = send_frame(s, LG_FRAME_AUDIO, 0, 0, pcm, (uint32_t)(2 * npad));
    fprintf(stderr, "mmcat: %s -> %.2fs audio (%zu tok)\n", path, npad / (double)LG_RATE, npad / LG_FRAME);
    free(pcm);
    return rc;
}

static int process_file(sock_t s, const char *path) {
    struct probe p;
    if (probe_file(path, &p) != 0) return -1;
    if (!p.has_video && !p.has_audio) { fprintf(stderr, "mmcat: %s: no decodable media stream\n", path); return -1; }
    int is_video = p.has_video && p.duration > 0.0;   // a still image reports no/zero duration

    if (p.has_video && !g_no_video) {
        int budget = (is_video && !g_budget_set) ? VIDEO_TOK : g_budget;   // auto-low for video
        double fps = g_fps;
        if (is_video) {
            // adapt: leave a span for audio if it's coming, then subsample frames
            // (2 spans each) uniformly across the whole clip to fit what's left.
            int reserve = (p.has_audio && !g_no_audio) ? 1 : 0;
            int max_frames = (LG_MAX_SEG - g_segs - reserve) / 2;
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
    if (p.has_audio && !g_no_audio && send_audio(s, path) != 0) return -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <socket> [-nv|--no-video] [-na|--no-audio] [--fps N] [-n budget] file... [\"question\"]\n"
            "  modality is auto-detected per file (ffprobe); no -i/-a switches.\n"
            "  --fps  video frame sample rate (default 1; auto-reduced to fit the turn span cap)\n"
            "  -n     vision tokens per frame (default %d; video auto-defaults to %d)\n",
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
        else if (file_ok(argv[i])) { if (nf < 256) files[nf++] = argv[i]; }
        else question = argv[i];   // last non-file arg is the turn's text
    }
    if (!nf && !question) { fprintf(stderr, "mmcat: no files and no question given\n"); return 1; }

    struct sockaddr_un sa; memset(&sa, 0, sizeof sa); sa.sun_family = AF_UNIX;
    if (sock_init() != 0 || strlen(spath) >= sizeof sa.sun_path) { fprintf(stderr, "mmcat: socket setup failed\n"); return 1; }
    strcpy(sa.sun_path, spath);
    sock_t s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET || connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "mmcat: connect to %s failed\n", spath); return 1;
    }
    for (int i = 0; i < nf; i++) if (process_file(s, files[i]) != 0) { sock_close(s); return 1; }

    char line[8192];
    snprintf(line, sizeof line, "%s\n", question ? question : "");
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

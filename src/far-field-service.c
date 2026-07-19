// far-field-service — the audio authority: ONE process owns both directions
// of the sound card, cancels the speaker out of the microphone where the
// reference actually lives (aec.h seam: webrtc AEC3 today, pass-through when
// hardware cancels on-chip), and hands everyone else clean audio over an
// AF_UNIX socket — no TCP, no localhost, filesystem permissions are the ACL,
// the stance little-gemma has kept from the start. AEC is only the hardest
// of its jobs: with the 4-mic array the same tap stream grows DoA/beam
// metadata frames, which is why the service is named for the field, not
// the filter.
//
//     far-field-service -s PATH [--source S] [--sink K] [--channels N]
//                       [--use-channel K] [--gain-db D] [--no-aec]
//     far-field-service --tap PATH [--raw]     clean (or raw) mic -> stdout
//     far-field-service --speak PATH [--rate N]  stdin PCM -> the speaker
//
// The wire is the house frame ({magic, type, u16 w, u16 h, u32 len} +
// payload, lg_media_proto.h). A client's FIRST frame declares it:
//   'e' ears  (w=1: the raw pre-cancel channel instead of the clean one)
//   'm' mouth (payload "rate=NNNNN\n"; one mouth at a time)
// then a mouth streams 'P' pcm (s16 mono at its declared rate — resampled
// here once, played, and fed to the canceller as reference) and may send
// 'F' to flush; taps just read 'P' frames of 16 kHz s16 mono. A mouth
// hanging up FLUSHES what it queued — killing the speak client IS the
// barge cut, which is exactly how voicecat's --mouth-play uses it.
//
// The pipeline it serves (voicecat unchanged, policy stays client-side):
//     far-field-service --tap /tmp/ff.sock | voicecat /tmp/lg.sock \
//         --stdin-pcm --mouth-synth "piper ... --output-mux --stream" \
//         --mouth-play "far-field-service --speak /tmp/ff.sock"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <fcntl.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pulse/simple.h>
#include <pulse/error.h>

#include "lg_media_proto.h"
#include "aec.h"

#define RATE   16000
#define BLOCK  160                       // 10 ms — the canceller's frame
#define MAXTAP 8
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FF_EARS  'e'
#define FF_MOUTH 'm'
#define FF_CTRL  'c'                     // a control client: duck/restore frames follow
#define FF_PCM   'P'
#define FF_FLUSH 'F'
#define FF_DRAIN 'D'                     // "I'm done": play out, then ack one byte —
                                         // hangup WITHOUT it is a cut (the barge)
#define FF_DUCK  'd'                     // stage-1 barge: playback drops --duck-db,
#define FF_REST  'u'                     // keeps playing; 'u' (or ctrl hangup) restores

static const char *g_source = NULL, *g_sink = NULL;
static int    g_channels = 1, g_use_ch = 0, g_no_aec = 0, g_scene = 0;
static double g_duck_gain = 0.25;            // --duck-db (default 12): stage-1 barge level
static volatile double g_duck = 1.0;         // current target; the duplex loop ramps to it
static double g_gain = 1.0;                  // after the canceller: lifts for VAD/ASR
static double g_pre_gain = 1.0;              // before it: a deaf channel's echo must
                                             // reach AEC3's working range — but stay
                                             // under clipping (clipped echo is
                                             // nonlinear = uncancelable; measured both)

static struct aec *g_aec = NULL;

// ---- taps ---------------------------------------------------------------------
static pthread_mutex_t t_mx = PTHREAD_MUTEX_INITIALIZER;
static struct { int fd; int raw; } g_tap[MAXTAP];
static int g_ntap = 0;

static void tap_add(int fd, int raw) {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    pthread_mutex_lock(&t_mx);
    if (g_ntap < MAXTAP) { g_tap[g_ntap].fd = fd; g_tap[g_ntap].raw = raw; g_ntap++; }
    else close(fd);
    pthread_mutex_unlock(&t_mx);
}

// One 10 ms block to every tap. A slow tap drops blocks (the mic never
// stalls for a reader); a dead one is reaped in place.
static void tap_cast(const int16_t *clean, const int16_t *raw) {
    uint8_t hdr[LG_FRAME_HDR] = { LG_FRAME_MAGIC, FF_PCM };
    uint32_t len = BLOCK * 2;
    memcpy(hdr + 6, &len, 4);
    pthread_mutex_lock(&t_mx);
    for (int i = 0; i < g_ntap; ) {
        const int16_t *pcm = g_tap[i].raw ? raw : clean;
        uint8_t msg[LG_FRAME_HDR + BLOCK * 2];
        memcpy(msg, hdr, LG_FRAME_HDR);
        memcpy(msg + LG_FRAME_HDR, pcm, BLOCK * 2);
        ssize_t k = send(g_tap[i].fd, msg, sizeof msg, MSG_NOSIGNAL);
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { i++; continue; }   // slow tap: drop the block
        if (k != (ssize_t)sizeof msg) {      // dead, or a short write that would desync its stream
            fprintf(stderr, "far-field-service: tap %d dropped (%s)\n",
                    g_tap[i].fd, k < 0 ? strerror(errno) : "short write");
            close(g_tap[i].fd);
            g_tap[i] = g_tap[--g_ntap];
            continue;
        }
        i++;
    }
    pthread_mutex_unlock(&t_mx);
}

// ---- the mouth: queue -> speaker + canceller reference --------------------------
static pthread_mutex_t m_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  m_cv = PTHREAD_COND_INITIALIZER;
static int16_t *m_q = NULL;              // 16 kHz mono, already resampled
static size_t   m_qn = 0, m_qcap = 0;
#define M_QMAX (RATE * 60)               // a minute of queued speech is a bug
static int      g_mouth_fd = -1;         // one mouth at a time
static pa_simple *g_play = NULL;
static pthread_mutex_t p_mx = PTHREAD_MUTEX_INITIALIZER;   // guards g_play use

// The cut clears the QUEUE only — the sink stream is never flushed (a flush
// jumps the playback clock and the canceller's delay lock with it; measured:
// every post-flush utterance leaked its onset). The tail that still plays is
// bounded by the sink buffer: ~100 ms.
static void mouth_flush(void) {
    pthread_mutex_lock(&m_mx);
    m_qn = 0;
    pthread_mutex_unlock(&m_mx);
}

// ---- the duplex loop: the CARD is the metronome ---------------------------------
// One thread, one clock. Each 10 ms capture read drives exactly one render
// block (queue or SILENCE -> canceller reference -> sink) and then the
// matching near block (channel pick -> cancel -> gain -> taps). This 1:1
// interleave on the device's own crystal is what AEC3 assumes; everything
// looser was measured to break its delay lock: a blocking queue wait starves
// the reference after gaps, pa's buffer-fill phase lets writes burst
// (seconds of reference in milliseconds at every stream start), wall-clock
// pacing jitters against fat capture fragments, and pa_simple_flush jumps
// the playback clock — hence also: the cut clears only the queue, and both
// streams run with tight buffer_attr forever.
// ---- acoustic scene: DoA + source tracking (--scene, stage 1) -------------------
// The 4 raw mics (channels 2..5 of the XVF3800 6-ch capture) sit at the corners
// of a 44 mm square (AEC_MIC_ARRAY_GEO: mic0 +x-y, mic1 +x+y, mic2 -x+y, mic3
// -x-y). We cross-correlate opposite pairs for the inter-mic delay, atan2 that
// into a per-window azimuth, and VOTE it into a fading histogram so the stable
// (direct-path) bearings accumulate into peaks while reverberant reflections
// scatter and fade — the peaks are the numbered sources (see scene_process).
// COARSE by nature: at 16 kHz the max delay across 44 mm is ~2 samples, so
// expect ~±20-30° and NO real distance. `str` is a bearing's accumulated
// histogram weight — how persistent/loud, not a range. Azimuth ORIENTATION
// needs a one-time calibration — talk from a known spot and note where it lands;
// swap the pair args below to rotate/flip. No ASR here: this is the pre-ASR
// acoustic map, meant to be tailed live. Measured: front ~0°, mic-left ~+85°,
// mic-right ~-90° (positive = the array's left).
#define SC_WIN   (BLOCK * 3)                  // 30 ms DoA window
#define SC_LAG   3                            // max |lag| in samples (44 mm ~ 2.05 @16k)
#define SC_NTR   8
#define SC_GATE  25.0                         // deg: fold a histogram peak into a nearby track
#define SC_REP   15                           // windows (~0.45 s) between reports
#define SC_BINS  36                           // 10° azimuth bins for the temporal histogram
#define SC_HMIN  0.8                          // floor: min conf^2 for a bin to be a source (quiet room)
#define SC_HRAT  4.0                          // and it must stand this many x above the diffuse median
#define SC_ONSET 3.0                          // dB above the recent trend = an attack (direct path)
#define SC_DECAY 0.98                         // per-window bin decay (~1.5 s fading memory): the
                                              // direct path recurs and accumulates, echoes scatter

struct sctrack { int id; double az, lvl; long last, rep; };

// Bandpass a mic window to ~300-3500 Hz (below the 44 mm spatial-aliasing limit
// ~3.9 kHz, above room rumble): one-pole LP then HP. Broadband transients like
// finger clicks alias above that and gave a garbage fixed azimuth; band-limiting
// to the coherent speech band fixes it and de-weights out-of-band noise.
static void sc_bp(const int16_t *in, double *out, int n) {
    double lp = 0, y = 0, xp = 0;
    for (int i = 0; i < n; i++) {
        lp += 0.75 * ((double)in[i] - lp);       // LP ~3.5 kHz (anti-alias)
        y   = 0.89 * (y + lp - xp);              // HP ~300 Hz (block DC/rumble)
        xp = lp; out[i] = y;
    }
}

// parabolic-interpolated TDOA (samples) between bandpassed a and b, and the
// normalized correlation peak as a confidence — high for a coherent directional
// source, low for diffuse/ambiguous sound (reported as directionless instead).
static double sc_tdoa(const double *a, const double *b, int n, double *conf) {
    double r[2 * SC_LAG + 1], best = -1e18, ea = 0, eb = 0; int bl = 0;
    for (int i = SC_LAG; i < n - SC_LAG; i++) { ea += a[i] * a[i]; eb += b[i] * b[i]; }
    for (int l = -SC_LAG; l <= SC_LAG; l++) {
        double s = 0;
        for (int i = SC_LAG; i < n - SC_LAG; i++) s += a[i] * b[i - l];
        r[l + SC_LAG] = s;
        if (s > best) { best = s; bl = l; }
    }
    *conf = (ea > 0 && eb > 0) ? best / sqrt(ea * eb) : 0;
    if (bl > -SC_LAG && bl < SC_LAG) {         // sub-sample vertex (concave peak)
        double rm = r[bl - 1 + SC_LAG], r0 = r[bl + SC_LAG], rp = r[bl + 1 + SC_LAG];
        double den = rm - 2 * r0 + rp;
        if (den < -1e-6) return bl + 0.5 * (rm - rp) / den;
    }
    return (double)bl;
}

static double sc_adist(double a, double b) { double d = fabs(a - b); return d > 180 ? 360 - d : d; }

// wall-clock HH:MM:SS.mmm so a tailed scene log can be matched to what was
// happening in the room at that instant (spoke here, chair moved there, ...).
static const char *sc_ts(void) {
    static char b[16];
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm; localtime_r(&ts.tv_sec, &tm);
    snprintf(b, sizeof b, "%02d:%02d:%02d.%03ld", tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
    return b;
}

// One 10 ms block of interleaved capture -> accumulate a DoA window, and on a
// full window: level -> activity -> per-window bearing -> VOTE into a fading
// azimuth histogram. The direct path lands in the same bin window after window
// and accumulates; reverberant reflections arrive from scattered directions and
// never build a peak. Every ~0.45 s we read the histogram's peaks as the stable
// sources (match/spawn/age numbered tracks) and log them. This is the temporal
// integration that collapses one real source's echo-spray to one bearing.
// Called from the duplex thread; cheap (a few hundred MACs / 30 ms).
static void scene_process(const int16_t *tick) {
    static int16_t w[4][SC_WIN]; static int wn = 0;
    static double H[SC_BINS]; static double flr = -60, senv = -60;
    static long scwin = 0, rep = 0; static int recent = 0;
    static struct sctrack tr[SC_NTR]; static int ntr = 0, nextid = 1;
    int ch = g_channels;
    for (int i = 0; i < BLOCK && wn < SC_WIN; i++, wn++)
        for (int k = 0; k < 4; k++) w[k][wn] = tick[i * ch + 2 + k];
    if (wn < SC_WIN) return;
    wn = 0; scwin++;
    double e = 0; for (int i = 0; i < SC_WIN; i++) e += (double)w[1][i] * w[1][i];
    double lvl = 10.0 * log10(e / SC_WIN / (32768.0 * 32768.0) + 1e-12);
    // adaptive floor: creep up slowly, fall toward a quieter level — tracks the
    // room ambient without latching onto a single quiet frame's minimum
    flr += (lvl > flr) ? 0.02 : (lvl - flr) * 0.2;
    for (int b = 0; b < SC_BINS; b++) H[b] *= SC_DECAY;   // ~1.5 s fading memory of bearings
    senv += (lvl - senv) * 0.05;                          // slow trend ~= the reverberant floor
    if (lvl > flr + 15.0 && lvl > -58.0) {                // clear, above-ambient activity
        recent = 1;
        // Precedence: localize on the ATTACK — where the direct path leads the
        // reflections — and ignore the sustain, where echoes have filled in and
        // "loudest direction" lies. lvl above its own recent trend = a leading edge.
        if (lvl - senv > SC_ONSET) {
            static double bp[4][SC_WIN];
            for (int k = 0; k < 4; k++) sc_bp(w[k], bp[k], SC_WIN);
            double cx, cy;
            double tx = sc_tdoa(bp[1], bp[2], SC_WIN, &cx);   // +x vs -x
            double ty = sc_tdoa(bp[1], bp[0], SC_WIN, &cy);   // +y vs -y
            double conf = cx < cy ? cx : cy;
            if (conf > 0.0) {                             // vote weighted conf^2: clean attacks
                double az = atan2(ty, tx) * 180.0 / M_PI; // concentrate on the direct-path bearing,
                int b = (int)floor((az + 180.0) / (360.0 / SC_BINS));   // noisy ones scatter thin
                b = b < 0 ? 0 : b >= SC_BINS ? SC_BINS - 1 : b;
                H[b] += conf * conf;
            }
        }
    }
    if (scwin - rep < SC_REP) return;                     // read out every ~0.45 s
    rep = scwin;
    const char *ts = sc_ts();
    // auto-calibrating bar: a real source stands well above the room's OWN diffuse
    // background, so scale the threshold to the median bin instead of a fixed number
    // (portable across dead vs reverberant rooms). The floor keeps a near-empty
    // histogram (quiet room) from promoting noise into a source.
    double tmp[SC_BINS]; memcpy(tmp, H, sizeof tmp);
    for (int i = 1; i < SC_BINS; i++) { double v = tmp[i]; int j = i - 1;
        while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; j--; } tmp[j + 1] = v; }
    double thr = SC_HRAT * tmp[SC_BINS / 2];              // SC_HRAT x the median (diffuse) bin
    if (thr < SC_HMIN) thr = SC_HMIN;
    int any = 0;                                          // did any bearing survive as a peak
    for (int b = 0; b < SC_BINS; b++) {                   // local maxima above threshold = sources
        double hl = H[(b - 1 + SC_BINS) % SC_BINS], hr = H[(b + 1) % SC_BINS];
        if (H[b] < thr || H[b] < hl || H[b] < hr) continue;
        double az = (b + 0.5) * (360.0 / SC_BINS) - 180.0;
        int m = -1; double bd = SC_GATE;
        for (int t = 0; t < ntr; t++) { double d = sc_adist(az, tr[t].az); if (d < bd) { bd = d; m = t; } }
        if (m < 0) {
            if (ntr < SC_NTR) {
                m = ntr++;
                tr[m] = (struct sctrack){ nextid++, az, H[b], scwin, scwin };
                fprintf(stderr, "%s scene: src#%d NEW   az=%+4.0f  str=%.1f\n", ts, tr[m].id, az, H[b]);
            }
        } else {
            tr[m].az = 0.6 * tr[m].az + 0.4 * az; tr[m].lvl = H[b]; tr[m].last = scwin;
            fprintf(stderr, "%s scene: src#%d       az=%+4.0f  str=%.1f\n", ts, tr[m].id, tr[m].az, H[b]);
        }
        any = 1;
    }
    if (!any && recent) {                                 // sound, but no bearing held together —
        int pb = 0;                                       // show the leading sub-threshold bin so we
        for (int b = 1; b < SC_BINS; b++) if (H[b] > H[pb]) pb = b;   // can tell "near miss" from
        fprintf(stderr, "%s scene: diffuse       (top az=%+4.0f str=%.1f, below %.1f)\n",   // "true smear"
                ts, (pb + 0.5) * (360.0 / SC_BINS) - 180.0, H[pb], thr);
    }
    recent = 0;
    for (int t = 0; t < ntr; ) {                          // a bearing gone quiet 2 reports -> GONE
        if (scwin - tr[t].last >= 2 * SC_REP) {
            fprintf(stderr, "%s scene: src#%d GONE  (was az=%+4.0f)\n", ts, tr[t].id, tr[t].az);
            tr[t] = tr[--ntr];
        } else t++;
    }
}

static void *duplex_main(void *arg) {
    (void)arg;
    int err = 0;
    pa_sample_spec cs = { PA_SAMPLE_S16LE, RATE, (uint8_t)g_channels };
    pa_buffer_attr cba;
    memset(&cba, 0xff, sizeof cba);
    cba.fragsize = (uint32_t)(BLOCK * g_channels * 2);   // 10 ms capture fragments
    pa_simple *cap = pa_simple_new(NULL, "far-field-service", PA_STREAM_RECORD,
                                   g_source, "mic", &cs, NULL, &cba, &err);
    if (!cap) { fprintf(stderr, "far-field-service: capture open: %s\n", pa_strerror(err)); exit(1); }
    pa_sample_spec ps = { PA_SAMPLE_S16LE, RATE, 1 };
    pa_buffer_attr pba;
    memset(&pba, 0xff, sizeof pba);
    pba.tlength = RATE / 10 * 2;             // ~100 ms sink buffer: the cut tail's bound
    pba.prebuf = 0;                          // and no fill-before-start burst
    pthread_mutex_lock(&p_mx);
    g_play = pa_simple_new(NULL, "far-field-service", PA_STREAM_PLAYBACK,
                           g_sink, "mouth", &ps, NULL, &pba, &err);
    pthread_mutex_unlock(&p_mx);
    if (!g_play) { fprintf(stderr, "far-field-service: playback open: %s\n", pa_strerror(err)); exit(1); }

    int16_t tick[BLOCK * 8], raw[BLOCK], clean[BLOCK], block[BLOCK];
    // an 80 ms silence cushion before lockstep: with prebuf=0 and strict
    // 1:1 writes the sink hovers at ~one block of buffer and every ≥10 ms
    // scheduling hiccup underruns audibly (measured: constant micro-gap
    // warble); the cushion sets the hover depth, the estimator absorbs
    // the constant shift
    memset(block, 0, sizeof block);
    for (int i = 0; i < 8; i++) {
        if (!g_no_aec) aec_ref(g_aec, block, BLOCK);
        pa_simple_write(g_play, block, sizeof block, &err);
    }
    long ticks = 0;
    for (;;) {
        if (pa_simple_read(cap, tick, (size_t)BLOCK * g_channels * 2, &err) < 0) {
            fprintf(stderr, "far-field-service: capture: %s\n", pa_strerror(err));
            exit(1);
        }
        if (!g_no_aec && ++ticks % 200 == 0) {           // one canceller status line / 2 s
            char st[128];
            aec_stats(g_aec, st, sizeof st);
            fprintf(stderr, "far-field-service: %s\n", st);
        }
        // render leg: one block out, reference first
        int have = 0;
        pthread_mutex_lock(&m_mx);
        if (m_qn >= BLOCK) {
            memcpy(block, m_q, sizeof block);
            m_qn -= BLOCK;
            memmove(m_q, m_q + BLOCK, m_qn * 2);
            have = 1;
        }
        pthread_mutex_unlock(&m_mx);
        if (!have) memset(block, 0, sizeof block);
        // the duck: ramp the gain across the block (a 12 dB step would click)
        // and scale BEFORE the canceller's reference — it must hear exactly
        // what plays, so ducking costs zero re-convergence
        static double dcur = 1.0;
        double dtgt = g_duck;
        if (have && (dcur != 1.0 || dtgt != 1.0)) {
            for (int i = 0; i < BLOCK; i++) {
                double gg = dcur + (dtgt - dcur) * i / BLOCK;
                block[i] = (int16_t)(block[i] * gg);
            }
        }
        dcur = dtgt;
        if (!g_no_aec) aec_ref(g_aec, block, BLOCK);
        pa_simple_write(g_play, block, sizeof block, &err);
        // near leg: pick, pre-gain (bounded), cancel, post-gain
        for (int i = 0; i < BLOCK; i++) {
            double v = tick[i * g_channels + g_use_ch] * g_pre_gain;
            raw[i] = (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
        }
        memcpy(clean, raw, sizeof clean);
        if (!g_no_aec) aec_mic(g_aec, clean, BLOCK);
        if (g_gain != 1.0)
            for (int i = 0; i < BLOCK; i++) {
                double v = clean[i] * g_gain;        // lift the CLEAN signal for VAD/ASR
                clean[i] = (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
            }
        tap_cast(clean, raw);
        if (g_scene) scene_process(tick);            // pre-ASR acoustic map (DoA + tracks)
    }
    return NULL;
}

// ---- per-client: hello, then serve ----------------------------------------------
static int read_full(int fd, void *buf, size_t n) {
    uint8_t *p = buf;
    while (n > 0) {
        ssize_t k = read(fd, p, n);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

// Linear resampler, stateful across frames: in_rate -> 16 kHz.
struct rs { double pos, step; int16_t last; int primed; };
static size_t rs_run(struct rs *r, const int16_t *in, size_t n, int16_t *out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        int16_t cur = in[i];
        if (!r->primed) { r->last = cur; r->primed = 1; }
        while (r->pos < 1.0 && o < cap) {
            out[o++] = (int16_t)(r->last + (cur - r->last) * r->pos);
            r->pos += r->step;
        }
        r->pos -= 1.0;
        r->last = cur;
    }
    return o;
}

static void *client_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    uint8_t hdr[LG_FRAME_HDR];
    if (read_full(fd, hdr, sizeof hdr) != 0 || hdr[0] != LG_FRAME_MAGIC) { close(fd); return NULL; }
    uint16_t w;
    uint32_t len;
    memcpy(&w, hdr + 2, 2);
    memcpy(&len, hdr + 6, 4);

    if (hdr[1] == FF_EARS) {
        if (len) { close(fd); return NULL; }
        tap_add(fd, w == 1);
        return NULL;                                 // capture thread owns it now
    }
    if (hdr[1] == FF_CTRL) {                         // duck/restore frames until hangup;
        while (read_full(fd, hdr, sizeof hdr) == 0 && hdr[0] == LG_FRAME_MAGIC) {
            if (hdr[1] == FF_DUCK) g_duck = g_duck_gain;
            else if (hdr[1] == FF_REST) g_duck = 1.0;
        }
        g_duck = 1.0;                                // a dead controller never leaves it ducked
        close(fd);
        return NULL;
    }
    if (hdr[1] != FF_MOUTH || len > 64) { close(fd); return NULL; }
    char cfg[65] = "";
    if (len && read_full(fd, cfg, len) != 0) { close(fd); return NULL; }
    int in_rate = 0;
    const char *r = strstr(cfg, "rate=");
    if (r) in_rate = atoi(r + 5);
    if (in_rate < 8000 || in_rate > 48000) { close(fd); return NULL; }
    if (__sync_val_compare_and_swap(&g_mouth_fd, -1, fd) != -1) {
        fprintf(stderr, "far-field-service: a second mouth turned away\n");
        close(fd);
        return NULL;
    }
    fprintf(stderr, "far-field-service: mouth connected (%d Hz)\n", in_rate);

    struct rs rs = { 0.0, (double)in_rate / RATE, 0, 0 };
    static int16_t in[4096], out[16384];             // one mouth: static is fine
    for (;;) {
        if (read_full(fd, hdr, sizeof hdr) != 0 || hdr[0] != LG_FRAME_MAGIC) break;
        memcpy(&len, hdr + 6, 4);
        if (hdr[1] == FF_FLUSH) { mouth_flush(); continue; }
        if (hdr[1] == FF_DRAIN) {
            for (;;) {                               // queue empty, then the sink's
                pthread_mutex_lock(&m_mx);           // ~100 ms tail (the stream runs
                size_t left = m_qn;                  // forever — a pa drain would
                pthread_mutex_unlock(&m_mx);         // never return)
                if (left < BLOCK) break;
                usleep(20000);
            }
            usleep(150000);
            uint8_t ok = 1;
            if (write(fd, &ok, 1) != 1) break;
            continue;
        }
        if (hdr[1] != FF_PCM || len > sizeof in || (len & 1)) break;
        size_t n = len / 2, off = 0;
        if (read_full(fd, in, len) != 0) break;
        size_t on = rs_run(&rs, in, n, out, sizeof out / 2);
        while (off < on) {                           // backpressure: the queue caps,
            pthread_mutex_lock(&m_mx);               // the socket read waits
            if (m_qn + (on - off) > M_QMAX) {
                pthread_mutex_unlock(&m_mx);
                usleep(20000);
                continue;
            }
            if (m_qn + on - off > m_qcap) {
                m_qcap = (m_qn + on - off) * 2 + 65536;
                m_q = realloc(m_q, m_qcap * 2);
            }
            memcpy(m_q + m_qn, out + off, (on - off) * 2);
            m_qn += on - off;
            off = on;
            pthread_cond_signal(&m_cv);
            pthread_mutex_unlock(&m_mx);
        }
    }
    mouth_flush();                                   // a mouth hanging up IS the cut
    fprintf(stderr, "far-field-service: mouth gone — queue flushed\n");
    close(fd);
    g_mouth_fd = -1;
    return NULL;
}

// ---- client modes ----------------------------------------------------------------
static int sock_connect(const char *path) {
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof sa.sun_path) return -1;
    strcpy(sa.sun_path, path);
    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) return -1;
    if (connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) { close(s); return -1; }
    return s;
}

static int send_hdr(int s, uint8_t type, uint16_t w, uint32_t len) {
    uint8_t hdr[LG_FRAME_HDR] = { LG_FRAME_MAGIC, type };
    memcpy(hdr + 2, &w, 2);
    memcpy(hdr + 6, &len, 4);
    return write(s, hdr, sizeof hdr) == (ssize_t)sizeof hdr ? 0 : -1;
}

static int run_tap(const char *path, int raw) {
    int s = sock_connect(path);
    if (s < 0) { fprintf(stderr, "far-field-service: connect to %s failed\n", path); return 1; }
    if (send_hdr(s, FF_EARS, raw ? 1 : 0, 0) != 0) return 1;
    setvbuf(stdout, NULL, _IONBF, 0);        // a mic stream must not sit in stdio
    uint8_t hdr[LG_FRAME_HDR];
    static uint8_t buf[1 << 16];
    for (;;) {
        if (read_full(s, hdr, sizeof hdr) != 0) { fprintf(stderr, "far-field-service: tap: server closed the stream\n"); break; }
        uint32_t len;
        memcpy(&len, hdr + 6, 4);
        if (hdr[0] != LG_FRAME_MAGIC || len > sizeof buf) { fprintf(stderr, "far-field-service: tap: bad frame\n"); break; }
        if (read_full(s, buf, len) != 0) { fprintf(stderr, "far-field-service: tap: truncated frame\n"); break; }
        if (hdr[1] == FF_PCM && fwrite(buf, 1, len, stdout) != len) { fprintf(stderr, "far-field-service: tap: stdout closed\n"); break; }
    }
    return 0;
}

static int run_speak(const char *path, int rate) {
    int s = sock_connect(path);
    if (s < 0) { fprintf(stderr, "far-field-service: connect to %s failed\n", path); return 1; }
    char cfg[32];
    int cn = snprintf(cfg, sizeof cfg, "rate=%d\n", rate);
    if (send_hdr(s, FF_MOUTH, 0, (uint32_t)cn) != 0 || write(s, cfg, cn) != cn) return 1;
    static uint8_t buf[4096];
    size_t held = 0;                                     // an odd trailing byte carries over:
    ssize_t k;                                           // 'P' frames are s16-aligned
    while ((k = read(0, buf + held, sizeof buf - held)) > 0) {   // read(): partial chunks
        size_t total = held + (size_t)k;                 // forward NOW, not at 4 KB boundaries
        size_t even = total & ~(size_t)1;
        held = total - even;
        if (even) {
            if (send_hdr(s, FF_PCM, 0, (uint32_t)even) != 0) return 1;
            uint8_t *p = buf;
            size_t n = even;
            while (n > 0) {
                ssize_t m = write(s, p, n);
                if (m <= 0) return 1;
                p += m; n -= (size_t)m;
            }
            if (held) buf[0] = buf[even];
        }
    }
    // a FINISHED mouth drains politely; a KILLED one (the barge) hangs up
    // without this and the server cuts what it queued
    uint8_t ok;
    if (send_hdr(s, FF_DRAIN, 0, 0) == 0)
        (void)!read(s, &ok, 1);
    close(s);
    return 0;
}

// ---- main -------------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *spath = NULL, *tap = NULL, *speak = NULL;
    int raw = 0, rate = 22050;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-s") && i + 1 < argc)             spath = argv[++i];
        else if (!strcmp(argv[i], "--tap") && i + 1 < argc)          tap = argv[++i];
        else if (!strcmp(argv[i], "--speak") && i + 1 < argc)        speak = argv[++i];
        else if (!strcmp(argv[i], "--raw"))                          raw = 1;
        else if (!strcmp(argv[i], "--rate") && i + 1 < argc)         rate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--source") && i + 1 < argc)       g_source = argv[++i];
        else if (!strcmp(argv[i], "--sink") && i + 1 < argc)         g_sink = argv[++i];
        else if (!strcmp(argv[i], "--channels") && i + 1 < argc)     g_channels = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--use-channel") && i + 1 < argc)  g_use_ch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gain-db") && i + 1 < argc)      g_gain = pow(10.0, atof(argv[++i]) / 20.0);
        else if (!strcmp(argv[i], "--pre-gain-db") && i + 1 < argc)  g_pre_gain = pow(10.0, atof(argv[++i]) / 20.0);
        else if (!strcmp(argv[i], "--duck-db") && i + 1 < argc)      g_duck_gain = pow(10.0, -atof(argv[++i]) / 20.0);
        else if (!strcmp(argv[i], "--no-aec"))                       g_no_aec = 1;
        else if (!strcmp(argv[i], "--scene"))                        g_scene = 1;
        else {
            fprintf(stderr, "far-field-service: unknown argument %s\n", argv[i]);
            return 1;
        }
    }
    signal(SIGPIPE, SIG_IGN);
    if (tap)   return run_tap(tap, raw);
    if (speak) return run_speak(speak, rate);
    if (!spath) {
        fprintf(stderr,
            "usage: far-field-service -s PATH [--source S] [--sink K] [--channels N]\n"
            "                         [--use-channel K] [--gain-db D] [--no-aec] [--scene]\n"
            "       far-field-service --tap PATH [--raw]     clean (raw) mic -> stdout\n"
            "       far-field-service --speak PATH [--rate N=22050]  stdin PCM -> speaker\n");
        return 1;
    }
    if (g_use_ch >= g_channels) { fprintf(stderr, "far-field-service: --use-channel out of range\n"); return 1; }
    if (g_scene && g_channels < 6) { fprintf(stderr, "far-field-service: --scene needs the 4 raw mics (run --channels 6)\n"); g_scene = 0; }

    if (!g_no_aec) {
        g_aec = aec_new(RATE);               // which impl = link time (aec.h seam);
        if (!g_aec) {                        // --no-aec = runtime bypass either way
            fprintf(stderr, "far-field-service: canceller init failed — pass-through\n");
            g_no_aec = 1;
        }
    }

    unlink(spath);
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    if (strlen(spath) >= sizeof sa.sun_path) { fprintf(stderr, "far-field-service: path too long\n"); return 1; }
    strcpy(sa.sun_path, spath);
    int ls = socket(AF_UNIX, SOCK_STREAM, 0);
    if (ls < 0 || bind(ls, (struct sockaddr *)&sa, sizeof sa) != 0 || listen(ls, 8) != 0) {
        fprintf(stderr, "far-field-service: bind %s: %s\n", spath, strerror(errno));
        return 1;
    }

    pthread_t duplex;
    pthread_create(&duplex, NULL, duplex_main, NULL);
    fprintf(stderr, "far-field-service: %s on %s (source %s, sink %s)\n",
            aec_name(), spath, g_source ? g_source : "default", g_sink ? g_sink : "default");

    // Prime the canceller: a soft fading hiss through the speaker teaches
    // AEC3 the echo path BEFORE the first reply. Without it the first TTS
    // leaks unconverged echo, the client barges on the leak, the flush
    // starves the reference, and the filter never learns — a measured
    // deadlock. The hiss doubles as the service's ready chime.
    if (!g_no_aec) {
        size_t n = RATE * 8 / 10;            // 0.8 s is plenty to seed the filter
        pthread_mutex_lock(&m_mx);
        if (m_qcap < n) { m_qcap = n; m_q = realloc(m_q, m_qcap * 2); }
        unsigned seed = 1;
        int prev = 0;
        for (size_t i = 0; i < n; i++) {
            seed = seed * 1103515245u + 12345u;
            int v = ((int)((seed >> 16) & 0x7fff) - 16384) / 28;   // ~-35 dBFS
            v = (v + prev * 3) / 4;                                // one-pole lowpass:
            prev = v;                                              // "shh", not static
            m_q[i] = (int16_t)(v * (double)(n - i) / (double)n);   // fading out
        }
        m_qn = n;
        pthread_cond_signal(&m_cv);
        pthread_mutex_unlock(&m_mx);
    }

    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; break; }
        pthread_t th;
        pthread_create(&th, NULL, client_main, (void *)(intptr_t)fd);
        pthread_detach(th);
    }
    return 0;
}

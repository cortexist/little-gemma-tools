// aec-webrtc.cpp — webrtc AEC3 behind the aec.h C seam; the repo's only C++,
// quarantined here. Links the freedesktop webrtc-audio-processing 1.x
// extraction (modules/audio_processing + minimal deps — NOT the WebRTC tree).
// The APM wants 10 ms mono frames: the mic path cleans IN PLACE and relies on
// the service reading capture in block multiples (160 samples at 16 kHz); the
// reference path takes arbitrary lengths and carries partial blocks, since it
// never writes back. High-pass on, noise suppression on, gain control OFF —
// levels are policy upstream (voicecat --gain), not DSP surprises.
#include <string.h>
#include <new>
#include <modules/audio_processing/include/audio_processing.h>

extern "C" {
#include "aec.h"
}

struct aec {
    rtc::scoped_refptr<webrtc::AudioProcessing> apm;
    webrtc::StreamConfig cfg;
    int block;                               // 10 ms of samples
    int16_t rbuf[480];                       // reference partial-block carry
    int rn;
};

extern "C" struct aec *aec_new(int rate) {
    struct aec *a = new (std::nothrow) aec();
    if (!a) return nullptr;
    a->apm = webrtc::AudioProcessingBuilder().Create();
    if (!a->apm) { delete a; return nullptr; }
    webrtc::AudioProcessing::Config c;
    c.echo_canceller.enabled = true;
    c.echo_canceller.mobile_mode = false;
    c.high_pass_filter.enabled = true;
    c.noise_suppression.enabled = true;
    c.gain_controller1.enabled = false;
    c.gain_controller2.enabled = false;
    a->apm->ApplyConfig(c);
    a->cfg = webrtc::StreamConfig(rate, 1);
    a->block = rate / 100;
    a->rn = 0;
    return a;
}

extern "C" void aec_ref(struct aec *a, const int16_t *pcm, int n) {
    while (n > 0) {
        int take = a->block - a->rn;
        if (take > n) take = n;
        memcpy(a->rbuf + a->rn, pcm, (size_t)take * 2);
        a->rn += take; pcm += take; n -= take;
        if (a->rn == a->block) {
            a->apm->ProcessReverseStream(a->rbuf, a->cfg, a->cfg, a->rbuf);
            a->rn = 0;
        }
    }
}

extern "C" void aec_mic(struct aec *a, int16_t *pcm, int n) {
    for (; n >= a->block; pcm += a->block, n -= a->block) {
        a->apm->set_stream_delay_ms(0);      // AEC3's own estimator does the work
        a->apm->ProcessStream(pcm, a->cfg, a->cfg, pcm);
    }
    // a ragged tail passes unprocessed; the service reads block multiples
}

extern "C" void aec_stats(struct aec *a, char *buf, int cap) {
    webrtc::AudioProcessingStats st = a->apm->GetStatistics();
    snprintf(buf, (size_t)cap, "erl %.1f erle %.1f delay %d ms",
             st.echo_return_loss.value_or(-999.0),
             st.echo_return_loss_enhancement.value_or(-999.0),
             st.delay_ms.value_or(-1));
}

extern "C" void aec_free(struct aec *a) { delete a; }
extern "C" const char *aec_name(void) { return "webrtc AEC3"; }

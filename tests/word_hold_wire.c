// Actual mouth protocol over a socketpair, without opening an audio device.
#define main far_field_service_main
#include "../src/far-field-service.c"
#undef main
#include <assert.h>

static void frame(int fd, uint8_t kind, const void *data, uint32_t n) {
    uint8_t h[LG_FRAME_HDR] = {LG_FRAME_MAGIC, kind};
    memcpy(h + 6, &n, 4);
    // Deliberately fragment headers; the reader must assemble complete frames.
    for (size_t i = 0; i < sizeof h; i++) assert(write(fd, h + i, 1) == 1);
    if (n) assert(write(fd, data, n) == n);
}

static void await_queue(size_t n) {
    for (int i = 0; i < 2000; i++) {
        pthread_mutex_lock(&m_mx);
        int ready = m_qn == n;
        pthread_mutex_unlock(&m_mx);
        if (ready) return;
        usleep(1000);
    }
    assert(!"mouth command timeout");
}

int main(void) {
    alarm(10);
    int fd[2];
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fd));
    pthread_t client;
    assert(!pthread_create(&client, NULL, client_main, (void *)(intptr_t)fd[1]));
    frame(fd[0], FF_MOUTH, "rate=16000\n", 11);
    uint32_t grace = 1280;
    frame(fd[0], FF_GRACE, &grace, 4);
    int16_t pcm[4000];
    for (int i = 0; i < 4000; i++) pcm[i] = i + 1;
    frame(fd[0], FF_PCM, pcm, sizeof pcm);
    await_queue(4000);  // PCM is a fence for the preceding control frame.
    pthread_mutex_lock(&m_mx);
    assert(m_hold_requested && m_hold_not_before == 1280 && m_hold_at == UINT64_MAX);
    pthread_mutex_unlock(&m_mx);
    int16_t block[BLOCK];
    for (int i = 0; i < 8; i++) assert(mouth_take_block(block));
    uint8_t control[4 + 3 * 8];
    uint64_t marks[] = {640, 1600, 3200};
    memcpy(control, &grace, 4); memcpy(control + 4, marks, sizeof marks);
    frame(fd[0], FF_GRACE, control, sizeof control);
    frame(fd[0], FF_PCM, pcm, BLOCK * 2);
    await_queue(4000 - 1280 + BLOCK);
    pthread_mutex_lock(&m_mx);
    assert(m_hold_not_before == 1280 && m_hold_at == 1600);
    pthread_mutex_unlock(&m_mx);
    assert(mouth_take_block(block)); assert(mouth_take_block(block));
    assert(!mouth_take_block(block) && m_held && m_rendered == 1600);
    frame(fd[0], FF_RESUME, NULL, 0);
    frame(fd[0], FF_PCM, pcm, BLOCK * 2);
    await_queue(4000 - 1600 + 2 * BLOCK);
    pthread_mutex_lock(&m_mx);
    assert(!m_held && !m_hold_requested && m_hold_at == UINT64_MAX);
    pthread_mutex_unlock(&m_mx);
    assert(mouth_take_block(block));
    // Resampler delays one source sample at equal rates; verify continuity.
    assert(block[0] == 1600 && block[BLOCK - 1] == 1759);
    // A new G followed by R before rendering never inserts a hold.
    frame(fd[0], FF_GRACE, control, sizeof control);
    frame(fd[0], FF_RESUME, NULL, 0);
    frame(fd[0], FF_PCM, pcm, BLOCK * 2);
    await_queue(4000 - 1760 + 3 * BLOCK);
    assert(mouth_take_block(block) && block[0] == 1760);
    shutdown(fd[0], SHUT_RDWR); close(fd[0]);
    pthread_join(client, NULL);
    assert(m_qn == 0 && !m_hold_requested && g_mouth_fd == -1);
    free(m_q);
    puts("mouth G/R transport: grace, late alignment, resume, cancellation, disconnect passed");
    return 0;
}

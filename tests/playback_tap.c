// Exercise the real socket sender: recording must not silently lose samples or
// block the microphone consumer when a recorder stops reading.
#define main far_field_service_main
#include "../src/far-field-service.c"
#undef main
#include <assert.h>

int main(void) {
    int mic[2], output[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, mic) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, output) == 0);
    int size = 1024;
    assert(setsockopt(output[0], SOL_SOCKET, SO_SNDBUF, &size, sizeof size) == 0);
    tap_add(mic[0], 0, 0, 0);
    tap_add(output[0], 0, 0, 1);
    int16_t clean[BLOCK], raw[BLOCK], rendered[BLOCK];
    for (int i = 0; i < BLOCK; i++) { clean[i] = 101; raw[i] = 202; rendered[i] = 303; }
    uint8_t received[LG_FRAME_HDR+BLOCK*2];
    tap_cast(clean, raw, rendered);
    assert(read_full(output[1], received, sizeof received) == 0);
    assert(received[0] == LG_FRAME_MAGIC && received[1] == FF_PCM);
    assert(memcmp(received+LG_FRAME_HDR, rendered, sizeof rendered) == 0);
    assert(read_full(mic[1], received, sizeof received) == 0);
    assert(memcmp(received+LG_FRAME_HDR, clean, sizeof clean) == 0);
    // The output tap never sees raw mic or samples still in the mouth queue.
    // Starve only the recording reader; mic blocks continue without omissions.
    for (int i = 0; i < 1000; i++) {
        clean[0] = (int16_t)i;
        tap_cast(clean, raw, rendered);
        assert(read_full(mic[1], received, sizeof received) == 0);
        assert(memcmp(received+LG_FRAME_HDR, clean, sizeof clean) == 0);
    }
    assert(g_ntap == 1 && g_tap[0].fd == mic[0]);
    while (read(output[1], received, sizeof received) > 0) {}
    close(mic[0]); close(mic[1]); close(output[1]);
    return 0;
}

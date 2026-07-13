/* aec-null.c — the pass-through canceller: for hardware that cancels on-chip
 * (XVF3800), for wiring tests, and for builds without the webrtc library.
 * far-field-service keeps its shape; the seam just stops subtracting. */
#include <stdlib.h>
#include "aec.h"

struct aec { int rate; };

struct aec *aec_new(int rate) {
    struct aec *a = malloc(sizeof *a);
    if (a) a->rate = rate;
    return a;
}
void aec_ref(struct aec *a, const int16_t *pcm, int n) { (void)a; (void)pcm; (void)n; }
void aec_mic(struct aec *a, int16_t *pcm, int n)       { (void)a; (void)pcm; (void)n; }
void aec_stats(struct aec *a, char *buf, int cap)      { (void)a; if (cap) buf[0] = 0; }
void aec_free(struct aec *a) { free(a); }
const char *aec_name(void) { return "null (pass-through)"; }

/* aec.h — the echo-canceller seam inside far-field-service. One mono 16-bit
 * stream each way; implementations buffer to their own internal block size.
 *
 * Implementations are link-time swappable behind this C surface:
 *   aec-webrtc.cpp  webrtc-audio-processing 1.x (AEC3), extern "C" wrapped —
 *                   the only C++ in the repo, quarantined in that one file
 *   aec-null.c      pass-through, for hardware that cancels on-chip (XVF3800)
 *                   or for wiring tests; also the fallback when the webrtc
 *                   library is absent at configure time
 */
#ifndef AEC_H
#define AEC_H

#include <stdint.h>

struct aec;

struct aec *aec_new(int rate);                            /* NULL on failure */
void aec_ref(struct aec *a, const int16_t *pcm, int n);   /* about to be played */
void aec_mic(struct aec *a, int16_t *pcm, int n);         /* cleaned in place */
void aec_stats(struct aec *a, char *buf, int cap);        /* one status line */
void aec_free(struct aec *a);
const char *aec_name(void);                               /* which impl linked */

#endif

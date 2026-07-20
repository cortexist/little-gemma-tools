// lg_media_proto.h — the little-gemma media wire protocol (the contract between
// this tools repo and the runner). Self-contained: a tool depends on THIS, not on
// little-gemma's source tree. Must stay byte-compatible with little-gemma's
// include/media.h frame format.
//
// A media turn is a sequence of typed, length-prefixed frames sent over the
// runner's -s socket, followed by the question as a plain "text\n" line and a
// half-close (shutdown SEND). Each frame is a 10-byte header + raw payload:
//
//   u8  MAGIC (0x01)
//   u8  type  ('I' image | 'A' audio | 'T' text)
//   u16 w, u16 h   (image: pixel dims; audio/text: 0)
//   u32 len        (payload bytes)
//   payload:
//     'I' : w*h*3 u8 RGB, each side a multiple of PATCH (48)
//     'A' : len/2 s16 mono PCM at 16 kHz, padded to a whole number of FRAMEs (640)
//     'T' : len bytes UTF-8 (e.g. a " MM:SS " video timestamp)
//
// The runner wraps each image/audio span in the model's marker tokens itself, so
// a tool sends RAW decoded media — never the markers, never embeddings.
#ifndef LG_MEDIA_PROTO_H
#define LG_MEDIA_PROTO_H

#define LG_FRAME_MAGIC 0x01
#define LG_FRAME_HDR   10
#define LG_FRAME_IMAGE 'I'
#define LG_FRAME_AUDIO 'A'
#define LG_FRAME_TEXT  'T'

// Barge-in: ONE bare byte (no header, no payload) sent while the runner is
// streaming a reply. The runner stops decoding at the next token, closes the
// turn with "<turn|>" on the wire, and the session (and its context, cut-off
// reply included) continues. Sent between turns it is ignored.
#define LG_BARGE       0x02

// Listener probe (duplex prototype): a 'P' frame sent while the user turn is
// still OPEN — "if the turn ended right here, what would you say?". w = max
// tokens to decode (0 -> the runner's default), h = 0, payload (>= 1 byte) =
// a steering suffix appended inside the user turn before the dry-run close.
// The runner answers mid-turn with "<|probe>text<probe|>\n" and rolls the
// context back, so probes never change the conversation. Send only while
// LISTENING (never with a reply pending).
#define LG_FRAME_PROBE 'P'

#define LG_PATCH   48      // pixels per patch side; image dims must be multiples
#define LG_RATE    16000   // audio sample rate (mono s16)
#define LG_FRAME   640     // samples per audio frame (40 ms @ 16 kHz)
#define LG_MAX_TOK 280     // default per-image vision-token budget (a Gemma 4 rung)

// Per-turn span cap the runner enforces (must match run.c MAX_SEG). EVERY frame
// is one span — including the 'T' timestamp a tool sends before each video frame,
// so a video frame costs TWO spans (timestamp + image) and audio costs one. A
// tool must fit within this (subsample frames, etc.); longer media is split
// across turns by the application. (Future: the runner advertises this on connect
// so clients need not hardcode it.)
#define LG_MAX_SEG 256

#endif

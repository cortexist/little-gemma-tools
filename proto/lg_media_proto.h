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

#define LG_PATCH   48      // pixels per patch side; image dims must be multiples
#define LG_RATE    16000   // audio sample rate (mono s16)
#define LG_FRAME   640     // samples per audio frame (40 ms @ 16 kHz)
#define LG_MAX_TOK 280     // default per-image vision-token budget (a Gemma 4 rung)

#endif

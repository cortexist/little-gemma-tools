# vocab-trim — MTP draft-head vocabulary tooling

Tooling for `little-gemma/docs/mtp-vocab-trim.md`: the draft head's tied LM
head carries 262,144 rows (512 MiB f16 on device) to answer a question whose
answer is ~10k distinct ids. Verification is greedy against the target's
*full* vocabulary, so any trim is byte-identity-safe — a token the trimmed
head can't express is just a rejected draft.

Both tools are stdlib-only Python and run the same on x86 and the Orin.

## trim-head.py — prefix-trim a draft GGUF, zero engine changes

```
trim-head.py SRC.gguf OUT.gguf K
```

Keeps rows `0..K-1` of `token_embd.weight` by row-aligned byte surgery
(no dequantize/requantize; row bytes derived from the actual quant type —
Q4_0 = 576 B, Q6_K = 840 B at ne[0]=1024). Draft row index == target token
id, so the **unmodified** engine runs the result: `mtp_open` reads `n_vocab`
from the tensor dims, and the draft GGUF's `token_embd` is only ever used as
the LM head (input embeddings come from the target model). The KV section is
copied byte-verbatim; the output is re-parsed and byte-compared against the
source before the tool reports success.

Phase-1 caveat, measured (2026-08-19): Gemma's vocabulary is NOT
frequency-ordered — `.` is id 236761, `,` is 236764 — so a prefix trim
caps coverage hard (57% of a technical corpus at K=32768, vs Qwen's 92.6%
where HauhauCS's prefix hybrid comes from). Prefix trims are a *measurement
instrument* (draft cost vs K, byte-identity, memory), not the deployable
mechanism; that is the corpus-selected id list + `d2t` map (phase 2,
engine-side).

## vocab-stats.py — token-stream statistics

Streams are text files containing token ids in order — `llama-tokenize --ids`
output, or anything else with standalone integers.

```
vocab-stats.py count    ids.txt...  -o counts.bin      # per-id u32 counts
vocab-stats.py coverage counts.bin [--k 4096,8192,..]  # prefix-K vs top-K table
vocab-stats.py missrate (--prefix K | --idlist f) ids.txt...   # acceptance ceiling
vocab-stats.py select   counts.bin -k K -o idlist.bin [--pin ids] [--base f]
vocab-stats.py heldout  ids.txt... [--points ..] [--k ..]      # drift table
```

`select` emits the phase-2 artifact: a sorted u32-LE id list (top-K by count,
with pinned specials and an optional generic base set unioned in). Gemma 4
specials worth pinning: `0-5` (pad/eos/bos/unk/mask/[multimodal]),
`100,101,105,106` (channel + turn markers; 106 = `<turn|>`, the eot),
`258880-258884` (media span sentinels).

## bench

`little-gemma/bench/mtp-vocab-ab.sh` runs the phase-1 A/B on the Orin:
full head vs prefix-trimmed at several K, measuring reply byte-identity (H1),
draft ms/round + decode tok/s (H2), acceptance vs the coverage prediction
(H3), and MemAvailable watermark/steady-state (H4).

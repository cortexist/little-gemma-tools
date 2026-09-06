#!/usr/bin/env python3
# vocab-stats.py — token-id-stream statistics for the MTP vocab trim
# (little-gemma/docs/mtp-vocab-trim.md). Answers, on real streams: how many
# distinct ids, what a prefix-K or top-K head would cover, what a given id
# list would miss (the draft's acceptance ceiling), and how coverage holds up
# on held-out data.
#
# Streams are text files containing token ids — `llama-tokenize --ids` output
# or anything else; every standalone integer in [0, n_vocab) counts, in order.
#
#   count    streams... -o counts.bin        accumulate per-id counts (u32 LE)
#   coverage counts.bin [--k a,b,..]         prefix-K / top-K coverage table
#   missrate (--prefix K | --idlist f) streams...   miss % per stream + total
#   select   counts.bin -k K -o idlist.bin [--pin ids] [--base f]   top-K set
#   heldout  streams... [--points a,b,..]    observed-vs-held-out drift table
#
# stdlib only.

import argparse
import array
import random
import re
import struct
import sys

N_VOCAB = 262144
_INT = re.compile(rb"(?<![\w.-])(\d+)(?![\w.])")


def read_ids(path, n_vocab):
    with open(path, "rb") as f:
        data = f.read()
    return [v for m in _INT.finditer(data) if (v := int(m.group(1))) < n_vocab]


def read_counts(path):
    with open(path, "rb") as f:
        c = array.array("I")
        c.frombytes(f.read())
    if sys.byteorder == "big":
        c.byteswap()
    return c


def write_u32(path, values):
    a = array.array("I", values)
    if sys.byteorder == "big":
        a.byteswap()
    with open(path, "wb") as f:
        f.write(a.tobytes())


def read_idlist(path):
    with open(path, "rb") as f:
        raw = f.read()
    return set(struct.unpack(f"<{len(raw) // 4}I", raw))


def cmd_count(args):
    counts = array.array("I", bytes(4 * args.n_vocab))
    total = 0
    for path in args.streams:
        ids = read_ids(path, args.n_vocab)
        for i in ids:
            if counts[i] < 0xFFFFFFFF:
                counts[i] += 1
        total += len(ids)
        print(f"{path}: {len(ids)} tokens", file=sys.stderr)
    write_u32(args.output, counts)
    distinct = sum(1 for c in counts if c)
    print(f"{args.output}: {total} tokens, {distinct} distinct ids of {args.n_vocab}")


def cmd_coverage(args):
    counts = read_counts(args.counts)
    total = sum(counts)
    if not total:
        sys.exit("empty counts")
    distinct = sum(1 for c in counts if c)
    ks = [int(k) for k in args.k.split(",")]
    ranked = sorted(counts, reverse=True)

    print(f"{args.counts}: {total} tokens, {distinct} distinct of {len(counts)}")
    print(f"{'K':>8}  {'prefix-K':>9}  {'top-K':>9}")
    prefix_cum = 0
    cum = list(ranked)
    for i in range(1, len(cum)):
        cum[i] += cum[i - 1]
    pfx = list(counts)
    for i in range(1, len(pfx)):
        pfx[i] += pfx[i - 1]
    for k in ks:
        print(f"{k:>8}  {100 * pfx[k - 1] / total:>8.2f}%  {100 * cum[k - 1] / total:>8.2f}%")

    for tgt in (0.99, 0.999, 1.0):
        need = next(i + 1 for i in range(len(cum)) if cum[i] >= total * tgt)
        print(f"top-K for {100 * tgt:g}%: K={need}")


def cmd_missrate(args):
    if args.idlist:
        kept = read_idlist(args.idlist)
        label = args.idlist
        member = kept.__contains__
    else:
        k = args.prefix
        label = f"prefix-{k}"
        member = lambda i: i < k
    tot_n = tot_miss = 0
    missed = {}
    for path in args.streams:
        ids = read_ids(path, args.n_vocab)
        miss = 0
        for i in ids:
            if not member(i):
                miss += 1
                missed[i] = missed.get(i, 0) + 1
        tot_n += len(ids)
        tot_miss += miss
        print(f"{path}: {miss}/{len(ids)} missed ({100 * miss / max(len(ids), 1):.2f}%)")
    print(f"total vs {label}: {tot_miss}/{tot_n} missed ({100 * tot_miss / max(tot_n, 1):.2f}%), "
          f"coverage {100 * (1 - tot_miss / max(tot_n, 1)):.2f}%, {len(missed)} distinct ids outside")
    if args.top and missed:
        worst = sorted(missed.items(), key=lambda kv: -kv[1])[:args.top]
        print("most-missed ids:", ", ".join(f"{i}×{n}" for i, n in worst))


def cmd_select(args):
    counts = read_counts(args.counts)
    pinned = set()
    if args.pin:
        pinned.update(int(x) for x in args.pin.split(","))
    if args.base:
        pinned.update(read_idlist(args.base))
    if len(pinned) > args.k:
        sys.exit(f"pins+base = {len(pinned)} ids exceed budget K={args.k}")
    ranked = sorted(range(len(counts)), key=lambda i: -counts[i])
    chosen = set(pinned)
    for i in ranked:
        if len(chosen) >= args.k:
            break
        if counts[i]:
            chosen.add(i)
    # pad with low ids so the file is exactly K rows (harmless, cheap rows)
    i = 0
    while len(chosen) < args.k:
        chosen.add(i)
        i += 1
    idlist = sorted(chosen)
    write_u32(args.output, idlist)
    total = sum(counts)
    cov = sum(counts[i] for i in idlist)
    print(f"{args.output}: K={len(idlist)} ({len(pinned)} pinned/base), "
          f"corpus coverage {100 * cov / max(total, 1):.2f}%")


def cmd_heldout(args):
    ids = []
    for path in args.streams:
        ids.extend(read_ids(path, args.n_vocab))
    points = [int(p) for p in args.points.split(",")]
    ks = [int(k) for k in args.k.split(",")]
    rng = random.Random(42)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    print(f"{len(ids)} tokens; coverage of the held-out remainder by top-K of the observed part")
    hdr = "  ".join(f"K={k}" for k in ks)
    print(f"{'observed':>9}  {'split':>10}  {hdr}")
    for n in points:
        if n >= len(ids):
            break
        for name, stream in (("shuffled", shuffled), ("sequential", ids)):
            seen = {}
            for i in stream[:n]:
                seen[i] = seen.get(i, 0) + 1
            ranked = sorted(seen, key=lambda i: -seen[i])
            rest = stream[n:]
            cells = []
            for k in ks:
                keep = set(ranked[:k])
                miss = sum(1 for i in rest if i not in keep)
                cells.append(f"{100 * (1 - miss / len(rest)):5.1f}%")
            print(f"{n:>9}  {name:>10}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-vocab", type=int, default=N_VOCAB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("count")
    p.add_argument("streams", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_count)

    p = sub.add_parser("coverage")
    p.add_argument("counts")
    p.add_argument("--k", default="4096,8192,16384,32768,65536")
    p.set_defaults(fn=cmd_coverage)

    p = sub.add_parser("missrate")
    p.add_argument("streams", nargs="+")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", type=int)
    g.add_argument("--idlist")
    p.add_argument("--top", type=int, default=0, help="show N most-missed ids")
    p.set_defaults(fn=cmd_missrate)

    p = sub.add_parser("select")
    p.add_argument("counts")
    p.add_argument("-k", type=int, required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--pin", help="comma-separated ids always kept (specials)")
    p.add_argument("--base", help="idlist.bin to union in (generic base set)")
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser("heldout")
    p.add_argument("streams", nargs="+")
    p.add_argument("--points", default="10000,25000,50000,100000,200000")
    p.add_argument("--k", default="8192,16384,32768")
    p.set_defaults(fn=cmd_heldout)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

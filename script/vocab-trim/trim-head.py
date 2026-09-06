#!/usr/bin/env python3
# trim-head.py — prefix-trim the MTP draft head's tied LM head in a GGUF, by
# byte surgery. Keeps rows 0..K-1 of token_embd.weight (draft row index ==
# target token id, the identity d2t), so little-gemma runs the result with ZERO
# engine changes: mtp_open takes n_vocab from the tensor dims, and the draft
# GGUF's token_embd is only ever read as the LM head (input embeddings come
# from the target model). A draft the trimmed head can no longer express is
# simply rejected by the target's full-vocab greedy verify — output stays
# byte-identical, only acceptance (and draft cost) moves.
#
# The slice is exact: rows of a linear head are independent, and quantized
# rows here are whole bytes (ne[0] divisible by the quant block), so this is a
# row-aligned byte copy — no dequantize, no requantize, no numerical change.
#
# Everything else in the file is copied byte-verbatim (the KV section is not
# even re-serialized), and the output is re-parsed and byte-compared against
# the source before the tool reports success.
#
# Usage: trim-head.py SRC.gguf OUT.gguf K [--tensor token_embd.weight]
#
# stdlib only, runs the same on x86 and the Orin.

import argparse
import hashlib
import struct
import sys

MAGIC = b"GGUF"

# ggml value types in GGUF KV encoding
_KV_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
KV_STRING, KV_ARRAY = 8, 9

# ggml tensor types: type id -> (bytes per block, elements per block).
# Only what these model families actually use; unknown types are a hard error
# (better than guessing sizes), cross-checked against the file's own offsets.
GGML_TYPE = {
    0:  ("F32", 4, 1),
    1:  ("F16", 2, 1),
    2:  ("Q4_0", 18, 32),
    3:  ("Q4_1", 20, 32),
    6:  ("Q5_0", 22, 32),
    7:  ("Q5_1", 24, 32),
    8:  ("Q8_0", 34, 32),
    9:  ("Q8_1", 36, 32),
    10: ("Q2_K", 84, 256),
    11: ("Q3_K", 110, 256),
    12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256),
    14: ("Q6_K", 210, 256),
    24: ("I8", 1, 1),
    25: ("I16", 2, 1),
    26: ("I32", 4, 1),
    27: ("I64", 8, 1),
    28: ("F64", 8, 1),
    30: ("BF16", 2, 1),
}


class Reader:
    def __init__(self, buf):
        self.buf = buf
        self.p = 0

    def take(self, n):
        b = self.buf[self.p:self.p + n]
        if len(b) != n:
            raise ValueError("truncated file")
        self.p += n
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def string(self): return self.take(self.u64())


def skip_kv_value(r, vtype):
    if vtype in _KV_SCALAR_SIZE:
        r.take(_KV_SCALAR_SIZE[vtype])
    elif vtype == KV_STRING:
        r.string()
    elif vtype == KV_ARRAY:
        etype, count = r.u32(), r.u64()
        if etype in _KV_SCALAR_SIZE:
            r.take(_KV_SCALAR_SIZE[etype] * count)
        elif etype == KV_STRING:
            for _ in range(count):
                r.string()
        else:
            raise ValueError(f"nested/unknown array element type {etype}")
    else:
        raise ValueError(f"unknown KV value type {vtype}")


class Gguf:
    """Parsed skeleton: header fields, verbatim KV bytes, tensor infos, data."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.raw = f.read()
        r = Reader(self.raw)
        if r.take(4) != MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        self.version = r.u32()
        if self.version != 3:
            raise ValueError(f"{path}: GGUF v{self.version}, only v3 supported")
        self.n_tensors = r.u64()
        self.n_kv = r.u64()

        kv_start = r.p
        self.alignment = 32
        for _ in range(self.n_kv):
            key = r.string()
            vtype = r.u32()
            if key == b"general.alignment" and vtype == 4:
                self.alignment = struct.unpack("<I", self.raw[r.p:r.p + 4])[0]
            skip_kv_value(r, vtype)
        self.kv_bytes = self.raw[kv_start:r.p]

        # tensor infos: (name, [ne...], type id, offset into data section)
        self.infos = []
        for _ in range(self.n_tensors):
            name = r.string().decode()
            n_dims = r.u32()
            ne = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            self.infos.append([name, ne, ttype, offset])

        a = self.alignment
        self.data_start = (r.p + a - 1) // a * a

    def row_bytes(self, info):
        name, ne, ttype, _ = info
        if ttype not in GGML_TYPE:
            raise ValueError(f"{name}: unknown ggml type {ttype}")
        tname, tsize, blck = GGML_TYPE[ttype]
        if ne[0] % blck:
            raise ValueError(f"{name}: ne[0]={ne[0]} not divisible by {tname} block {blck}")
        return ne[0] // blck * tsize

    def nbytes(self, info):
        n = self.row_bytes(info)
        for d in info[1][1:]:
            n *= d
        return n

    def tensor_data(self, info):
        off = self.data_start + info[3]
        return self.raw[off:off + self.nbytes(info)]

    def check_offsets(self):
        """The type-size table must agree with the file's own layout."""
        order = sorted(self.infos, key=lambda i: i[3])
        a = self.alignment
        pos = 0
        for info in order:
            if info[3] != pos:
                raise ValueError(f"{info[0]}: offset {info[3]}, expected {pos} "
                                 f"(type-size table disagrees with file layout)")
            pos = (info[3] + self.nbytes(info) + a - 1) // a * a
        end = self.data_start + order[-1][3] + self.nbytes(order[-1])
        if not end <= len(self.raw) <= (end + a - 1) // a * a:
            raise ValueError(f"data section ends at {end}, file is {len(self.raw)} bytes")


def write_gguf(path, src, infos, data_of):
    """infos in output layout order; data_of(info) -> bytes for that tensor."""
    a = src.alignment
    out = bytearray()
    out += MAGIC
    out += struct.pack("<IQQ", src.version, len(infos), src.n_kv)
    out += src.kv_bytes

    blobs, pos = [], 0
    info_bytes = bytearray()
    for info in infos:
        data = data_of(info)
        name, ne, ttype, _ = info
        nb = name.encode()
        info_bytes += struct.pack("<Q", len(nb)) + nb
        info_bytes += struct.pack("<I", len(ne))
        for d in ne:
            info_bytes += struct.pack("<Q", d)
        info_bytes += struct.pack("<IQ", ttype, pos)
        blobs.append((pos, data))
        pos = (pos + len(data) + a - 1) // a * a
    out += info_bytes

    pad = (-len(out)) % a
    out += b"\x00" * pad
    data_start = len(out)
    for off, data in blobs:
        out += b"\x00" * (data_start + off - len(out))
        out += data

    with open(path, "wb") as f:
        f.write(out)


def validate(src, out_path, tensor_name, k):
    out = Gguf(out_path)
    out.check_offsets()
    assert out.kv_bytes == src.kv_bytes, "KV section changed"
    assert out.n_tensors == src.n_tensors
    src_by_name = {i[0]: i for i in src.infos}
    for info in out.infos:
        s = src_by_name[info[0]]
        assert info[2] == s[2], f"{info[0]}: type changed"
        if info[0] == tensor_name:
            assert info[1] == [s[1][0], k], f"{tensor_name}: dims {info[1]}"
            want = src.tensor_data(s)[:src.row_bytes(s) * k]
        else:
            assert info[1] == s[1], f"{info[0]}: dims changed"
            want = src.tensor_data(s)
        got = out.tensor_data(info)
        assert hashlib.sha256(got).digest() == hashlib.sha256(want).digest(), \
            f"{info[0]}: data bytes differ"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("k", type=int, help="rows to keep (ids 0..K-1)")
    ap.add_argument("--tensor", default="token_embd.weight")
    args = ap.parse_args()

    src = Gguf(args.src)
    src.check_offsets()
    by_name = {i[0]: i for i in src.infos}
    if args.tensor not in by_name:
        sys.exit(f"{args.src}: no tensor {args.tensor!r}")
    head = by_name[args.tensor]
    if len(head[1]) != 2:
        sys.exit(f"{args.tensor}: expected 2 dims, got {head[1]}")
    n_vocab = head[1][1]
    if not 0 < args.k <= n_vocab:
        sys.exit(f"K={args.k} out of range (1..{n_vocab})")
    rb = src.row_bytes(head)
    tname = GGML_TYPE[head[2]][0]

    def data_of(info):
        blob = src.tensor_data(info)
        return blob[:rb * args.k] if info[0] == args.tensor else blob

    trimmed = [i.copy() for i in src.infos]
    for i in trimmed:
        if i[0] == args.tensor:
            i[1] = [i[1][0], args.k]
    write_gguf(args.out, src, trimmed, data_of)
    validate(src, args.out, args.tensor, args.k)

    import os
    print(f"{args.tensor}: {tname}, {rb} B/row, {n_vocab} -> {args.k} rows "
          f"({rb * n_vocab / 2**20:.1f} -> {rb * args.k / 2**20:.1f} MiB in file, "
          f"{head[1][0] * n_vocab * 2 / 2**20:.0f} -> {head[1][0] * args.k * 2 / 2**20:.0f} MiB as device f16)")
    print(f"{args.out}: {os.path.getsize(args.out) / 2**20:.1f} MiB "
          f"(from {os.path.getsize(args.src) / 2**20:.1f}), validated byte-exact")


if __name__ == "__main__":
    main()

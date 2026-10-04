"""Static SASS view of the kernel: what one K/V tile costs, instruction by instruction.

Builds _attn_fwd for sm_90 ahead of time (no GPU needed: Triton ships ptxas and nvdisasm), or takes the
kernel a real launch compiled (--from-cache, on a GPU). For each loop it prints the instructions by class
and the schedule skeleton: where the tensor-core matmuls, the waits, the exponentials, the loads and the
barriers sit. The kernel's registers, spills and shared memory come from cuobjdump.

Usage:
  TRITON_INTERPRET=0 python bench/sass_stats.py                  # D=128 default config, fp16, causal
  python bench/sass_stats.py --pack 4 --dump sass.txt            # GQA-packed variant, keep the SASS
  python bench/sass_stats.py --variant before_fixes              # a variant from bench/ablation.py
  python bench/sass_stats.py --from-cache                        # on a GPU: the kernel a launch compiled
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# Instruction classes, matched on the opcode (predicate stripped). First match wins.
CLASSES = (
    ("tensor", re.compile(r"^(HGMMA|HMMA|IMMA)")),
    ("mufu", re.compile(r"^MUFU")),
    ("wait/sync", re.compile(r"^(WARPGROUP|BAR|DEPBAR|LDGDEPBAR|MEMBAR|WARPSYNC|SYNCS|ARRIVES)")),
    ("load/store", re.compile(r"^(LDGSTS|LDG|STG|LDS|STS|LDSM|STSM|UTMALDG|UTMASTG|UBLKCP|LD|ST|ATOM|RED)\b")),
    ("shuffle", re.compile(r"^SHFL")),
    ("fp32", re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FSEL|FSETP|FCHK)")),
    ("convert", re.compile(r"^(F2FP|F2F|I2F|F2I|I2FP)")),
    ("uniform", re.compile(r"^U")),
    ("control", re.compile(r"^(BRA|EXIT|RET|BSSY|BSYNC|NOP|CALL|YIELD)")),
)
# Instructions that mark a step of the schedule; the runs of other instructions between them are summarised.
MARKERS = ("tensor", "wait/sync")
LOADS = re.compile(r"^(LDGSTS|LDG|UTMALDG|UBLKCP)")
# H100 per-SM, per-clock throughputs used for the per-tile cycle budget: dense FP16/BF16 tensor-core FLOPs
# (989.4 TFLOP/s = 132 SMs x 1.83 GHz x 4096), MUFU ops (16), FP32 FADD/FMUL/FFMA lanes (128), and one warp
# instruction issued per clock on each of the 4 SM sub-partitions.
SM_TENSOR_FLOP, SM_MUFU, SM_FP32, SM_ISSUE = 4096, 16, 128, 4
LABEL = re.compile(r"^(\w+):$")
BRANCH = re.compile(r"\bBRA\s+(\w+)")


def opcode(text):
    """'@!P2 HGMMA.64x64x16.F32 R88, ...' -> 'HGMMA.64x64x16.F32'."""
    words = text.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return words[0].rstrip(";") if words else ""


def klass(op):
    return next((name for name, rx in CLASSES if rx.match(op)), "int/other")


def parse(sass):
    """SASS text from triton.tools.disasm.get_sass -> (instructions, {label: index of its first instruction})."""
    insts, labels = [], {}
    for line in sass.splitlines():
        line = line.strip()
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(insts)
        elif "\t" in line:
            insts.append(line.split("\t", 1)[1].strip().rstrip(";"))
    return insts, labels


def loops(insts, labels):
    """(start, end) instruction ranges of every loop: a branch back to an earlier label, end inclusive."""
    out = []
    for i, text in enumerate(insts):
        m = BRANCH.search(text)
        if m and m.group(1) in labels and labels[m.group(1)] <= i:
            out.append((labels[m.group(1)], i))
    return out


def histogram(insts):
    by_class, by_op = collections.Counter(), collections.Counter()
    for text in insts:
        op = opcode(text)
        by_class[klass(op)] += 1
        by_op[op] += 1
    return by_class, by_op


def _is_marker(op):
    return klass(op) in MARKERS or LOADS.match(op) is not None


def skeleton(insts, glue=12):
    """The schedule as steps: tensor-core, wait, barrier and load instructions, with the other instructions
    between them summarised by count and most common opcodes. A run of the same marker opcode is one step
    even when up to `glue` address or uniform instructions sit between its members."""
    steps = []                               # [op, count, interleaved] for markers, [None, run] otherwise
    for text in insts:
        op = opcode(text)
        if _is_marker(op):
            if (len(steps) >= 2 and steps[-2][0] == op and steps[-1][0] is None and len(steps[-1][1]) <= glue
                    and all(klass(opcode(t)) in ("uniform", "int/other") for t in steps[-1][1])):
                glued = steps.pop()[1]
                steps[-1][1] += 1
                steps[-1][2] += len(glued)
            elif steps and steps[-1][0] == op:
                steps[-1][1] += 1
            else:
                steps.append([op, 1, 0])
        elif steps and steps[-1][0] is None:
            steps[-1][1].append(text)
        else:
            steps.append([None, [text]])
    lines = []
    for step in steps:
        if step[0] is None:
            ops = collections.Counter(opcode(t) if opcode(t).startswith("MUFU") else opcode(t).split(".")[0]
                                      for t in step[1])
            lines.append(f"    {len(step[1]):4d} other ({', '.join(f'{n} {op}' for op, n in ops.most_common(5))})")
        else:
            op, n, glued = step
            lines.append(f"  {op}" + (f" x{n}" if n > 1 else "") + (f" (+{glued} address/uniform)" if glued else ""))
    return lines


def budget(body, tile):
    """Clocks per loop iteration per SM if each resource ran alone at its peak: tensor cores, instruction
    issue, MUFU, FP32 pipe. tile = (BLOCK_M, BLOCK_N, HEAD_DIM, num_warps), one CTA per SM."""
    bm, bn, d, warps = tile
    _, by_op = histogram(body)
    threads = warps * 32
    n_mufu = sum(n for op, n in by_op.items() if op.startswith("MUFU"))
    n_fp32 = sum(n for op, n in by_op.items() if op.split(".")[0] in ("FADD", "FMUL", "FFMA"))
    return {"tensor": 2 * 2 * bm * bn * d / SM_TENSOR_FLOP, "issue": len(body) * warps / SM_ISSUE,
            "mufu": n_mufu * threads / SM_MUFU, "fp32": n_fp32 * threads / SM_FP32}


def report(sass, resources=None, tile=None):
    insts, labels = parse(sass)
    lines = [f"{len(insts)} instructions"]
    if resources:
        lines.append(resources)
    for n, (lo, hi) in enumerate([(lo, hi) for lo, hi in loops(insts, labels) if hi > lo], start=1):
        body = insts[lo:hi + 1]
        by_class, by_op = histogram(body)
        lines += ["", f"loop {n}: {len(body)} instructions per iteration",
                  "  by class: " + ", ".join(f"{c} {k}" for c, k in by_class.most_common()),
                  "  top opcodes: " + ", ".join(f"{op} {k}" for op, k in by_op.most_common(12))]
        if tile:
            b = budget(body, tile)
            lines.append("  clocks per iteration per SM, each resource alone at peak: "
                         + ", ".join(f"{k} {v:.0f}" for k, v in b.items()))
        lines.append("  schedule:")
        lines += skeleton(body)
    return "\n".join(lines)


# ---------------------------------------------------------------------------- compile

def _signature(fn, constexprs):
    import inspect

    sig, consts, attrs = {}, {}, {}
    for i, name in enumerate(inspect.signature(fn.fn).parameters):
        if name in ("Q", "K", "V", "Out"):
            sig[name] = "*" + constexprs["dtype"]
            attrs[(i,)] = [["tt.divisibility", 16]]
        elif name.startswith("stride_") and name.endswith("d"):     # unit stride: Triton specializes 1
            sig[name] = "constexpr"
            consts[(i,)] = 1
        elif name.startswith("stride_") or name in ("H_Q", "N_Q", "N_KV"):
            sig[name] = "i32"
            attrs[(i,)] = [["tt.divisibility", 16]]             # true for the benchmark shapes
        elif name == "qk_scale":
            sig[name] = "fp32"
        else:
            sig[name] = "constexpr"
            consts[(i,)] = constexprs[name]
    return sig, consts, attrs


def compile_aot(args):
    """CompiledKernel for sm_90 without a GPU, with the launch's config and specialization."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from fa2.kernel import _attn_fwd, _default_config

    bm, bn, nw, ns = _default_config(args.head_dim, on_cuda=True)
    constexprs = dict(dtype=args.dtype, GQA_GROUP=args.group, PACK=args.pack, HEAD_DIM=args.head_dim,
                      BLOCK_M=bm, BLOCK_N=bn, CAUSAL=not args.no_causal, EVEN_N=not args.odd_n)
    fn = _attn_fwd
    if args.variant:
        from ablation import KNOBS, VARIANTS, _attn_fwd_ablate

        fn = _attn_fwd_ablate
        on = VARIANTS[args.variant]
        constexprs.update({k.upper(): k in on for k in KNOBS if k != "mask_every_tile"})
        constexprs["EVEN_N"] = constexprs["EVEN_N"] and "mask_every_tile" not in on
    sig, consts, attrs = _signature(fn, constexprs)
    src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
    return triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": nw, "num_stages": ns})


def compile_from_cache(args):
    """Launch once on the GPU, then take the kernel Triton compiled for that launch."""
    import torch

    from fa2.kernel import _attn_fwd, attention

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    q = torch.randn(1, args.group * 8, 4096, args.head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 8, 4096, args.head_dim, device="cuda", dtype=dtype)
    attention(q, k, torch.randn_like(k), causal=not args.no_causal, pack_gqa=args.pack > 1)
    torch.cuda.synchronize()
    kernels = [ck for caches in _attn_fwd.device_caches.values() for ck in caches[0].values()]
    assert kernels, "no compiled kernel in the cache"
    return kernels[-1]


def resources(cubin):
    import triton

    tool = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        out = subprocess.run([tool, "--dump-resource-usage", f.name], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", out)
    return f"registers {m.group(1)}/thread, stack {m.group(2)} B, local (spills) {m.group(4)} B" if m else out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--group", type=int, default=4, help="GQA group (H_q / H_kv)")
    ap.add_argument("--pack", type=int, default=1, help="heads per program: 1, or the group to pack")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--odd-n", action="store_true", help="N_kv not a multiple of BLOCK_N (keeps the end mask)")
    ap.add_argument("--from-cache", action="store_true", help="compile with a real launch (needs a GPU)")
    ap.add_argument("--variant", help="compile this variant of bench/ablation.py instead (ahead of time only)")
    ap.add_argument("--dump", help="also write the full SASS here")
    args = ap.parse_args()
    if os.environ.get("TRITON_INTERPRET") == "1":
        sys.exit("unset TRITON_INTERPRET: the interpreter has no compiled kernel")

    from triton.tools.disasm import get_sass

    from fa2.kernel import _default_config

    ck = compile_from_cache(args) if args.from_cache else compile_aot(args)
    bm, bn, nw, _ = _default_config(args.head_dim, on_cuda=True)
    sass = get_sass(ck.asm["cubin"])
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(sass)
    shared = getattr(ck.metadata, "shared", None)
    res = resources(ck.asm["cubin"]) + (f", shared {shared / 1024:.0f} KiB" if shared else "")
    print(report(sass, res, tile=(bm, bn, args.head_dim, nw)))


if __name__ == "__main__":
    main()

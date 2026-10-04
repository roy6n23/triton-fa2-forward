"""Side-by-side table of the Nsight Compute reports that bench/profile_ncu.sh exports, plus where the kernel's
warps stall, instruction by instruction. Runs anywhere (reads the CSV exports, no GPU or ncu needed).

  python bench/ncu_summary.py bench/results/2026-10-04/ncu
  python bench/ncu_summary.py bench/results/2026-10-04/ncu --top 25        # more instructions in the stall list

Note: ncu locks the SM clock (to about 1.44 GHz on H100 SXM), so durations are longer than in bench.py; compare
the percentages.
"""

import argparse
import collections
import csv
import os
import re

TARGETS = ("triton-fa2", "before_fixes", "no_exp", "matmul_only", "fa3-vllm")
DETAILS = (   # (label, details-page metric name)
    ("duration (ms, at ncu's locked clock)", "Duration"),
    ("SM clock (GHz)", "SM Frequency"),
    ("executed instructions (M)", "Executed Instructions"),
    ("issue slots busy (%)", "Issue Slots Busy"),
    ("no eligible warp (% of cycles)", "No Eligible"),
    ("warps per scheduler", "Active Warps Per Scheduler"),
    ("warp cycles per issued instruction", "Warp Cycles Per Issued Instruction"),
    ("registers per thread", "Registers Per Thread"),
    ("L2 hit rate (%)", "L2 Hit Rate"),
    ("DRAM throughput (% of peak)", "DRAM Throughput"),
)
RAW = (       # (label, raw-page metric name)
    ("tensor pipe active (%)", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("MUFU/XU pipe (%)", "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed"),
    ("FMA pipe (%)", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("ALU pipe (%)", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"),
)
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
STALLS = ("selected", "barrier", "wait", "long_scoreboard", "mio_throttle", "not_selected", "short_scoreboard",
          "dispatch_stall", "math_pipe_throttle", "warpgroup_arrive", "gmma", "sleeping")


def details(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["Metric Name"], r["Metric Value"])
    return out


def raw(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return dict(zip(rows[0], rows[2]))


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def stall_shares(r):
    counts = {k[len(STALL_PREFIX):]: num(v) or 0.0 for k, v in r.items()
              if k.startswith(STALL_PREFIX) and not k.endswith("_not_issued")}
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def table(names, rows):
    out = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    out += ["| " + label + " | " + " | ".join(cells) + " |" for label, cells in rows]
    return "\n".join(out)


def fmt(x, digits=1):
    v = num(x)
    return "n/a" if v is None else f"{v:,.{digits}f}"


def summary(d):
    names = [n for n in TARGETS if os.path.exists(os.path.join(d, f"{n}.details.csv"))]
    det = {n: details(os.path.join(d, f"{n}.details.csv")) for n in names}
    rw = {n: raw(os.path.join(d, f"{n}.raw.csv")) for n in names}
    rows = []
    for label, key in DETAILS:
        cells = []
        for n in names:
            v = det[n].get(key)
            if key == "Executed Instructions" and num(v) is not None:
                v = num(v) / 1e6
            cells.append(fmt(v, 2 if key in ("Duration", "SM Frequency", "Warp Cycles Per Issued Instruction",
                                             "Active Warps Per Scheduler") else (0 if key == "Registers Per Thread" else 1)))
        rows.append((label, cells))
    rows += [(label, [fmt(rw[n].get(key)) for n in names]) for label, key in RAW]
    shares = {n: stall_shares(rw[n]) for n in names}
    rows += [(f"stall: {s} (% of samples)", [fmt(100 * shares[n].get(s, 0.0)) for n in names]) for s in STALLS]
    return table(names, rows)


def _step(op):
    """Which part of the K/V-tile schedule (README, "Where the time goes") an opcode belongs to."""
    if op.startswith("WARPGROUP.DEPBAR"):
        return "wait for the Q K^T matmul (WARPGROUP.DEPBAR)"
    if op.startswith(("HGMMA", "WARPGROUP.ARRIVE", "FENCE")):
        return "issue the matmuls (HGMMA, WARPGROUP.ARRIVE, fences)"
    if op.startswith(("BAR", "LDGSTS", "LDGDEPBAR", "DEPBAR")) or op == "LDS":
        return "barriers and K/V copies (BAR, LDGSTS, DEPBAR, LDS)"
    if op.startswith(("MUFU", "FADD", "FMUL", "FFMA", "FMNMX", "FSEL", "FSETP", "F2FP", "SHFL")):
        return "softmax (MUFU, FP32, shuffles, conversions)"
    return "address and loop bookkeeping (IMAD, IADD, LEA, uniform, branch)"


def per_instruction(path, top):
    """Stall samples per SASS instruction of the kernel, and summed by opcode; the hottest instructions first."""
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"Address"'))
    reader = csv.DictReader(lines[start:])
    rows = [r for r in reader if r.get("Source")]
    # "stall_<reason>" counts every sample; "stall_<reason> (Not Issued)" is a subset of it, so it is left out.
    stall_cols = [c for c in reader.fieldnames if c.startswith("stall_") and "Not Issued" not in c]
    for r in rows:
        r["_samples"] = sum(num(r[c]) or 0 for c in stall_cols)
    total = sum(r["_samples"] for r in rows) or 1
    by_op = collections.Counter()
    for r in rows:
        op = re.sub(r"^@!?U?P\w+\s+", "", r["Source"].strip()).split(" ")[0].rstrip(";")
        by_op[op] += r["_samples"]
    groups = collections.Counter()
    for r in rows:
        op = re.sub(r"^@!?U?P\w+\s+", "", r["Source"].strip()).split(" ")[0].rstrip(";")
        groups[_step(op)] += r["_samples"]
    out = ["### Stall samples by step of the schedule", "", "| step | share of samples |", "|---|---|"]
    out += [f"| {g} | {n / total:.1%} |" for g, n in groups.most_common()]
    out += ["", "### Stall samples by opcode", "", "| opcode | share of samples |", "|---|---|"]
    out += [f"| {op} | {n / total:.1%} |" for op, n in by_op.most_common(12)]
    out += ["", f"### The {top} instructions with the most samples", "",
            "| address | instruction | share | main reasons |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["_samples"])[:top]:
        reasons = sorted(((c[6:], num(r[c]) or 0) for c in stall_cols), key=lambda kv: -kv[1])[:3]
        why = ", ".join(f"{k} {v / r['_samples']:.0%}" for k, v in reasons if v) if r["_samples"] else ""
        out.append(f"| {r['Address']} | `{r['Source'].strip()[:60]}` | {r['_samples'] / total:.1%} | {why} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory with <target>.details.csv, .raw.csv and .sass.csv")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--kernel", default="triton-fa2", help="target whose per-instruction stalls to list")
    args = ap.parse_args()
    print(summary(args.dir))
    sass = os.path.join(args.dir, f"{args.kernel}.sass.csv")
    if os.path.exists(sass):
        print()
        print(per_instruction(sass, args.top))


if __name__ == "__main__":
    main()

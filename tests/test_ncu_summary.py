"""bench/ncu_summary.py on the Nsight Compute exports committed under bench/results/2026-10-04/ncu (no GPU)."""

import os

from bench.ncu_summary import per_instruction, summary

NCU = os.path.join(os.path.dirname(__file__), "..", "bench", "results", "2026-10-04", "ncu")


def test_summary_table_reads_both_pages():
    text = summary(NCU)
    rows = {line.split(" | ")[0].lstrip("| "): line for line in text.splitlines()}
    assert text.splitlines()[0] == "| | triton-fa2 | before_fixes | no_exp | matmul_only | fa3-vllm |"
    assert "| 51.5 | 45.4 | 54.4 | 83.6 | 81.2 |" in rows["tensor pipe active (%)"]      # raw page
    assert "| 5.53 | 6.27 | 5.25 | 3.42 | 3.53 |" in rows["duration (ms, at ncu's locked clock)"]   # details page
    assert "| 186 |" in rows["registers per thread"]


def test_per_instruction_counts_each_sample_once():
    text = per_instruction(os.path.join(NCU, "triton-fa2.sass.csv"), top=3)
    lines = text.splitlines()
    steps = [l for l in lines[lines.index("### Stall samples by step of the schedule") + 4:] if l.startswith("| ")]
    shares = [float(l.rsplit("|", 2)[1].strip().rstrip("%")) for l in steps[:5]]
    assert abs(sum(shares) - 100) < 0.5                       # the five steps partition the samples
    hottest = next(l for l in lines if l.startswith("| 0x"))
    assert "WARPGROUP.DEPBAR.LE" in hottest and "(Not Issued)" not in hottest

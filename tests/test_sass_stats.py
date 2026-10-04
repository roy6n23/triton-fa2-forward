"""bench/sass_stats.py's SASS parsing, on a hand-written listing (no Triton, no GPU)."""

from bench.sass_stats import budget, histogram, klass, loops, opcode, parse, skeleton

SASS = """Function:_attn_fwd
--:-:1:-:8\tLDC R1, c[0x0][0x28];
LBB0:
--:-:-:-:1\tBAR.SYNC.DEFER_BLOCKING 0x0;
--:-:-:-:1\tWARPGROUP.ARRIVE;
--:-:-:-:1\tHGMMA.64x64x16.F32 R88, gdesc[UR12], RZ, !UPT;
--:-:-:-:1\tUIADD3 UR12, UR12, 0x2, URZ;
--:-:-:-:1\tHGMMA.64x64x16.F32 R88, gdesc[UR12], R88;
--:-:-:-:1\tWARPGROUP.DEPBAR.LE gsb0, 0x0;
--:-:-:-:1\tFMUL R2, R88, R3;
--:-:-:-:1\tMUFU.EX2 R4, R2;
--:-:-:-:1\tF2FP.F16.F32.PACK_AB R5, R4, R4;
--:-:-:-:1\tLDGSTS.E.BYPASS.128 [R179], desc[UR20][R8.64], P3;
--:-:-:-:6\t@!P2 BRA LBB0;
--:-:-:Y:0\tEXIT;
LBB1:
--:-:-:Y:0\tBRA LBB1;
"""


def test_opcode_and_class():
    assert opcode("@!P2 HGMMA.64x64x16.F32 R88, gdesc[UR12]") == "HGMMA.64x64x16.F32"
    assert [klass(op) for op in ("HGMMA.64x64x16.F32", "MUFU.EX2", "WARPGROUP.DEPBAR.LE", "LDGSTS.E.BYPASS.128",
                                 "FFMA", "F2FP.F16.F32.PACK_AB", "UIADD3", "BRA", "IMAD")] == \
        ["tensor", "mufu", "wait/sync", "load/store", "fp32", "convert", "uniform", "control", "int/other"]


def test_loops_are_backward_branches():
    insts, labels = parse(SASS)
    assert len(insts) == 14 and labels == {"LBB0": 1, "LBB1": 13}
    assert loops(insts, labels) == [(1, 11), (13, 13)]          # the second is the one-instruction trap


def test_histogram_skeleton_and_budget():
    insts, labels = parse(SASS)
    body = insts[1:12]
    by_class, by_op = histogram(body)
    assert by_class["tensor"] == 2 and by_class["mufu"] == 1 and by_op["HGMMA.64x64x16.F32"] == 2
    steps = skeleton(body)
    assert "  HGMMA.64x64x16.F32 x2 (+1 address/uniform)" in steps     # glued across the descriptor update
    assert steps.index("  WARPGROUP.DEPBAR.LE") < next(i for i, s in enumerate(steps) if "MUFU.EX2" in s)
    b = budget(body, (128, 64, 128, 8))
    assert b == {"tensor": 1024.0, "issue": 22.0, "mufu": 16.0, "fp32": 2.0}

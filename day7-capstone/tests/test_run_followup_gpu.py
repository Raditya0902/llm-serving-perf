"""Arm definitions of the follow-up runner (no GPU needed: only the generated commands are checked)."""
from __future__ import annotations

import run_followup_gpu as rf


def _runs(arms):
    return rf.build_runs("m", "llama3-8b", "a100-80gb", False, arms)


def test_single_gpu_arms_keep_concurrency_64_and_their_rep_counts():
    runs = _runs(["cap16", "pool092"])
    assert {r.case.concurrency for r in runs} == {64}
    assert [r.case.config.name for r in runs].count("cap16") == 1
    assert [r.case.config.name for r in runs].count("pool092") == 3


def test_tensor_parallel_arms_run_both_concurrencies_and_set_tp_only_for_tp2():
    runs = _runs(["tp1_c16", "tp1_c64", "tp2_c16", "tp2_c64", "tp2nc_c64"])
    ids = {r.case.case_id for r in runs}
    assert ids == {"B-c16-tp1", "B-c64-tp1", "B-c16-tp2", "B-c64-tp2", "B-c64-tp2nc"}
    cmds = {r.case.case_id: r.case.serve_command(8000) for r in runs}
    assert "--tensor-parallel-size" not in cmds["B-c64-tp1"]
    tp2 = cmds["B-c64-tp2"]
    assert tp2[tp2.index("--tensor-parallel-size") + 1] == "2"
    assert "--disable-custom-all-reduce" not in tp2
    nc = cmds["B-c64-tp2nc"]
    assert "--disable-custom-all-reduce" in nc and nc[nc.index("--tensor-parallel-size") + 1] == "2"


def test_tp_arms_differ_from_tp1_only_in_the_serve_flags():
    c = {n: [r for r in _runs([n])][0].case.config for n in ("tp1_c64", "tp2_c64")}
    assert (c["tp1_c64"].gpu_memory_utilization, c["tp1_c64"].max_num_batched_tokens) == \
           (c["tp2_c64"].gpu_memory_utilization, c["tp2_c64"].max_num_batched_tokens)

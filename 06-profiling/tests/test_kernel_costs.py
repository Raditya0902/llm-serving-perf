"""Per-kernel attribution of vLLM steps (synthetic Chrome-trace events, no GPU)."""
from __future__ import annotations

import pytest

from gpu import kernel_costs as kc
from gpu import step_costs as sc


def _step(prefill, decodes, ts, dur_us):
    return {"name": f"execute_context_{1 if prefill else 0}({prefill})_generation_{decodes}({decodes})",
            "ph": "X", "cat": "gpu_user_annotation", "ts": ts, "dur": dur_us}


def _kernel(name, ts, dur_us):
    return {"name": name, "ph": "X", "cat": "kernel", "ts": ts, "dur": dur_us}


@pytest.mark.parametrize("name,family", [
    ("void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 128>>", "attention"),
    ("kernel_unified_attention_2d", "attention"),
    ("void vllm::reshape_and_cache_flash_kernel<__half>", "kv_write"),
    ("sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128x32", "gemm"),
    ("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn", "gemm"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16>", "gemm"),
    ("void vllm::rms_norm_kernel<c10::Half>", "norm_act"),
    ("void vllm::rotary_embedding_kernel<c10::Half, true>", "norm_act"),
    ("triton_poi_fused_mul_silu_0", "norm_act"),
    ("void vllm::act_and_mul_kernel<c10::Half, &silu>", "norm_act"),
    ("void at::native::vectorized_elementwise_kernel<4, add>", "norm_act"),
    ("void at_cuda_detail::cub::DeviceRadixSortOnesweepKernel", "sampling"),
    ("void at::native::(anonymous namespace)::softmax_warp_forward", "sampling"),
    ("ncclDevKernel_AllReduce_Sum_f16_RING_LL", "comm"),
    ("void vllm::cross_device_reduce_1stage<half, 2>", "comm"),
    ("some_unknown_kernel", "other"),
])
def test_classify_kernel_families(name, family):
    assert kc.classify_kernel(name) == family


def test_parse_kernels_keeps_only_complete_kernel_events_in_time_order():
    events = [_kernel("b", 20, 5), _kernel("a", 10, 5), {"name": "memcpy", "ph": "X", "cat": "gpu_memcpy", "ts": 1, "dur": 1},
              {"name": "x", "ph": "B", "cat": "kernel", "ts": 3, "dur": 1}]
    assert [k.name for k in kc.parse_kernels(events)] == ["a", "b"]


def test_attribute_assigns_each_kernel_to_the_step_that_contains_its_start_and_reports_coverage():
    events = [_step(0, 8, ts=0, dur_us=1000), _step(2000, 8, ts=2000, dur_us=4000),
              _kernel("flash_fwd", 100, 300), _kernel("sm80_xmma_gemm", 500, 400),
              _kernel("flash_fwd", 2100, 2000), _kernel("sm80_xmma_gemm", 4200, 1500),
              _kernel("orphan_between_steps", 1500, 100)]
    steps = sc.parse_steps(events)
    attributed = kc.attribute(steps, kc.parse_kernels(events))
    assert [round(a.family_ms["attention"], 3) for a in attributed] == [0.3, 2.0]
    assert [round(a.family_ms["gemm"], 3) for a in attributed] == [0.4, 1.5]
    assert attributed[0].coverage == pytest.approx(0.7) and attributed[1].coverage == pytest.approx(0.875)


def test_summary_finds_the_family_that_carries_the_per_decode_cost_and_reports_low_coverage():
    events, ts = [], 0
    for i in range(14):  # attention = 0.5 ms per decode inside mixed steps, gemm = 3 ms per 1k prefill tokens, norm constant
        prefill, decodes = 1000 * (1 + i % 5), 40 * (1 + i % 4)
        attn, gemm, norm = 0.5 * decodes, 3.0 * prefill / 1000, 0.7
        total_us = (attn + gemm + norm) * 1000
        events += [_step(prefill, decodes, ts, int(total_us)), _kernel("flash_fwd_kernel", ts, int(attn * 1000)),
                   _kernel("sm80_xmma_gemm", ts + int(attn * 1000), int(gemm * 1000)),
                   _kernel("rms_norm_kernel", ts + int((attn + gemm) * 1000), int(norm * 1000))]
        ts += int(total_us) + 100
    events += [_step(0, 256, ts, 7000), _kernel("flash_fwd_splitkv_kernel", ts, 1000)]
    summary = kc.summarize(kc.attribute(sc.parse_steps(events), kc.parse_kernels(events)))

    fits = summary["mixed"]["fits"]
    assert fits["attention"]["ms_per_decode"] == pytest.approx(0.5, abs=1e-3)
    assert fits["gemm"]["ms_per_1000_prefill_tokens"] == pytest.approx(3.0, abs=1e-3)
    assert summary["mixed"]["decode_slope_share"]["attention"] == pytest.approx(1.0, abs=1e-2)
    assert kc.dominant_decode_family(summary) == "attention"
    assert summary["pure"]["mean_ms_by_family"]["attention"] == pytest.approx(1.0)
    assert any("coverage" in w for w in summary["warnings"])  # the 7 ms pure step has only 1 ms of kernels


def test_dominant_family_is_none_when_no_family_reaches_the_threshold():
    summary = {"mixed": {"decode_slope_share": {"attention": 0.5, "gemm": 0.5}}}
    assert kc.dominant_decode_family(summary, threshold=0.7) is None


def test_unclassified_top_lists_the_biggest_other_kernels_first():
    events = [_step(0, 8, 0, 10_000), _kernel("mystery_a", 0, 500), _kernel("mystery_b", 600, 900), _kernel("flash_fwd", 2000, 100)]
    top = kc.unclassified_top(kc.attribute(sc.parse_steps(events), kc.parse_kernels(events)), n=5)
    assert [name for name, _ms in top] == ["mystery_b", "mystery_a"]


def test_summary_of_no_steps_is_empty_not_an_error():
    summary = kc.summarize([])
    assert summary["pure"]["n"] == 0 and summary["mixed"]["n"] == 0 and summary["mixed"]["fits"] is None


def test_summary_warns_when_the_per_decode_slope_is_not_identified():
    events, ts = [], 0
    for i in range(10):  # decodes nearly constant (249-251) while prefill varies: the decode slope cannot be separated from the intercept
        prefill, decodes = 8000 - 700 * i, 250 + i % 3
        total_us = int((150 + 10 * prefill / 1000) * 1000)
        events += [_step(prefill, decodes, ts, total_us), _kernel("flash_fwd_kernel", ts, int(140_000)),
                   _kernel("sm80_xmma_gemm", ts + 140_000, total_us - 140_000)]
        ts += total_us + 100
    summary = kc.summarize(kc.attribute(sc.parse_steps(events), kc.parse_kernels(events)))
    assert summary["mixed"]["decode_identifiable"] is False
    assert any("not identified" in w for w in summary["warnings"])
    assert kc.dominant_decode_family(summary) is None  # a slope that is not identified must not be reported as a winner

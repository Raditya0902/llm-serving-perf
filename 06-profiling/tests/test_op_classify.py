import pytest

from op_classify import OpRow, category_totals, classify, rows_from_events, top_ops


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("aten::mm", "matmul"),
        ("aten::addmm", "matmul"),
        ("aten::bmm", "matmul"),
        ("aten::linear", "matmul"),
        ("aten::scaled_dot_product_attention", "attention"),
        ("aten::_scaled_dot_product_attention_math_for_mps", "attention"),
        ("aten::_scaled_dot_product_flash_attention_for_cpu", "attention"),
        ("aten::_softmax", "attention"),
        ("aten::_local_scalar_dense", "sync"),
        ("aten::item", "sync"),
        ("aten::cat", "memory"),
        ("aten::copy_", "memory"),
        ("aten::empty", "memory"),
        ("aten::mul", "elementwise"),
        ("aten::silu", "elementwise"),
        ("d6::layer0.attn", "annotation"),
        ("ProfilerStep#1", "annotation"),
        ("some_python_thing", "other"),
    ],
)
def test_classify_maps_operator_names_to_categories(name, category):
    assert classify(name) == category


def _rows() -> list[OpRow]:
    return [
        OpRow("aten::mm", count=10, self_cpu_us=400.0, self_device_us=0.0, total_cpu_us=450.0),
        OpRow("aten::_softmax", count=5, self_cpu_us=100.0, self_device_us=0.0, total_cpu_us=100.0),
        OpRow("aten::mul", count=20, self_cpu_us=50.0, self_device_us=0.0, total_cpu_us=50.0),
        OpRow("d6::layer0.attn", count=1, self_cpu_us=5.0, self_device_us=0.0, total_cpu_us=300.0),
    ]


def test_category_totals_exclude_annotations_and_report_shares():
    totals = category_totals(_rows())
    assert "annotation" not in totals
    assert totals["matmul"].self_cpu_us == 400.0
    assert totals["matmul"].share == pytest.approx(400 / 550)
    assert sum(t.share for t in totals.values()) == pytest.approx(1.0)


def test_category_totals_of_empty_input_is_empty():
    assert category_totals([]) == {}


def test_category_totals_uses_device_time_when_present():
    rows = [
        OpRow("aten::mm", 1, self_cpu_us=10.0, self_device_us=90.0, total_cpu_us=10.0),
        OpRow("aten::mul", 1, self_cpu_us=10.0, self_device_us=10.0, total_cpu_us=10.0),
    ]
    totals = category_totals(rows, use_device_time=True)
    assert totals["matmul"].share == pytest.approx(0.9)


def test_top_ops_sorts_by_self_time_and_skips_annotations():
    names = [r.name for r in top_ops(_rows(), 2)]
    assert names == ["aten::mm", "aten::_softmax"]


def test_rows_from_events_builds_rows_from_profiler_like_objects():
    class Event:
        key = "aten::mm"
        count = 3
        self_cpu_time_total = 12.0
        self_device_time_total = 0.0
        cpu_time_total = 15.0

    (row,) = rows_from_events([Event()])
    assert row == OpRow("aten::mm", 3, 12.0, 0.0, 15.0)

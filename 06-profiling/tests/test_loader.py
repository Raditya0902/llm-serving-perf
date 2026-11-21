import pytest

import _day1  # noqa: F401  (puts 01-inference-loop on sys.path)
from loader import ATTN_IMPLS, validate_attn_impl


def test_day1_modules_are_importable_through_the_shim():
    import device  # noqa: F401
    import generate

    assert generate.DEFAULT_MODEL_ID.startswith("Qwen/")


def test_day1_directory_is_searched_after_day6_so_day6_modules_win():
    import sys

    day6 = str(_day1.Path(__file__).resolve().parent.parent)
    assert sys.path.index(day6) < sys.path.index(str(_day1.DAY1_DIR))


@pytest.mark.parametrize("impl", ATTN_IMPLS)
def test_validate_attn_impl_accepts_supported_values(impl):
    assert validate_attn_impl(impl) == impl


def test_validate_attn_impl_rejects_unknown_value():
    with pytest.raises(ValueError, match="attn_impl"):
        validate_attn_impl("flash_attention_9")

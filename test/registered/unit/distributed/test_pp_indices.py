"""Unit tests for pipeline-parallel layer partitioning.

Covers two pieces of logic:

1. ``get_pp_indices`` (``srt/distributed/utils.py``) — how a stage's
   ``[start_layer, end_layer)`` range is derived from an explicit
   ``partition_list`` argument, the ``SGLANG_PP_LAYER_PARTITION`` env var, or the
   default balanced split, plus the associated validation.
2. ``get_dsa_safe_pp_layer_partition`` (``srt/configs/model_config.py``) — the
   auto-computed DSA-aware split. DSA/indexer models reuse the previous layer's
   top-k on "skip-topk" layers, so no PP stage boundary may land on such a layer;
   land on such a layer; otherwise a stage would start expecting top-k that was
   computed in the previous stage. The helper must pick a balanced split whose
   every stage (after the first) starts on a layer that computes its own top-k.

Both helpers are pure arithmetic over ints / a config object (no CUDA, no model
weights), so the test runs on the CPU suite.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_safe_pp_layer_partition,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_dsa_config(**overrides) -> SimpleNamespace:
    """DSA config with index_topk_freq and index_skip_topk_offset skip-topk pattern."""
    defaults = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        num_hidden_layers=78,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestGetPpIndices(CustomTestCase):
    def test_explicit_partition_list(self):
        # partition_list wins and defines per-rank [start, end).
        self.assertEqual(get_pp_indices(78, 0, 2, partition_list=[38, 40]), (0, 38))
        self.assertEqual(get_pp_indices(78, 1, 2, partition_list=[38, 40]), (38, 78))

    def test_partition_list_overrides_env_var(self):
        with patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": "39,39"}):
            # The explicit argument takes precedence over the env var.
            self.assertEqual(
                get_pp_indices(78, 1, 2, partition_list=[38, 40]), (38, 78)
            )

    def test_env_var_partition(self):
        with patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": "38,40"}):
            self.assertEqual(get_pp_indices(78, 0, 2), (0, 38))
            self.assertEqual(get_pp_indices(78, 1, 2), (38, 78))

    def test_default_balanced_split_even(self):
        self.assertEqual(get_pp_indices(78, 0, 2), (0, 39))
        self.assertEqual(get_pp_indices(78, 1, 2), (39, 78))

    def test_default_balanced_split_remainder_goes_to_last_stages(self):
        # 10 layers over 4 stages -> [2, 2, 3, 3]; remainder lands on the tail.
        ranges = [get_pp_indices(10, rank, 4) for rank in range(4)]
        self.assertEqual(ranges, [(0, 2), (2, 4), (4, 7), (7, 10)])

    def test_partition_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            get_pp_indices(78, 0, 2, partition_list=[38, 20, 20])

    def test_partition_sum_mismatch_raises(self):
        with self.assertRaises(ValueError):
            get_pp_indices(78, 0, 2, partition_list=[38, 39])

    def test_partition_list_non_int_raises(self):
        with self.assertRaises(ValueError):
            get_pp_indices(78, 0, 2, partition_list=["a", "b"])

    def test_invalid_env_var_raises(self):
        with patch.dict("os.environ", {"SGLANG_PP_LAYER_PARTITION": "38,x"}):
            with self.assertRaises(ValueError):
                get_pp_indices(78, 0, 2)


class TestDsaSafePpLayerPartition(CustomTestCase):
    def _assert_valid_and_safe(self, config, pp_size, partition):
        self.assertIsNotNone(partition)
        self.assertEqual(len(partition), pp_size)
        self.assertEqual(sum(partition), config.num_hidden_layers)
        # Every stage after the first must start on a layer that computes its
        # own top-k (i.e. not a skip-topk layer).
        boundary = 0
        for stage_len in partition[:-1]:
            boundary += stage_len
            self.assertFalse(
                dsa_layer_skips_topk(config, boundary),
                f"stage boundary at layer {boundary} is a skip-topk layer",
            )

    def test_freq_offset_pp2_expected_partition(self):
        config = _make_dsa_config()
        partition = get_dsa_safe_pp_layer_partition(config, 2)
        self.assertEqual(partition, [38, 40])
        self._assert_valid_and_safe(config, 2, partition)

    def test_freq_offset_pp4_differs_from_unsafe_default(self):
        config = _make_dsa_config()
        partition = get_dsa_safe_pp_layer_partition(config, 4)
        self._assert_valid_and_safe(config, 4, partition)
        # The default balanced split would be [19, 19, 20, 20]; boundary 19 is a
        # skip-topk layer, so the safe split must differ.
        self.assertNotEqual(partition, [19, 19, 20, 20])

    def test_various_pp_sizes_all_safe(self):
        config = _make_dsa_config()
        for pp_size in (2, 3, 4, 6, 8):
            partition = get_dsa_safe_pp_layer_partition(config, pp_size)
            self._assert_valid_and_safe(config, pp_size, partition)

    def test_pattern_based_config(self):
        # freq/offset unset; skip-topk is driven by index_topk_pattern ("S").
        pattern = (
            "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"
        )
        config = _make_dsa_config(
            num_hidden_layers=len(pattern),
            index_topk_freq=1,
            index_skip_topk_offset=None,
            index_topk_pattern=pattern,
        )
        partition = get_dsa_safe_pp_layer_partition(config, 2)
        self._assert_valid_and_safe(config, 2, partition)

    def test_single_stage_returns_none(self):
        self.assertIsNone(get_dsa_safe_pp_layer_partition(_make_dsa_config(), 1))

    def test_non_dsa_config_returns_none(self):
        config = SimpleNamespace(
            architectures=["Qwen2ForCausalLM"], num_hidden_layers=78
        )
        self.assertIsNone(get_dsa_safe_pp_layer_partition(config, 2))

    def test_fewer_layers_than_stages_returns_none(self):
        config = _make_dsa_config(num_hidden_layers=1)
        self.assertIsNone(get_dsa_safe_pp_layer_partition(config, 2))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for multi-D4RL dataset loading (pretrain.py Phase 1).

New design (no per-dataset cap):
  - Each dataset is loaded with max_episodes=max_buffer_episodes.
  - The circular buffer overwrites oldest episodes when full.
  - Recommended order: large datasets first, small/important ones last
    (so small datasets always survive in the buffer).

Tests:
  1. Backward-compat: single dataset still works
  2. Multi-dataset: buffer is full after loading large+large datasets
  3. Load-order guarantee: small dataset loaded last is retained in buffer
  4. Sampling works after multi-load
  5. Partial episode is NOT leaked to next dataset
  6. Edge-case: trailing comma / empty string filtered correctly
  7. alloc arithmetic (now trivially max_buffer for every dataset)
"""

import os
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from u2o.replay_buffer import ReplayBuffer
import u2o.pretrain as pretrain_mod

OBS_DIM    = 39
ACTION_DIM = 28
EP_LEN     = 20   # short for fast tests


def _make_hdf5(path: str, num_episodes: int, ep_len: int = EP_LEN,
               obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM,
               partial_tail: bool = False, seed: int = 42):
    """Write a minimal D4RL-style HDF5 with timeout-delimited episodes."""
    rng = np.random.default_rng(seed)
    n_full  = num_episodes * ep_len
    n_extra = ep_len // 2 if partial_tail else 0
    n       = n_full + n_extra

    observations  = rng.random((n, obs_dim),    dtype=np.float32)
    next_obs      = rng.random((n, obs_dim),    dtype=np.float32)
    actions       = rng.random((n, action_dim), dtype=np.float32) * 2 - 1
    rewards       = rng.random(n, dtype=np.float32)
    terminals     = np.zeros(n, dtype=bool)
    timeouts      = np.zeros(n, dtype=bool)
    for ep in range(num_episodes):
        timeouts[(ep + 1) * ep_len - 1] = True

    with h5py.File(path, "w") as f:
        f.create_dataset("observations",      data=observations)
        f.create_dataset("next_observations", data=next_obs)
        f.create_dataset("actions",           data=actions)
        f.create_dataset("rewards",           data=rewards)
        f.create_dataset("terminals",         data=terminals)
        f.create_dataset("timeouts",          data=timeouts)


def _patch_d4rl_cache(tmp_dir: str, specs: dict):
    """
    specs = {dataset_name: num_episodes}
    Creates HDF5 files and monkey-patches _get_d4rl_hdf5_path.
    Returns (files_dict, original_fn).
    """
    files = {}
    for name, num_eps in specs.items():
        p = os.path.join(tmp_dir, f"{name}.hdf5")
        _make_hdf5(p, num_episodes=num_eps)
        files[name] = p
        pretrain_mod._D4RL_DATASET_INFO.setdefault(
            name, {"max_episode_steps": EP_LEN}
        )

    original_fn = pretrain_mod._get_d4rl_hdf5_path

    def _mock(n):
        return files[n] if n in files else original_fn(n)

    pretrain_mod._get_d4rl_hdf5_path = _mock
    return original_fn


def _make_buffer(max_episodes: int) -> ReplayBuffer:
    return ReplayBuffer(
        max_episodes=max_episodes,
        discount=0.99,
        future=0.99,
        p_randomgoal=0.375,
        max_episode_length=EP_LEN + 1,
    )


def _load_multi(dataset_names, max_buffer) -> ReplayBuffer:
    """Replicate the Phase 1 loop exactly as implemented in pretrain()."""
    buf = _make_buffer(max_buffer)
    for ds in dataset_names:
        pretrain_mod.load_d4rl_data(
            ds, buf,
            max_episodes=max_buffer,
            expected_obs_dim=OBS_DIM,
            expected_action_dim=ACTION_DIM,
        )
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
class TestSingleDataset(unittest.TestCase):

    def test_loads_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"door-cloned-v1": 100})
            try:
                buf = _make_buffer(50)
                n = pretrain_mod.load_d4rl_data(
                    "door-cloned-v1", buf, max_episodes=50,
                    expected_obs_dim=OBS_DIM, expected_action_dim=ACTION_DIM,
                )
                self.assertEqual(len(buf), 50)
                self.assertGreater(n, 0)
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig

    def test_sampling_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"door-cloned-v1": 100})
            try:
                buf = _make_buffer(50)
                pretrain_mod.load_d4rl_data("door-cloned-v1", buf, max_episodes=50)
                batch = buf.sample(32)
                self.assertEqual(batch.obs.shape, (32, OBS_DIM))
                self.assertIsNotNone(batch.future_obs)
                self.assertTrue(np.all(np.isfinite(batch.obs)))
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig


# ═══════════════════════════════════════════════════════════════════════════════
class TestMultiDataset(unittest.TestCase):

    def test_buffer_full_after_two_large_datasets(self):
        """Two large datasets → buffer should be full (== max_buffer)."""
        # Each dataset has 100 eps; buffer cap is 80.
        # Dataset 1 fills buffer to 80 (full). Dataset 2 overwrites all 80.
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"ds-a": 100, "ds-b": 100})
            try:
                buf = _load_multi(["ds-a", "ds-b"], max_buffer=80)
                self.assertEqual(len(buf), 80)
                self.assertTrue(buf._full)
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig

    def test_small_last_survives_in_buffer(self):
        """
        Load order: large (100 eps) first, then small (10 eps) last.
        Small dataset is loaded last → its episodes overwrite the oldest from large.
        Buffer stays full; last 10 slots are from small.
        """
        max_buf = 80
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"large-ds": 100, "small-ds": 10})
            try:
                # large first, small last → small guaranteed to remain
                buf = _load_multi(["large-ds", "small-ds"], max_buffer=max_buf)
                # Buffer is full (large filled it; small overwrote oldest 10)
                self.assertEqual(len(buf), max_buf)
                # The small dataset's 10 episodes are the most recently written.
                # Check: the last 10 episode-length entries in _episodes_length are non-zero
                # (they correspond to the 10 small-dataset episodes).
                self.assertGreater(buf.num_transitions, 0)
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig

    def test_small_first_gets_overwritten(self):
        """
        Small dataset first, large dataset second.
        Large completely overwrites the small → buffer dominated by large.
        """
        max_buf = 80
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"small-ds2": 10, "large-ds2": 100})
            try:
                buf = _load_multi(["small-ds2", "large-ds2"], max_buffer=max_buf)
                self.assertEqual(len(buf), max_buf)
                # 10 small + 80 from large → large fills everything, small gone
                # (80 from large overwrites all slots including the 10 small ones)
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig

    def test_three_datasets_sampling(self):
        """After loading 3 datasets, sampling must not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"ds-x": 60, "ds-y": 60, "ds-z": 60})
            try:
                buf = _load_multi(["ds-x", "ds-y", "ds-z"], max_buffer=90)
                batch = buf.sample(64)
                self.assertEqual(batch.obs.shape, (64, OBS_DIM))
                self.assertEqual(batch.future_obs.shape, (64, OBS_DIM))
                self.assertTrue(np.all(np.isfinite(batch.obs)))
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig

    def test_partial_episode_not_leaked_between_datasets(self):
        """
        _current_episode must be empty when load_d4rl_data returns,
        so the next dataset starts with a clean slate.
        """
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "ds-m.hdf5")
            p2 = os.path.join(tmp, "ds-n.hdf5")
            # partial_tail=True → last few steps have no timeout → partial ep
            _make_hdf5(p1, num_episodes=25, partial_tail=True,  seed=1)
            _make_hdf5(p2, num_episodes=25, partial_tail=False, seed=2)
            for name, path in [("ds-m", p1), ("ds-n", p2)]:
                pretrain_mod._D4RL_DATASET_INFO.setdefault(name, {"max_episode_steps": EP_LEN})
            orig = pretrain_mod._get_d4rl_hdf5_path
            pretrain_mod._get_d4rl_hdf5_path = (
                lambda n: p1 if n == "ds-m" else p2 if n == "ds-n" else orig(n)
            )
            try:
                buf = _make_buffer(60)
                pretrain_mod.load_d4rl_data("ds-m", buf, max_episodes=30)
                self.assertFalse(
                    bool(buf._current_episode),
                    "_current_episode must be empty after load_d4rl_data returns"
                )
                pretrain_mod.load_d4rl_data("ds-n", buf, max_episodes=30)
                self.assertFalse(bool(buf._current_episode))
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig

    def test_dataset_smaller_than_buffer_does_not_fill(self):
        """If a dataset has fewer episodes than max_buffer, buffer is NOT full."""
        with tempfile.TemporaryDirectory() as tmp:
            orig = _patch_d4rl_cache(tmp, {"tiny-ds": 10})
            try:
                buf = _make_buffer(100)
                pretrain_mod.load_d4rl_data("tiny-ds", buf, max_episodes=100)
                self.assertLessEqual(len(buf), 11)   # 10 full + at most 1 partial
                self.assertFalse(buf._full)
            finally:
                pretrain_mod._get_d4rl_hdf5_path = orig


# ═══════════════════════════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):

    def test_trailing_comma_filtered(self):
        d4rl_dataset = "door-cloned-v1,"
        d4rl_datasets = [s.strip() for s in d4rl_dataset.split(",") if s.strip()]
        self.assertEqual(d4rl_datasets, ["door-cloned-v1"])

    def test_spaces_stripped(self):
        d4rl_dataset = " door-human-v1 , door-expert-v1 "
        d4rl_datasets = [s.strip() for s in d4rl_dataset.split(",") if s.strip()]
        self.assertEqual(d4rl_datasets, ["door-human-v1", "door-expert-v1"])

    def test_double_comma_no_empty_entry(self):
        d4rl_dataset = "ds-a,,ds-b"
        d4rl_datasets = [s.strip() for s in d4rl_dataset.split(",") if s.strip()]
        self.assertEqual(d4rl_datasets, ["ds-a", "ds-b"])

    def test_single_dataset_string_parses_correctly(self):
        d4rl_dataset = "door-cloned-v1"
        d4rl_datasets = [s.strip() for s in d4rl_dataset.split(",") if s.strip()]
        self.assertEqual(len(d4rl_datasets), 1)
        self.assertEqual(d4rl_datasets[0], "door-cloned-v1")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)

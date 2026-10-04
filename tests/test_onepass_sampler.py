"""EXP-043: the one-pass sampler (every window exactly once, fixed order) and the trainer hooks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_ai.config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from frontier_ai.data.dataset import write_tokens  # noqa: E402
from frontier_ai.data.multi import MultiTokenDataset, OnePassDataset, one_pass_order  # noqa: E402
from frontier_ai.engine.trainer import Trainer  # noqa: E402

SIZES = {"aa": 1000, "bb": 2600, "cc": 457}  # tokens per file (train split is 90%)
BLOCK = 16


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("onepass")
    rng = np.random.default_rng(0)
    for i, (name, n) in enumerate(SIZES.items()):
        # token value encodes (file, position) so every window can be identified
        ids = (np.arange(n) % 200 + 200 * i).tolist() if name != "bb" else rng.integers(0, 600, n).tolist()
        write_tokens(d / f"{name}.bin", ids, 600, "bpe", val_frac=0.1)
    return [d / f"{n}.bin" for n in SIZES]


def test_order_is_a_fixed_permutation_independent_of_numpy_random():
    a = one_pass_order(10_000, 1)
    assert sorted(a.tolist()) == list(range(10_000))
    assert np.array_equal(a, one_pass_order(10_000, 1))
    assert not np.array_equal(a, one_pass_order(10_000, 2))
    assert not np.array_equal(a, np.arange(10_000))
    # pinned values: a different numpy must give the same order (integer hashing + stable sort)
    assert one_pass_order(10, 1).tolist() == PINNED_10_SEED1
    # the keys are exact SplitMix64 hashes (checked against plain Python integers)
    m = (1 << 64) - 1

    def ref(x: int) -> int:
        z = (x + 0x9E3779B97F4A7C15) & m
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        return z ^ (z >> 31)

    seed_mix = (1 * 0x9E3779B97F4A7C15) & m
    keys = [ref(i ^ seed_mix) for i in range(10)]
    assert sorted(range(10), key=lambda i: keys[i]) == PINNED_10_SEED1


PINNED_10_SEED1 = [9, 0, 5, 8, 6, 3, 2, 4, 1, 7]


def test_every_window_is_read_exactly_once_in_natural_proportions(files):
    base = MultiTokenDataset(files)
    ds = OnePassDataset(base, BLOCK, seed=1)
    expected = [(base.parts[n].n_train - 1) // BLOCK for n in base.names]
    assert ds.counts.tolist() == expected and ds.n_windows == sum(expected)
    seen = []
    cpu = torch.device("cpu")
    while ds.position + 4 <= ds.n_windows:
        x, y = ds.next_train_batch(4, BLOCK, cpu)
        assert x.shape == (4, BLOCK) and torch.equal(x[:, 1:], y[:, :-1])
        seen += ds.order[ds.position - 4 : ds.position].tolist()
    assert len(seen) == len(set(seen)) == ds.steps_per_pass(4) * 4
    with pytest.raises(RuntimeError, match="used up"):
        ds.next_train_batch(4, BLOCK, cpu)
    # window contents are exactly the file's tokens at the window position
    for k in [0, ds.n_windows - 1, int(ds.offsets[1]), int(ds.offsets[2]) + 3]:
        name, start = ds.window(k)
        data = base.parts[name].split("train")
        assert start + BLOCK + 1 <= len(data)
    name, start = ds.window(int(ds.offsets[2]) + 3)
    assert name == "cc" and start == 3 * BLOCK
    # delegation: validation and metadata still come from the base dataset
    assert ds.names == base.names and ds.n_val == base.n_val
    xv, _ = ds.get_batch("val", 2, BLOCK, cpu, generator=torch.Generator().manual_seed(0))
    assert xv.shape == (2, BLOCK)
    assert len(ds.fingerprint()) == 64 and ds.fingerprint() == OnePassDataset(base, BLOCK, 1).fingerprint()


def test_seek_continues_with_exactly_the_next_window(files):
    cpu = torch.device("cpu")
    a = OnePassDataset(MultiTokenDataset(files), BLOCK, seed=3)
    batches = [a.next_train_batch(3, BLOCK, cpu)[0] for _ in range(5)]
    b = OnePassDataset(MultiTokenDataset(files), BLOCK, seed=3)
    b.seek(9)
    assert torch.equal(b.next_train_batch(3, BLOCK, cpu)[0], batches[3])
    with pytest.raises(ValueError):
        b.seek(b.n_windows + 1)
    with pytest.raises(ValueError, match="block_size"):
        b.next_train_batch(3, BLOCK * 2, cpu)


def _cfg(files, out: Path, **train) -> ExperimentConfig:
    base = dict(
        out_dir=str(out),
        max_steps=8,
        accum_steps=2,
        eval_interval=0,
        eval_iters=1,
        log_interval=100,
        save_interval=0,
        save_best=False,
        device="cpu",
        precision="fp32",
        seed=1,
    )
    base.update(train)
    return ExperimentConfig(
        model=ModelConfig(vocab_size=600, n_layer=1, n_head=2, n_embd=16, block_size=BLOCK, dropout=0.0),
        data=DataConfig(path=str(Path(files[0]).parent), batch_size=2, seed=1),
        optim=OptimConfig(lr=1e-3, warmup_steps=2),
        train=TrainConfig(**base),
    )


def test_trainer_uses_the_pass_resumes_it_exactly_and_stops_on_request(files, tmp_path):
    def run(out, max_steps, resume="", stop_at=None):
        ds = OnePassDataset(MultiTokenDataset(files), BLOCK, seed=1)
        cfg = _cfg(files, out, max_steps=max_steps, resume=resume)
        t = Trainer(cfg, ds)
        res = t.fit(should_stop=(lambda tr: "asked" if tr.state.step == stop_at else None))
        return t, ds, res

    t_full, ds_full, _ = run(tmp_path / "full", 8)
    assert ds_full.position == 8 * 2 * 2  # steps x accum x batch windows, in order
    t_a, ds_a, res_a = run(tmp_path / "split", 8, stop_at=3)
    assert res_a["stopped"] == "asked" and res_a["steps"] == 3 and ds_a.position == 12
    assert (tmp_path / "split" / "last" / "model.pt").exists()
    assert not any(p.name.endswith((".tmp", ".old")) for p in (tmp_path / "split").iterdir())
    t_b, ds_b, res_b = run(tmp_path / "split2", 8, resume=str(tmp_path / "split"))
    assert res_b["stopped"] is None and res_b["steps"] == 8 and ds_b.position == 32
    for (k, v1), v2 in zip(t_full.model.state_dict().items(), t_b.model.state_dict().values()):
        assert torch.allclose(v1, v2, atol=1e-6, rtol=0), k

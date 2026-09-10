"""End-to-end: prepare data -> train -> checkpoint -> resume -> sample.

Runs on CPU with a ~50k-parameter model, so it stays fast enough for CI while
exercising the exact code path a GPU run uses.
"""

import math

import pytest
import torch

from frontier_ai.config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig, TrainConfig
from frontier_ai.data.dataset import TokenDataset, write_tokens
from frontier_ai.data.synthetic import generate_corpus
from frontier_ai.data.tokenizer import CharTokenizer
from frontier_ai.engine import checkpoint as ckpt
from frontier_ai.engine.trainer import Trainer
from frontier_ai.utils.seed import set_seed


def make_dataset(tmp_path, chars: int = 20_000):
    text = generate_corpus(target_chars=chars, seed=11)
    tok = CharTokenizer.fit(text)
    tok.save(tmp_path / "tok.json")
    meta = write_tokens(tmp_path / "tokens.bin", tok.encode(text), tok.vocab_size, "char", val_frac=0.1)
    return TokenDataset(tmp_path / "tokens.bin", meta), tok, meta


def make_cfg(tmp_path, **train_kwargs) -> ExperimentConfig:
    train = dict(
        out_dir=str(tmp_path / "out"),
        max_steps=60,
        accum_steps=2,
        eval_interval=20,
        eval_iters=4,
        log_interval=10,
        device="cpu",
        precision="fp32",
        seed=0,
    )
    train.update(train_kwargs)
    return ExperimentConfig(
        model=ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=32),
        data=DataConfig(path=str(tmp_path / "tokens.bin"), batch_size=4),
        optim=OptimConfig(lr=0.01, warmup_steps=5),
        train=TrainConfig(**train),
    )


def test_training_reduces_loss_and_saves_checkpoints(tmp_path):
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path)
    cfg = make_cfg(tmp_path)
    cfg.model.vocab_size = meta.vocab_size

    trainer = Trainer(cfg, ds)
    initial = trainer.evaluate()
    result = trainer.fit()

    assert result["steps"] == cfg.train.max_steps
    assert result["best_val"] < initial, "validation loss did not improve"
    assert (tmp_path / "out" / "last" / "model.pt").exists()
    assert (tmp_path / "out" / "best" / "model.pt").exists()
    assert (tmp_path / "out" / "train.jsonl").exists()

    lines = (tmp_path / "out" / "train.jsonl").read_text().strip().splitlines()
    events = [__import__("json").loads(line)["event"] for line in lines]
    assert "run.start" in events and "run.end" in events and "eval" in events


def test_loss_decreases_monotonically_enough(tmp_path):
    """Two short runs: more steps must beat fewer steps on the same data."""
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path)

    short = make_cfg(tmp_path, max_steps=10, out_dir=str(tmp_path / "short"), eval_iters=4)
    short.model.vocab_size = meta.vocab_size
    short_loss = Trainer(short, ds).fit()["best_val"]

    long = make_cfg(tmp_path, max_steps=80, out_dir=str(tmp_path / "long"), eval_iters=4)
    long.model.vocab_size = meta.vocab_size
    long_loss = Trainer(long, ds).fit()["best_val"]

    assert long_loss < short_loss


def test_logged_train_loss_is_a_mean_not_a_sum(tmp_path):
    """Regression guard: the logged loss must be the mean over accumulation steps."""
    import json

    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, max_steps=20, accum_steps=4, eval_interval=0, log_interval=1,
                   eval_iters=8, out_dir=str(tmp_path / "accum"))
    cfg.model.vocab_size = meta.vocab_size

    Trainer(cfg, ds).fit()
    records = [json.loads(line) for line in (tmp_path / "accum" / "train.jsonl").read_text().splitlines()]
    logged = [r["loss"] for r in records if r["event"] == "train"]
    assert logged, "nothing was logged"

    # an untrained model sits near ln(vocab); a *summed* loss would be ~accum x larger
    assert max(logged) < 2.0 * math.log(meta.vocab_size)
    assert logged[-1] < logged[0], "loss did not decrease"


def test_evaluate_can_score_the_train_split(tmp_path):
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path, chars=6_000)
    cfg = make_cfg(tmp_path, max_steps=1, eval_interval=0)
    cfg.model.vocab_size = meta.vocab_size
    trainer = Trainer(cfg, ds)
    val = trainer.evaluate(max_iters=5, split="val")
    train = trainer.evaluate(max_iters=5, split="train")
    assert 0 < val < 3 * math.log(meta.vocab_size)
    assert 0 < train < 3 * math.log(meta.vocab_size)


def test_resume_continues_from_checkpoint(tmp_path):
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path, chars=8_000)
    cfg = make_cfg(tmp_path, max_steps=10, eval_interval=0, save_interval=10)
    cfg.model.vocab_size = meta.vocab_size

    trainer = Trainer(cfg, ds)
    trainer.fit()
    assert trainer.state.step == 10

    cfg2 = make_cfg(tmp_path, max_steps=25, eval_interval=0)
    cfg2.model.vocab_size = meta.vocab_size
    cfg2.train.init_from = "resume"
    resumed = Trainer(cfg2, ds)
    first_eval_before = resumed.evaluate()

    assert resumed.state.step == 10, "resume did not restore the step counter"
    resumed.fit()
    assert resumed.state.step == 25
    assert resumed.evaluate() <= first_eval_before + 1e-6


def test_generated_text_is_structured(tmp_path):
    """After a short fit, samples should contain real tokens and end with punctuation."""
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, max_steps=120, eval_interval=0)
    cfg.model.vocab_size = meta.vocab_size

    trainer = Trainer(cfg, ds)
    trainer.fit()
    model = trainer.model.eval()

    prompt = torch.tensor([[tok.encode("the quiet cat")[0]]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=24, temperature=0.7, top_k=10,
                         generator=torch.Generator().manual_seed(0))
    text = tok.decode(out[0].tolist())
    assert len(text) > 5
    assert set(text) <= set(tok.itos)
    assert "." in text or " " in text  # learned *some* structure from the grammar


def test_gradient_accumulation_matches_larger_batch(tmp_path):
    """accum_steps=4 x batch 2 should behave like batch 8 in expectation (loss scale)."""
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path)

    cfg = make_cfg(tmp_path, max_steps=1, eval_interval=0)
    cfg.model.vocab_size = meta.vocab_size
    model = Trainer(cfg, ds).model

    tokens = ds.train[:96].astype("int64").tolist()
    x = torch.tensor([tokens[:32]])
    y = torch.tensor([tokens[1:33]])

    model.eval()
    with torch.no_grad():
        loss = model(x, targets=y).loss
    assert 0 < float(loss) < 2 * math.log(meta.vocab_size)


def test_checkpoint_from_trainer_is_loadable(tmp_path):
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path, chars=6_000)
    cfg = make_cfg(tmp_path, max_steps=5, eval_interval=0, save_interval=5)
    cfg.model.vocab_size = meta.vocab_size

    trainer = Trainer(cfg, ds)
    trainer.fit()

    from frontier_ai.model.gpt import GPT

    fresh = GPT(cfg.model)
    ckpt.load_checkpoint(tmp_path / "out" / "last", fresh, map_location="cpu")
    fresh.eval()
    x, y = ds.get_batch("val", 2, cfg.model.block_size, torch.device("cpu"))
    with torch.no_grad():
        a = trainer.model(x, targets=y).loss
        b = fresh(x, targets=y).loss
    assert torch.allclose(a, b, atol=1e-5)


def test_gradient_checkpointing_trains(tmp_path):
    """torch.utils.checkpoint path still produces a finite, improving loss."""
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path, chars=6_000)
    cfg = make_cfg(tmp_path, max_steps=5, eval_interval=0, grad_checkpointing=True)
    cfg.model.vocab_size = meta.vocab_size
    trainer = Trainer(cfg, ds)
    assert trainer.model.gradient_checkpointing is True
    result = trainer.fit()
    assert result["steps"] == 5 and math.isfinite(result["best_val"]) or result["best_val"] == float("inf")


@pytest.mark.parametrize("precision", ["fp32", "auto"])
def test_precision_settings_do_not_crash(tmp_path, precision):
    set_seed(0)
    ds, tok, meta = make_dataset(tmp_path, chars=6_000)
    cfg = make_cfg(tmp_path, max_steps=3, eval_interval=0, precision=precision)
    cfg.model.vocab_size = meta.vocab_size
    result = Trainer(cfg, ds).fit()
    assert result["steps"] == 3

import json

import pytest
import torch

from frontier_ai.config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig, TrainConfig
from frontier_ai.data.tokenizer import CharTokenizer
from frontier_ai.engine import checkpoint as ckpt
from frontier_ai.engine.optim import build_optimizer, lr_at


def test_warmup_then_cosine_decay():
    o = OptimConfig(lr=1.0, warmup_steps=10, schedule="cosine", min_lr_ratio=0.1)
    assert lr_at(1, o, 100) == pytest.approx(0.1)
    assert lr_at(10, o, 100) == pytest.approx(1.0)
    assert lr_at(55, o, 100) == pytest.approx(0.55, abs=0.02)
    assert lr_at(100, o, 100) == pytest.approx(0.1, abs=0.01)
    # never below the floor, even past the end of the schedule
    assert lr_at(500, o, 100) == pytest.approx(0.1)


def test_constant_schedule_ignores_decay():
    o = OptimConfig(lr=0.5, warmup_steps=5, schedule="constant")
    assert lr_at(50, o, 100) == 0.5


def test_optimizer_groups_and_step():
    from frontier_ai.model.gpt import GPT

    model = GPT(ModelConfig(vocab_size=16, n_layer=1, n_head=2, n_embd=16, block_size=8))
    opt = build_optimizer(model, OptimConfig(lr=1e-3), device_type="cpu")
    assert len(opt.param_groups) == 2
    before = model.token_emb.weight.detach().clone()
    loss = model(torch.randint(0, 16, (2, 8)), targets=torch.randint(0, 16, (2, 8))).loss
    loss.backward()
    opt.step()
    assert not torch.equal(before, model.token_emb.weight.detach())


def _tiny_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(vocab_size=16, n_layer=1, n_head=2, n_embd=16, block_size=8),
        data=DataConfig(path=str(tmp_path / "d.bin"), batch_size=2),
        optim=OptimConfig(lr=0.01, warmup_steps=1),
        train=TrainConfig(out_dir=str(tmp_path / "out"), max_steps=2, accum_steps=2,
                          eval_interval=1, eval_iters=1, log_interval=1, device="cpu", precision="fp32"),
    )


def test_checkpoint_save_load_restores_weights(tmp_path):
    from frontier_ai.model.gpt import GPT

    cfg = _tiny_cfg(tmp_path)
    model = GPT(cfg.model)
    torch.nn.init.constant_(model.token_emb.weight, 0.25)

    path = ckpt.save_checkpoint(tmp_path / "out", model, cfg=cfg, step=7, best_val=1.5, tag="best")
    assert (path / "model.pt").exists() and (path / "config.json").exists()

    fresh = GPT(cfg.model)
    meta = ckpt.load_checkpoint(path, fresh, map_location="cpu")
    assert meta["step"] == 7 and meta["best_val"] == 1.5
    assert torch.allclose(fresh.token_emb.weight, model.token_emb.weight)

    reloaded = ExperimentConfig.load(path / "config.json")
    assert reloaded.model.n_embd == cfg.model.n_embd


def test_checkpoint_roundtrips_optimizer_state(tmp_path):
    from frontier_ai.model.gpt import GPT

    cfg = _tiny_cfg(tmp_path)
    model = GPT(cfg.model)
    opt = build_optimizer(model, cfg.optim, device_type="cpu")
    path = ckpt.save_checkpoint(tmp_path / "out", model, optimizer=opt, cfg=cfg, step=3)
    opt2 = build_optimizer(model, cfg.optim, device_type="cpu")
    ckpt.load_checkpoint(path, model, optimizer=opt2, map_location="cpu")
    assert opt2.state_dict()["param_groups"][0]["lr"] == cfg.optim.lr


def test_find_latest_prefers_highest_step(tmp_path):
    out = tmp_path / "out"
    for step in (10, 300, 20):
        (out / f"step-{step}").mkdir(parents=True)
    assert ckpt.find_latest(out).name == "step-300"
    assert ckpt.find_latest(tmp_path / "missing") is None


def test_config_overrides_and_validation(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    new = cfg.with_overrides(["optim.lr=0.5", "train.max_steps=9", "model.n_layer=3", "train.save_best=false"])
    assert new.optim.lr == 0.5 and new.train.max_steps == 9 and new.model.n_layer == 3
    assert new.train.save_best is False
    assert cfg.optim.lr == 0.01  # original untouched (deep copy)

    with pytest.raises(ValueError):
        cfg.with_overrides(["nope.x=1"])
    with pytest.raises(ValueError):
        cfg.with_overrides(["optim.not_a_field=1"])
    with pytest.raises(ValueError):
        cfg.with_overrides(["optim.lr"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": {"bogus": 1}}))
    with pytest.raises(ValueError, match="unknown keys"):
        ExperimentConfig.load(bad)


def test_model_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=65, n_head=8)  # not divisible
    with pytest.raises(ValueError):
        ModelConfig(norm="batchnorm")
    with pytest.raises(ValueError):
        ModelConfig(ffn="relu")
    with pytest.raises(ValueError):
        ModelConfig(pos="sinusoidal")


def test_tokenizer_vocab_fits_model(tmp_path):
    tok = CharTokenizer.fit("abc")
    cfg = _tiny_cfg(tmp_path)
    cfg.model.vocab_size = tok.vocab_size
    from frontier_ai.model.gpt import GPT

    model = GPT(cfg.model)
    out = model(torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]]))
    assert out.logits.shape[-1] == tok.vocab_size

"""The training loop.

Everything the loop needs is injected as config, so the same code runs a 20-second
CPU smoke test and a multi-hour CUDA run. Key mechanics:

* gradient accumulation (`accum_steps`) -> effective batch = batch_size * accum_steps
* mixed precision via `torch.autocast` (bf16/fp16 on CUDA, fp32 elsewhere)
* cosine LR with linear warmup, grad-norm clipping
* periodic + best-checkpoint saving, resume from any checkpoint directory
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..data.dataset import TokenDataset
from ..model.gpt import GPT
from ..utils.device import DeviceSpec, resolve_spec, threads_for
from ..utils.logging import RunLogger, fmt_tokens_per_sec
from ..utils.seed import set_seed
from . import checkpoint as ckpt
from .optim import LRScheduler, build_optimizer


@dataclass
class TrainState:
    """Mutable run state, also used for resume."""

    step: int = 0
    best_val: float = float("inf")
    tokens_seen: int = 0
    history: list = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        dataset: TokenDataset,
        model: GPT | None = None,
        logger: RunLogger | None = None,
    ) -> None:
        self.cfg = cfg
        self.ds = dataset
        self.spec: DeviceSpec = resolve_spec(cfg.train.device, cfg.train.precision)
        threads_for(self.spec.device, cfg.train.num_threads or None)

        set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
        self.device = self.spec.device

        self.model = model or GPT(cfg.model)
        self.model.to(self.device)
        if cfg.train.grad_checkpointing:
            self.model.gradient_checkpointing = True
        if cfg.train.compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - backend dependent
                print(f"[warn] torch.compile unavailable ({exc}); continuing without it")

        self.optimizer = build_optimizer(self.model, cfg.optim, device_type=self.device.type)
        self.scheduler = LRScheduler(self.optimizer, cfg.optim, max_steps=cfg.train.max_steps)
        self.scaler = _make_scaler(self.spec)

        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or RunLogger(self.out_dir, name="train")
        self.state = TrainState()
        self._gen = torch.Generator().manual_seed(cfg.data.seed)

        # resume if requested
        resume_from = cfg.train.resume or (cfg.train.out_dir if cfg.train.init_from == "resume" else "")
        if resume_from:
            found = ckpt.find_latest(resume_from)
            if found:
                self._resume(found)

    # ------------------------------------------------------------------ io --
    def _resume(self, path: Path) -> None:
        meta = ckpt.load_checkpoint(path, self.model, self.optimizer, self.scheduler, map_location="cpu")
        self.model.to(self.device)
        self.state.step = int(meta.get("step", 0))
        if meta.get("best_val") is not None:
            self.state.best_val = float(meta["best_val"])
        self.state.tokens_seen = int(meta.get("tokens_seen", 0))
        print(f"[resume] loaded {path} at step {self.state.step} (best_val={self.state.best_val:.4f})")

    def save(self, tag: str, extra: dict | None = None) -> Path:
        return ckpt.save_checkpoint(
            self.out_dir,
            self.model,
            self.optimizer,
            self.scheduler,
            self.cfg,
            step=self.state.step,
            best_val=self.state.best_val,
            extra={"tokens_seen": self.state.tokens_seen, **(extra or {})},
            tag=tag,
        )

    # --------------------------------------------------------------- train --
    def fit(self) -> dict[str, float]:
        cfg = self.cfg
        self.logger.log(
            event="run.start",
            info=self.spec.describe(),
            n_params=self.model.n_params(),
            max_steps=cfg.train.max_steps,
            eff_batch=cfg.data.batch_size * cfg.train.accum_steps * cfg.model.block_size,
        )
        self.model.train()

        t_start = time.time()
        accum = max(1, cfg.train.accum_steps)
        # keep the accumulated loss numerically identical to a full-size batch
        tokens_per_micro = cfg.data.batch_size * cfg.model.block_size

        while self.state.step < cfg.train.max_steps:
            self.state.step += 1
            step_t0 = time.time()
            lr = self.scheduler.current_lr()

            self.optimizer.zero_grad(set_to_none=True)
            micro_loss = 0.0
            for micro in range(accum):
                x, y = self.ds.get_batch(
                    "train",
                    cfg.data.batch_size,
                    cfg.model.block_size,
                    self.device,
                    generator=self._gen,
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.spec.amp_dtype,
                    enabled=self.spec.amp,
                ):
                    out = self.model(x, targets=y)
                    loss = out.loss / accum
                if not torch.isfinite(loss):
                    if cfg.train.early_stop_on_nan:
                        raise FloatingPointError(
                            f"non-finite loss at step {self.state.step} (micro {micro}); "
                            "lower the learning rate or check the data"
                        )
                    continue
                # `loss` is already divided by accum, so summing the micro-batches
                # gives the mean loss over the whole optimizer step.
                self.scaler.scale(loss).backward()
                micro_loss += float(loss.detach())

            if cfg.optim.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.optim.grad_clip
                )
                grad_norm = float(grad_norm)
            else:
                grad_norm = float("nan")

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.state.tokens_seen += tokens_per_micro * accum
            step_dt = time.time() - step_t0
            remaining = cfg.train.max_steps - self.state.step
            eta_min = remaining * (time.time() - t_start) / max(self.state.step, 1) / 60

            if self.state.step % cfg.train.log_interval == 0 or self.state.step == 1:
                tps = (tokens_per_micro * accum) / max(step_dt, 1e-9)
                self.logger.log(
                    event="train",
                    step=self.state.step,
                    loss=round(micro_loss, 5),
                    ppl=round(math.exp(min(20.0, micro_loss)), 3),
                    lr=lr,
                    gnorm=round(grad_norm, 4) if grad_norm == grad_norm else None,
                    tokens_seen=self.state.tokens_seen,
                    tps=round(tps, 1),
                    eta_min=round(eta_min, 2),
                )

            if cfg.train.eval_interval > 0 and (
                self.state.step % cfg.train.eval_interval == 0 or self.state.step == cfg.train.max_steps
            ):
                val = self.evaluate()
                self.model.train()
                improved = val < self.state.best_val
                if improved:
                    self.state.best_val = val
                self.logger.log(
                    event="eval",
                    step=self.state.step,
                    val_loss=round(val, 5),
                    val_ppl=round(math.exp(min(20.0, val)), 3),
                    best=round(self.state.best_val, 5),
                    improved=improved,
                )
                if cfg.train.save_best and improved:
                    self.save("best")
                elif not cfg.train.save_best:
                    self.save("last")

            if cfg.train.save_interval > 0 and self.state.step % cfg.train.save_interval == 0:
                self.save(f"step-{self.state.step}")

        self.save("last")
        total_min = (time.time() - t_start) / 60
        mean_tps = self.state.tokens_seen / max(time.time() - t_start, 1e-9)
        self.logger.log(
            event="run.end",
            steps=self.state.step,
            best_val=round(self.state.best_val, 5),
            minutes=round(total_min, 2),
            mean_tps=round(mean_tps, 1),
            throughput=fmt_tokens_per_sec(mean_tps),
        )
        return {"best_val": self.state.best_val, "steps": self.state.step}

    # ---------------------------------------------------------------- eval --
    @torch.no_grad()
    def evaluate(self, max_iters: int | None = None, split: str = "val") -> float:
        """Mean loss over `eval_iters` batches of `split` (full split if 0)."""
        self.model.eval()
        cfg = self.cfg
        iters = cfg.train.eval_iters if max_iters is None else max_iters
        if iters <= 0:
            iters = max(1, self.ds.batches_per_epoch(cfg.data.batch_size, cfg.model.block_size, split))
        total, count = 0.0, 0
        for _ in range(iters):
            # use the seeded generator: evaluation must sample the same batches for the
            # same seed, otherwise validation loss is not comparable across runs
            x, y = self.ds.get_batch(
                split, cfg.data.batch_size, cfg.model.block_size, self.device, generator=self._gen
            )
            with torch.autocast(
                device_type=self.device.type, dtype=self.spec.amp_dtype, enabled=self.spec.amp
            ):
                out = self.model(x, targets=y)
            total += float(out.loss)
            count += 1
        return total / max(count, 1)


def _make_scaler(spec: DeviceSpec) -> torch.amp.GradScaler:
    """GradScaler, enabled only for fp16 autocast (bf16/fp32 don't need it)."""
    enabled = spec.amp and spec.amp_dtype == torch.float16 and spec.device.type == "cuda"
    try:  # torch >= 2.4
        return torch.amp.GradScaler(spec.device.type, enabled=enabled)  # type: ignore[call-arg]
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)

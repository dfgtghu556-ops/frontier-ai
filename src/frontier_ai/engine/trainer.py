"""The training loop.

Everything the loop needs is injected as config, so the same code runs a 20-second
CPU smoke test and a multi-hour CUDA run. Key mechanics:

* gradient accumulation (`accum_steps`) -> effective batch = batch_size * accum_steps
* mixed precision via `torch.autocast` (bf16/fp16 on CUDA, fp32 elsewhere)
* cosine LR with linear warmup, grad-norm clipping
* periodic + best-checkpoint saving, resume from any checkpoint directory. A checkpoint also
  stores the batch-sampling generator and the fp16 GradScaler (``trainer_state.pt``), so a
  resumed run draws exactly the batches the uninterrupted run would have drawn (EXP-038).
* several GPUs of one machine (EXP-044): when ``torchrun`` started a process group
  (:mod:`.distributed`), the model is wrapped in DistributedDataParallel, each process reads its
  fixed slice of every batch from the one-pass sampler, only process 0 writes files, and
  checkpoints keep the one-GPU format. Without a process group none of this code runs.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..data.dataset import TokenDataset
from ..model.gpt import GPT
from ..utils.device import DeviceSpec, resolve_spec, threads_for
from ..utils.logging import RunLogger, fmt_tokens_per_sec
from ..utils.seed import set_seed
from . import checkpoint as ckpt
from . import distributed as dd
from .metrics import loss_summary
from .optim import LRScheduler, build_optimizer


@dataclass
class TrainState:
    """Mutable run state, also used for resume."""

    step: int = 0
    best_val: float = float("inf")
    tokens_seen: int = 0
    skipped_steps: int = 0  # optimizer steps skipped by the fp16 GradScaler (inf/NaN gradients)
    history: list = field(default_factory=list)


TRAINER_STATE = "trainer_state.pt"


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
        self.raw_model = self.model  # the plain model, whatever wraps it below
        self.rank, self.world = dd.rank(), dd.world()
        if self.world > 1:  # EXP-044: data parallel across the processes torchrun started
            if not hasattr(dataset, "set_shard"):
                raise ValueError("multi-GPU training needs the one-pass sampler (OnePassDataset)")
            dataset.set_shard(self.rank, self.world)
            from torch.nn.parallel import DistributedDataParallel

            ids = [torch.cuda.current_device()] if self.device.type == "cuda" else None
            self.model = DistributedDataParallel(
                self.model, device_ids=ids, broadcast_buffers=False, gradient_as_bucket_view=True
            )
        if cfg.train.compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - backend dependent
                print(f"[warn] torch.compile unavailable ({exc}); continuing without it")

        opt_model = self.model if self.world == 1 else self.raw_model
        self.optimizer = build_optimizer(opt_model, cfg.optim, device_type=self.device.type)
        self.scheduler = LRScheduler(self.optimizer, cfg.optim, max_steps=cfg.train.max_steps)
        self.scaler = _make_scaler(self.spec)

        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if logger is None:
            logger = RunLogger(self.out_dir, name="train") if self.rank == 0 else _QuietLogger()
        self.logger = logger
        # what evaluation and checkpoints use: the model itself on one process; with several, the
        # plain model (no gradient exchange during evaluation) and the one-GPU checkpoint format
        self.eval_model = self.model if self.world == 1 else self.raw_model
        self._ckpt_model = (
            self.model
            if self.world == 1
            else dd.SingleFormat(self.raw_model, compiled=hasattr(self.model, "_orig_mod"))
        )
        self.state = TrainState()
        self._gen = torch.Generator().manual_seed(cfg.data.seed)
        self.stop_reason: str | None = None
        if hasattr(dataset, "seek"):  # a one-pass dataset (EXP-043) starts at its first window
            dataset.seek(0)

        # resume if requested
        resume_from = cfg.train.resume or (cfg.train.out_dir if cfg.train.init_from == "resume" else "")
        if resume_from:
            found = ckpt.find_latest(resume_from)
            if found:
                self._resume(found)

    # ------------------------------------------------------------------ io --
    def _resume(self, path: Path) -> None:
        meta = ckpt.load_checkpoint(
            path, self._ckpt_model, self.optimizer, self.scheduler, map_location="cpu"
        )
        self.model.to(self.device)
        self.state.step = int(meta.get("step", 0))
        if meta.get("best_val") is not None:
            self.state.best_val = float(meta["best_val"])
        self.state.tokens_seen = int(meta.get("tokens_seen", 0))
        self.state.skipped_steps = int(meta.get("skipped_steps", 0))
        extra_state = path / TRAINER_STATE
        if extra_state.exists():  # checkpoints written before EXP-038 do not have it
            blob = ckpt._torch_load(extra_state, "cpu")  # noqa: SLF001 - same package
            self._gen.set_state(blob["data_generator"])
            if blob.get("scaler"):
                self.scaler.load_state_dict(blob["scaler"])
        if hasattr(self.ds, "seek"):  # continue the one pass with exactly the next window
            accum = max(1, self.cfg.train.accum_steps)
            self.ds.seek(self.state.step * accum * self.cfg.data.batch_size * self.world)
        print(f"[resume] loaded {path} at step {self.state.step} (best_val={self.state.best_val:.4f})")

    def save(self, tag: str, extra: dict | None = None) -> Path:
        """Write ``out_dir/tag``. Written to ``tag.tmp`` first and swapped in only when complete,
        so a crash while saving never destroys the previous checkpoint. With several processes only
        process 0 writes."""
        final = self.out_dir / tag
        if self.rank != 0:
            return final
        path = ckpt.save_checkpoint(
            self.out_dir,
            self._ckpt_model,
            self.optimizer,
            self.scheduler,
            self.cfg,
            step=self.state.step,
            best_val=self.state.best_val,
            extra={
                "tokens_seen": self.state.tokens_seen,
                "skipped_steps": self.state.skipped_steps,
                **(extra or {}),
            },
            tag=f"{tag}.tmp",
        )
        torch.save(
            {"data_generator": self._gen.get_state(), "scaler": self.scaler.state_dict()},
            path / TRAINER_STATE,
        )
        old = self.out_dir / f"{tag}.old"
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            final.rename(old)
        path.rename(final)
        if old.exists():
            shutil.rmtree(old)
        return final

    # --------------------------------------------------------------- train --
    def fit(self, should_stop: Callable[[Trainer], str | None] | None = None) -> dict[str, Any]:
        """Train to ``max_steps``. ``should_stop(trainer)`` runs after every optimizer step; a
        non-empty string it returns ends the run early (saved as ``last``, the reason returned as
        ``stopped``)."""
        cfg = self.cfg
        self.stop_reason = None
        self.logger.log(
            event="run.start",
            info=self.spec.describe(),
            n_params=self.raw_model.n_params(),
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
                if hasattr(self.ds, "next_train_batch"):  # one pass in a fixed order (EXP-043)
                    x, y = self.ds.next_train_batch(cfg.data.batch_size, cfg.model.block_size, self.device)
                else:
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
                finite = bool(torch.isfinite(loss))
                if self.world > 1:  # every process must take the same branch (EXP-044)
                    finite = dd.all_true(finite)
                if not finite:
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
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
                grad_norm = float(grad_norm)
            else:
                grad_norm = float("nan")

            scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if scale_before is not None and self.scaler.get_scale() < scale_before:
                self.state.skipped_steps += 1  # the scaler lowers its scale exactly when it skips
            self.scheduler.step()

            self.state.tokens_seen += tokens_per_micro * accum * self.world
            self.last_loss = micro_loss  # this process's share of the step's mean loss
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
                    **self._comparable_fields(val),
                )
                if cfg.train.save_best and improved:
                    self.save("best")
                elif not cfg.train.save_best:
                    self.save("last")

            if cfg.train.save_interval > 0 and self.state.step % cfg.train.save_interval == 0:
                self.save(f"step-{self.state.step}")

            if should_stop is not None:
                reason = should_stop(self)
                if reason:
                    self.stop_reason = reason
                    break

        self.save("last")
        total_min = (time.time() - t_start) / 60
        mean_tps = self.state.tokens_seen / max(time.time() - t_start, 1e-9)
        best_bpb = self.loss_report(self.state.best_val)["bits_per_byte"]
        self.logger.log(
            event="run.end",
            steps=self.state.step,
            best_val=round(self.state.best_val, 5),
            best_bpb=best_bpb,
            minutes=round(total_min, 2),
            mean_tps=round(mean_tps, 1),
            throughput=fmt_tokens_per_sec(mean_tps),
            skipped_steps=self.state.skipped_steps,
            stopped=self.stop_reason,
        )
        return {
            "best_val": self.state.best_val,
            "steps": self.state.step,
            "best_bpb": best_bpb,
            "skipped_steps": self.state.skipped_steps,
            "stopped": self.stop_reason,
        }

    # ------------------------------------------------------------ reporting --
    def loss_report(self, nats: float, split: str = "val") -> dict[str, float | None]:
        """Per-token loss *and* the tokenizer-independent bits per byte/character.

        The per-token numbers are what the optimiser sees; the per-byte and
        per-character numbers are the ones that can be compared between a
        char-level, word-level or BPE corpus. They are ``None`` when the corpus
        does not record byte/character counts.
        """
        return loss_summary(
            nats,
            tokens_per_byte=self.ds.tokens_per_byte(split),
            tokens_per_char=self.ds.tokens_per_char(split),
        )

    def _comparable_fields(self, nats: float, split: str = "val") -> dict[str, float]:
        """Log-friendly bits-per-byte/char, present only when they are known."""
        report = self.loss_report(nats, split)
        return {k: v for k, v in report.items() if k.startswith("bits_per_") and v is not None}

    # ---------------------------------------------------------------- eval --
    @torch.no_grad()
    def evaluate(self, max_iters: int | None = None, split: str = "val") -> float:
        """Mean loss over `eval_iters` batches of `split` (full split if 0)."""
        model = self.eval_model
        model.eval()
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
                out = model(x, targets=y)
            total += float(out.loss)
            count += 1
        return total / max(count, 1)


class _QuietLogger:
    """The logger of processes other than 0: they train, process 0 reports (EXP-044)."""

    def log(self, **fields: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _make_scaler(spec: DeviceSpec) -> torch.amp.GradScaler:
    """GradScaler, enabled only for fp16 autocast (bf16/fp32 don't need it)."""
    enabled = spec.amp and spec.amp_dtype == torch.float16 and spec.device.type == "cuda"
    try:  # torch >= 2.4
        return torch.amp.GradScaler(spec.device.type, enabled=enabled)  # type: ignore[call-arg]
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)

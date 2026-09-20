"""Three-phase training loop for the edge student.

Phase plan (this ordering is the recipe, not a suggestion):

  Phase 1 - Broad supervised pretraining.  All data (lab + field), heavy field
            augmentation including background replacement, label smoothing.
            Purpose: give the trunk a usable feature space cheaply. Lab images
            are allowed here and only here to dominate.

  Phase 2 - Distillation.  Field-heavy sampling, teacher soft targets on every
            batch, plus a stream of *unlabeled* field photos supervised by the
            teacher alone. This is where the cross-domain robustness comes from.
            Expect the biggest single jump in field macro-F1 here.

  Phase 3 - Field-only fine-tune.  Low LR, backbone BatchNorm statistics
            re-estimated on field data (adaptive BN). Cheap and reliably worth
            1-3 points; target-aware BN adaptation is one of the few simple
            mitigations that measurably survives controlled-to-field shift.

Then: calibration and OOD threshold fitting (inference/calibration.py,
inference/ood.py). Those are *post-hoc* and must be fitted on the field
validation split, never on test.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.student import AgriShieldStudent
from ..models.teacher import TeacherModel
from .losses import DistillationLoss, HierarchicalNLL, LossWeights, cutmix_within_crop


@dataclass
class TrainConfig:
    epochs: int = 20
    backbone_lr: float = 3e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    min_lr_factor: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    cutmix_alpha: float = 0.2
    cutmix_probability: float = 0.3
    distill_temperature: float = 3.0
    weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 50
    device: str = "cuda"


class ModelEMA:
    """Exponential moving average of weights.

    Worth the four lines: on noisy merged datasets the EMA weights are
    consistently better calibrated than the final-step weights, and calibration
    is a first-class requirement here, not a nicety.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # Ramp the decay so early steps are not dominated by random init.
        decay = min(self.decay, (1 + step) / (10 + step))
        for ema_param, param in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_param.dtype.is_floating_point:
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            else:
                ema_param.copy_(param)


def cosine_schedule_with_warmup(
    step: int, total_steps: int, warmup_steps: int, min_factor: float
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))


def _cycle(loader: Optional[DataLoader]) -> Iterator:
    if loader is None:
        while True:
            yield None
    while True:
        for batch in loader:
            yield batch


def train_student(
    student: AgriShieldStudent,
    train_loader: DataLoader,
    config: TrainConfig,
    teacher: Optional[TeacherModel] = None,
    unlabeled_loader: Optional[DataLoader] = None,
    val_fn: Optional[callable] = None,
    checkpoint_dir: str | Path = "checkpoints",
) -> Dict[str, float]:
    """Run one phase. Pass ``teacher=None`` for Phase 1, a teacher for Phase 2/3."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()

    criterion = HierarchicalNLL(label_smoothing=0.1, crop_weight=0.3)
    distiller = DistillationLoss(temperature=config.distill_temperature)

    optimiser = torch.optim.AdamW(
        student.param_groups(config.backbone_lr, config.head_lr, config.weight_decay)
    )
    base_lrs = [group["lr"] for group in optimiser.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    ema = ModelEMA(student, config.ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    unlabeled_iter = _cycle(unlabeled_loader)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_metric = -1.0
    history: Dict[str, float] = {}

    for epoch in range(config.epochs):
        student.train()
        running = 0.0
        epoch_start = time.time()

        for batch_index, batch in enumerate(train_loader):
            factor = cosine_schedule_with_warmup(
                global_step, total_steps, warmup_steps, config.min_lr_factor
            )
            for group, base in zip(optimiser.param_groups, base_lrs):
                group["lr"] = base * factor

            images = batch["image"].to(device, non_blocking=True)
            class_target = batch["class_index"].to(device, non_blocking=True)
            crop_target = batch["crop_index"].to(device, non_blocking=True)

            target_b, lam = class_target, torch.tensor(1.0, device=device)
            if config.cutmix_alpha > 0 and torch.rand(1).item() < config.cutmix_probability:
                images, class_target, target_b, lam = cutmix_within_crop(
                    images, class_target, crop_target, config.cutmix_alpha
                )

            with torch.autocast(device.type, enabled=config.amp and device.type == "cuda"):
                out = student(images)
                loss_a = criterion(out.class_log_probs, out.crop_logits, class_target, crop_target)
                loss_b = criterion(out.class_log_probs, out.crop_logits, target_b, crop_target)
                loss = config.weights.supervised * (lam * loss_a + (1 - lam) * loss_b)

                if config.weights.prototype_separation > 0:
                    loss = loss + config.weights.prototype_separation * (
                        student.head.prototype.separation_loss()
                    )

                if teacher is not None:
                    with torch.no_grad():
                        teacher_out = teacher(images)
                    loss = loss + config.weights.distillation * distiller(
                        out.class_logits,
                        teacher_out.class_logits,
                        student.project(out.embedding),
                        teacher_out.embedding,
                    )

                    unlabeled = next(unlabeled_iter)
                    if unlabeled is not None:
                        u_images = unlabeled["image"].to(device, non_blocking=True)
                        with torch.no_grad():
                            u_teacher = teacher(u_images)
                        u_out = student(u_images)
                        loss = loss + config.weights.distillation * distiller(
                            u_out.class_logits,
                            u_teacher.class_logits,
                            student.project(u_out.embedding),
                            u_teacher.embedding,
                        )

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            ema.update(student, global_step)

            running += float(loss.detach())
            global_step += 1
            if batch_index % config.log_every == 0:
                print(
                    f"  epoch {epoch:>2} step {batch_index:>5}/{steps_per_epoch} "
                    f"loss {running / (batch_index + 1):.4f} lr {optimiser.param_groups[0]['lr']:.2e}"
                )

        message = (
            f"[epoch {epoch}] loss {running / steps_per_epoch:.4f} "
            f"({time.time() - epoch_start:.0f}s)"
        )
        if val_fn is not None:
            metrics = val_fn(ema.module)
            history.update(metrics)
            message += " | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            score = metrics.get("field_macro_f1", metrics.get("macro_f1", 0.0))
            if score > best_metric:
                best_metric = score
                torch.save(
                    {"model": ema.module.state_dict(), "config": student.config, "metrics": metrics},
                    checkpoint_dir / "student_best.pt",
                )
                message += "  <- best"
        print(message)

    torch.save({"model": ema.module.state_dict(), "config": student.config},
               checkpoint_dir / "student_last.pt")
    student.load_state_dict(ema.module.state_dict())
    return history


@torch.no_grad()
def recalibrate_batchnorm(model: nn.Module, loader: DataLoader, device: str, batches: int = 100) -> None:
    """Adaptive BN: re-estimate running statistics on the target (field) domain.

    No gradients, no labels, a couple of minutes of compute. It is one of the
    few simple domain-shift mitigations that reliably helps rather than hurts;
    adversarial (DANN-style) adaptation has been shown to *degrade* accuracy on
    this exact controlled-to-field transfer, so it is not offered here.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average
    model.train()
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        model(batch["image"].to(device))
    model.eval()


__all__ = ["TrainConfig", "ModelEMA", "train_student", "recalibrate_batchnorm"]

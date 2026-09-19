"""End-to-end driver: manifest -> splits -> teacher -> distill -> calibrate -> export.

Run it in stages, not all at once. Each stage writes an artifact the next one
reads, so a Colab disconnect costs you one stage, not the day.

    python -m scripts.run_pipeline manifest   --data-root /content/data
    python -m scripts.run_pipeline teacher    --data-root /content/data
    python -m scripts.run_pipeline phase1     --data-root /content/data
    python -m scripts.run_pipeline phase2     --data-root /content/data
    python -m scripts.run_pipeline phase3     --data-root /content/data
    python -m scripts.run_pipeline calibrate  --data-root /content/data
    python -m scripts.run_pipeline export     --data-root /content/data

Expected layout under ``--data-root``::

    raw/plantvillage/<class>/*.jpg
    raw/plantdoc/<class>/*.jpg
    raw/plantwild/<class>/*.jpg
    raw/plantseg/images/<class>/*.jpg  +  raw/plantseg/masks/<class>/*.png
    raw/rice_leaf_diseases/<class>/*.jpg
    raw/unlabeled_field/*.jpg          (your own phone photos - no labels needed)
    raw/backgrounds/*.jpg              (soil, sky, canopy, hands, mulch)
    raw/ood/non_leaf/*.jpg             (anything that is not a plant)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from agrishield.data import (
    EvalTransform,
    SourceSpec,
    TrainTransform,
    UnifiedPlantDataset,
    UnlabeledImageDataset,
    assign_group_keys,
    build_manifest,
    collect_backgrounds,
    group_aware_split,
    verify_no_leakage,
)
from agrishield.eval import EvaluationBundle, classification_report, confusable_confusion
from agrishield.export import ExportManifest, benchmark_latency, export_onnx, export_tflite
from agrishield.inference.calibration import (
    TemperatureScaler,
    coverage_risk_curve,
    expected_calibration_error,
    threshold_for_target_risk,
)
from agrishield.inference.ood import OODScorer, energy_score
from agrishield.models import (
    AgriShieldStudent,
    FrozenEncoderBank,
    StudentConfig,
    TeacherConfig,
    TeacherModel,
    build_student,
)
from agrishield.taxonomy import load_taxonomy
from agrishield.train import TrainConfig, recalibrate_batchnorm, train_student

ARTIFACTS = Path("artifacts")
IMAGE_SIZE = 224


# --------------------------------------------------------------------------
def make_sources(root: Path) -> list[SourceSpec]:
    raw = root / "raw"
    candidates = [
        SourceSpec("plantvillage", raw / "plantvillage", domain="lab"),
        SourceSpec("plantdoc", raw / "plantdoc", domain="field"),
        SourceSpec("plantwild", raw / "plantwild", domain="field"),
        SourceSpec("plantseg", raw / "plantseg" / "images", domain="field",
                   mask_root=raw / "plantseg" / "masks"),
        SourceSpec("rice_leaf_diseases", raw / "rice_leaf_diseases", domain="field"),
        SourceSpec("cotton_disease", raw / "cotton_disease", domain="field"),
        SourceSpec("sugarcane_disease", raw / "sugarcane_disease", domain="field"),
        SourceSpec("mango_leaf", raw / "mango_leaf", domain="field"),
        SourceSpec("chilli_disease", raw / "chilli_disease", domain="field"),
        SourceSpec("plant_pathology_2021", raw / "plant_pathology_2021", domain="field"),
    ]
    present = [s for s in candidates if s.root.exists()]
    print(f"[sources] using {len(present)}/{len(candidates)}: {[s.name for s in present]}")
    return present


def cmd_manifest(args) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    print(taxonomy.describe())
    df = build_manifest(make_sources(Path(args.data_root)), taxonomy, strict=False)
    df = assign_group_keys(df)
    df = group_aware_split(df, field_only_test=True, seed=args.seed)
    verify_no_leakage(df)
    ARTIFACTS.mkdir(exist_ok=True)
    df.to_csv(ARTIFACTS / "manifest.csv", index=False)
    print(f"[manifest] -> {ARTIFACTS / 'manifest.csv'}")


# --------------------------------------------------------------------------
def _loaders(args, taxonomy, train_transform, eval_transform):
    df = pd.read_csv(ARTIFACTS / "manifest.csv").fillna({"mask_path": ""})
    backgrounds = collect_backgrounds([Path(args.data_root) / "raw" / "backgrounds"])

    train_df = df[df.split == "train"]
    if args.field_only:
        train_df = train_df[train_df.domain == "field"]

    train_ds = UnifiedPlantDataset(train_df, train_transform, backgrounds)
    weights = train_ds.sampling_weights(taxonomy.n_classes, power=0.5)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        UnifiedPlantDataset(df[(df.split == "val") & (df.domain == "field")], eval_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )
    test_loader = DataLoader(
        UnifiedPlantDataset(df[df.split == "test"], eval_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )
    return df, train_loader, val_loader, test_loader, backgrounds


def _student(taxonomy, args) -> AgriShieldStudent:
    return build_student(
        StudentConfig(
            backbone=args.backbone,
            n_classes=taxonomy.n_classes,
            n_crops=taxonomy.n_crops,
            class_to_crop=taxonomy.class_to_crop,
        )
    )


def _teacher(taxonomy, args) -> TeacherModel:
    bank = FrozenEncoderBank(TeacherConfig(backbones=args.teacher_backbones))
    return TeacherModel(bank, taxonomy.n_classes, taxonomy.n_crops, taxonomy.class_to_crop)


@torch.no_grad()
def _collect_outputs(model, loader, device):
    """Returns logits, cosine scores, targets for a whole loader."""
    model.eval().to(device)
    logits, cosines, targets = [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(out.class_logits.cpu())
        cosines.append(out.max_prototype_cosine.cpu())
        targets.append(batch["class_index"])
    return torch.cat(logits), torch.cat(cosines), torch.cat(targets)


def _make_val_fn(taxonomy, val_loader, device):
    def val_fn(model):
        logits, _, targets = _collect_outputs(model, val_loader, device)
        predictions = logits.argmax(-1).numpy()
        report = classification_report(targets.numpy(), predictions, taxonomy.index_to_class)
        return {"field_macro_f1": report.macro_f1, "field_acc": report.accuracy}

    return val_fn


# --------------------------------------------------------------------------
def cmd_teacher(args) -> None:
    """Train the frozen-backbone head. Minutes, not hours, once features cache."""
    taxonomy = load_taxonomy(args.taxonomy)
    device = args.device
    eval_transform = EvalTransform(IMAGE_SIZE)
    _, train_loader, val_loader, _, _ = _loaders(args, taxonomy, eval_transform, eval_transform)

    teacher = _teacher(taxonomy, args).to(device)
    optimiser = torch.optim.AdamW(teacher.head.parameters(), lr=1e-3, weight_decay=0.05)
    criterion = torch.nn.NLLLoss()

    for epoch in range(args.teacher_epochs):
        teacher.train()
        total = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            target = batch["class_index"].to(device)
            out = teacher(images)
            loss = criterion(out.class_log_probs, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss)
        metrics = _make_val_fn(taxonomy, val_loader, device)(teacher)
        print(f"[teacher] epoch {epoch} loss={total / len(train_loader):.4f} "
              f"field_macro_f1={metrics['field_macro_f1']:.4f}")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"head": teacher.head.state_dict()}, ARTIFACTS / "teacher_head.pt")
    print(f"[teacher] -> {ARTIFACTS / 'teacher_head.pt'}")


def cmd_phase1(args) -> None:
    """Broad supervised pretraining on everything."""
    taxonomy = load_taxonomy(args.taxonomy)
    _, train_loader, val_loader, _, _ = _loaders(
        args, taxonomy, TrainTransform(IMAGE_SIZE, bg_probability=0.5), EvalTransform(IMAGE_SIZE)
    )
    student = _student(taxonomy, args)
    train_student(
        student, train_loader,
        TrainConfig(epochs=args.epochs, device=args.device),
        val_fn=_make_val_fn(taxonomy, val_loader, args.device),
        checkpoint_dir=ARTIFACTS / "phase1",
    )


def cmd_phase2(args) -> None:
    """Distillation from the frozen teacher, including unlabeled field photos."""
    taxonomy = load_taxonomy(args.taxonomy)
    _, train_loader, val_loader, _, _ = _loaders(
        args, taxonomy, TrainTransform(IMAGE_SIZE, bg_probability=0.6), EvalTransform(IMAGE_SIZE)
    )
    student = _student(taxonomy, args)
    student.load_state_dict(torch.load(ARTIFACTS / "phase1" / "student_best.pt")["model"])

    teacher = _teacher(taxonomy, args)
    teacher.head.load_state_dict(torch.load(ARTIFACTS / "teacher_head.pt")["head"])

    unlabeled_root = Path(args.data_root) / "raw" / "unlabeled_field"
    unlabeled_loader = None
    if unlabeled_root.exists():
        unlabeled_loader = DataLoader(
            UnlabeledImageDataset([unlabeled_root], TrainTransform(IMAGE_SIZE, bg_probability=0.0)),
            batch_size=args.batch_size // 2, shuffle=True, num_workers=args.workers, drop_last=True,
        )
        print(f"[phase2] {len(unlabeled_loader.dataset)} unlabeled field photos in the mix")

    train_student(
        student, train_loader,
        TrainConfig(epochs=args.epochs, backbone_lr=2e-4, device=args.device),
        teacher=teacher, unlabeled_loader=unlabeled_loader,
        val_fn=_make_val_fn(taxonomy, val_loader, args.device),
        checkpoint_dir=ARTIFACTS / "phase2",
    )


def cmd_phase3(args) -> None:
    """Field-only fine-tune + adaptive BatchNorm."""
    args.field_only = True
    taxonomy = load_taxonomy(args.taxonomy)
    _, train_loader, val_loader, _, _ = _loaders(
        args, taxonomy, TrainTransform(IMAGE_SIZE, bg_probability=0.35), EvalTransform(IMAGE_SIZE)
    )
    student = _student(taxonomy, args)
    student.load_state_dict(torch.load(ARTIFACTS / "phase2" / "student_best.pt")["model"])
    student.to(args.device)

    recalibrate_batchnorm(student, train_loader, args.device, batches=80)
    train_student(
        student, train_loader,
        TrainConfig(epochs=max(3, args.epochs // 3), backbone_lr=5e-5, head_lr=2e-4,
                    warmup_epochs=1, device=args.device),
        val_fn=_make_val_fn(taxonomy, val_loader, args.device),
        checkpoint_dir=ARTIFACTS / "phase3",
    )


def cmd_calibrate(args) -> None:
    """Fit temperature, decision thresholds and OOD calibration on field val."""
    taxonomy = load_taxonomy(args.taxonomy)
    device = args.device
    eval_transform = EvalTransform(IMAGE_SIZE)
    df, _, val_loader, test_loader, _ = _loaders(args, taxonomy, eval_transform, eval_transform)

    student = _student(taxonomy, args)
    student.load_state_dict(torch.load(ARTIFACTS / "phase3" / "student_best.pt")["model"])

    val_logits, val_cosine, val_targets = _collect_outputs(student, val_loader, device)
    scaler = TemperatureScaler().fit(val_logits, val_targets)
    temperature = scaler.temperature

    calibrated = torch.softmax(val_logits / temperature, dim=-1).numpy()
    confident_threshold, coverage = threshold_for_target_risk(
        calibrated, val_targets.numpy(), target_risk=args.target_risk
    )
    tentative_threshold, _ = threshold_for_target_risk(
        calibrated, val_targets.numpy(), target_risk=min(0.45, args.target_risk * 2.5)
    )
    print(f"[calibrate] confident>={confident_threshold:.3f} (coverage {coverage:.1%}, "
          f"risk<={args.target_risk:.0%}) | tentative>={tentative_threshold:.3f}")

    # OOD: in-distribution = field val; OOD sets = whatever you have on disk.
    id_energy = energy_score(val_logits).numpy()
    ood_sets = {}
    ood_root = Path(args.data_root) / "raw" / "ood"
    for subdir in sorted(p for p in ood_root.glob("*") if p.is_dir()) if ood_root.exists() else []:
        loader = DataLoader(
            UnlabeledImageDataset([subdir], eval_transform), batch_size=args.batch_size,
            num_workers=args.workers,
        )
        logits, cosines = [], []
        student.eval().to(device)
        with torch.no_grad():
            for batch in loader:
                out = student(batch["image"].to(device))
                logits.append(out.class_logits.cpu())
                cosines.append(out.max_prototype_cosine.cpu())
        if logits:
            ood_sets[subdir.name] = (
                energy_score(torch.cat(logits)).numpy(), torch.cat(cosines).numpy()
            )

    scorer = OODScorer().fit(id_energy, val_cosine.numpy(), ood_sets, target_tpr=0.95)

    # Final test-set report.
    test_logits, _, test_targets = _collect_outputs(student, test_loader, device)
    test_probs = torch.softmax(test_logits / temperature, dim=-1).numpy()
    report = classification_report(
        test_targets.numpy(), test_probs.argmax(1), taxonomy.index_to_class
    )
    bundle = EvaluationBundle(
        field_report=report,
        ece_before=expected_calibration_error(torch.softmax(test_logits, -1).numpy(), test_targets.numpy()),
        ece_after=expected_calibration_error(test_probs, test_targets.numpy()),
        temperature=temperature,
        coverage_risk=coverage_risk_curve(test_probs, test_targets.numpy()),
        ood_auroc=scorer.calibration.measured_auroc,
        latency_ms=benchmark_latency(student, IMAGE_SIZE),
    )
    print(bundle.headline())

    blocks = confusable_confusion(
        test_targets.numpy(), test_probs.argmax(1), taxonomy.index_to_class, taxonomy.confusion_groups()
    )
    for anchor, block in list(blocks.items())[:6]:
        print(f"\nlook-alike block: {anchor}\n{block}")

    (ARTIFACTS / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "confident_threshold": confident_threshold,
                "tentative_threshold": tentative_threshold,
                "ood": scorer.calibration.to_dict(),
                "field_macro_f1": report.macro_f1,
                "ece_after": bundle.ece_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[calibrate] -> {ARTIFACTS / 'calibration.json'}")


def cmd_export(args) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    student = _student(taxonomy, args)
    student.load_state_dict(torch.load(ARTIFACTS / "phase3" / "student_best.pt")["model"])
    calibration = json.loads((ARTIFACTS / "calibration.json").read_text())

    eval_transform = EvalTransform(IMAGE_SIZE)
    _, _, val_loader, _, _ = _loaders(args, taxonomy, eval_transform, eval_transform)
    representative = []
    for batch in val_loader:
        representative.extend(batch["image"].numpy())
        if len(representative) >= 300:
            break

    onnx_path = export_onnx(student, ARTIFACTS / "agrishield_student.onnx", IMAGE_SIZE)
    tflite_path = export_tflite(
        student, ARTIFACTS / "agrishield_student.tflite", IMAGE_SIZE, representative, quantize=True
    )

    size_mb = (tflite_path or onnx_path).stat().st_size / 1e6
    ood = calibration["ood"]
    ExportManifest(
        model_name=args.backbone,
        input_size=IMAGE_SIZE,
        input_layout="NCHW float32 in [0,1]; normalisation is inside the graph",
        class_ids=taxonomy.index_to_class,
        crop_ids=[c.id for c in taxonomy.crops],
        class_to_crop=taxonomy.class_to_crop,
        temperature=calibration["temperature"],
        confident_threshold=calibration["confident_threshold"],
        tentative_threshold=calibration["tentative_threshold"],
        ood_threshold=ood["reject_threshold"],
        ood_energy_mean=ood["energy_mean"],
        ood_energy_std=ood["energy_std"],
        ood_cosine_mean=ood["cosine_mean"],
        ood_cosine_std=ood["cosine_std"],
        field_macro_f1=calibration["field_macro_f1"],
        ece=calibration["ece_after"],
        notes="Field-only test metrics. Lab (PlantVillage) numbers are not reported as headline.",
    ).save(ARTIFACTS / "model_manifest.json")
    print(f"[export] manifest written; deployable size {size_mb:.2f} MB")


COMMANDS = {
    "manifest": cmd_manifest,
    "teacher": cmd_teacher,
    "phase1": cmd_phase1,
    "phase2": cmd_phase2,
    "phase3": cmd_phase3,
    "calibrate": cmd_calibrate,
    "export": cmd_export,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--backbone", default="mnv4s")
    parser.add_argument("--teacher-backbones", nargs="+", default=["dinov2_b"])
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-risk", type=float, default=0.15)
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()

"""Self-contained smoke test. Run it after any edit; it needs no dataset.

    python -m scripts.smoke_test

Covers the parts that are easy to get subtly wrong: the hierarchical
normalisation identity, prototype OOD scoring, calibration monotonicity, the
leakage guard, the wash-off physics, and the fusion guardrails.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta

import numpy as np
import torch

from agrishield.eval.metrics import classification_report, macro_f1
from agrishield.fusion.late_fusion import FusionConfig, fuse
from agrishield.fusion.risk_priors import DiseaseRiskModel
from agrishield.fusion.weather import WeatherSeries
from agrishield.inference.calibration import (
    TemperatureScaler,
    coverage_risk_curve,
    expected_calibration_error,
    threshold_for_target_risk,
)
from agrishield.inference.ood import OODScorer, auroc, energy_score, fpr_at_tpr
from agrishield.kinetics.residue import (
    DepositType,
    PesticideRecord,
    ResidueKineticsEngine,
    SprayEvent,
)
from agrishield.models.heads import HierarchicalHybridHead
from agrishield.taxonomy import load_taxonomy

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(f"{name} {detail}".strip())
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


# --------------------------------------------------------------------------
def test_taxonomy() -> None:
    print("\ntaxonomy")
    tax = load_taxonomy("configs/taxonomy.yaml")
    check("loads", tax.n_classes > 20, f"({tax.n_classes} classes, {tax.n_crops} crops)")
    check(
        "alias mapping",
        tax.map_raw("plantvillage", "Tomato___Early_blight") == "tomato::early_blight",
    )
    check(
        "whitespace/case insensitive",
        tax.map_raw("plantdoc", "tomato early blight leaf") == "tomato::early_blight",
    )
    check("unknown label raises", _raises(lambda: tax.map_raw("plantdoc", "banana mystery")))
    check("crop indices consistent", len(tax.class_to_crop) == tax.n_classes)
    check("lab-only quarantine non-empty", len(tax.lab_only_indices()) > 0,
          f"({len(tax.lab_only_indices())} classes)")
    return tax


def test_hierarchical_head(tax) -> None:
    print("\nhierarchical head")
    torch.manual_seed(0)
    head = HierarchicalHybridHead(
        dim=64, n_classes=tax.n_classes, n_crops=tax.n_crops,
        class_to_crop=tax.class_to_crop, n_prototypes=4,
    ).eval()
    out = head(torch.randn(8, 64))

    total = out.class_log_probs.exp().sum(dim=-1)
    check("log-probs normalise to 1", torch.allclose(total, torch.ones(8), atol=1e-4),
          f"(sum={total.mean():.6f})")

    # p(crop) must equal the sum of p(c) over that crop's classes.
    probs = out.class_log_probs.exp()
    crop_probs = torch.softmax(out.crop_logits, dim=-1)
    aggregated = torch.stack(
        [probs[:, head.group_mask[g]].sum(dim=-1) for g in range(tax.n_crops)], dim=-1
    )
    check("marginalises to crop posterior", torch.allclose(aggregated, crop_probs, atol=1e-4),
          f"(max diff {float((aggregated - crop_probs).abs().max()):.2e})")

    check("cosine score in range",
          bool((out.max_prototype_cosine.abs() <= 1.0001).all()),
          f"(max {float(out.max_prototype_cosine.max()):.3f})")
    check("prototype separation loss finite", torch.isfinite(head.prototype.separation_loss()).item())

    # Export-friendliness: the head must trace.
    try:
        torch.jit.trace(head.prototype, torch.randn(2, 64))
        traceable = True
    except Exception as exc:  # pragma: no cover
        traceable = False
        print(f"      trace error: {exc}")
    check("prototype head is traceable", traceable)


def test_calibration() -> None:
    print("\ncalibration")
    torch.manual_seed(1)
    n, c = 4000, 12
    targets = torch.randint(0, c, (n,))
    logits = torch.randn(n, c)
    logits[torch.arange(n), targets] += 1.4
    overconfident = logits * 3.0  # simulate an over-sharp model

    before = expected_calibration_error(torch.softmax(overconfident, -1).numpy(), targets.numpy())
    scaler = TemperatureScaler().fit(overconfident, targets, verbose=False)
    after = expected_calibration_error(
        torch.softmax(overconfident / scaler.temperature, -1).numpy(), targets.numpy()
    )
    check("temperature > 1 for over-sharp logits", scaler.temperature > 1.2, f"(T={scaler.temperature:.2f})")
    check("ECE improves", after < before, f"({before:.4f} -> {after:.4f})")

    probs = torch.softmax(overconfident / scaler.temperature, -1).numpy()
    check("argmax unchanged by scaling",
          bool((probs.argmax(1) == torch.softmax(overconfident, -1).numpy().argmax(1)).all()))

    coverage, risk = coverage_risk_curve(probs, targets.numpy())
    check("risk rises with coverage", risk[-1] >= risk[0] - 1e-9,
          f"(risk@10%={risk[4]:.3f} -> risk@100%={risk[-1]:.3f})")

    threshold, cov = threshold_for_target_risk(probs, targets.numpy(), target_risk=0.15)
    check("threshold search returns a usable operating point", 0.0 < cov <= 1.0,
          f"(t={threshold:.3f}, coverage={cov:.1%})")


def test_ood() -> None:
    print("\nopen-set scoring")
    torch.manual_seed(2)
    n = 2000

    # Synthetic that reflects the actual mechanism. Under OOD input, feature
    # norms collapse, so the whole logit vector is attenuated *and shifted down*.
    # Softmax is invariant to an additive shift, so MSP can only see the
    # flattening; energy sees the magnitude loss too. That asymmetry is exactly
    # why energy beats MSP in practice, and a synthetic that gives both
    # distributions the same logit scale cannot demonstrate it.
    in_dist = torch.randn(n, 30)
    in_dist[torch.arange(n), torch.randint(0, 30, (n,))] += 2.5

    out_dist = torch.randn(n, 30)
    out_dist[torch.arange(n), torch.randint(0, 30, (n,))] += 2.0
    out_dist = out_dist * 0.85 - 1.2

    id_energy = energy_score(in_dist).numpy()
    ood_energy = energy_score(out_dist).numpy()
    energy_auroc = auroc(id_energy, ood_energy)

    msp_id = torch.softmax(in_dist, -1).amax(-1).numpy()
    msp_ood = torch.softmax(out_dist, -1).amax(-1).numpy()
    msp_auroc = auroc(msp_id, msp_ood)
    check("energy separates ID from OOD", energy_auroc > 0.7, f"(AUROC={energy_auroc:.3f})")
    check("energy beats MSP when logit magnitude carries signal", energy_auroc > msp_auroc,
          f"(energy {energy_auroc:.3f} vs MSP {msp_auroc:.3f})")

    # Second scenario, for the fusion claim. Complementarity can only be shown
    # when *neither* signal is saturated, which is the realistic near-OOD regime:
    # an unknown disease on a known crop does not collapse the feature norm the
    # way a photo of a wall does.
    weak_in = torch.randn(n, 30)
    weak_in[torch.arange(n), torch.randint(0, 30, (n,))] += 2.2
    weak_out = torch.randn(n, 30)
    weak_out[torch.arange(n), torch.randint(0, 30, (n,))] += 1.3

    weak_id_energy = energy_score(weak_in).numpy()
    weak_ood_energy = energy_score(weak_out).numpy()
    weak_energy_auroc = auroc(weak_id_energy, weak_ood_energy)

    rng = np.random.default_rng(0)
    id_cos = rng.normal(0.70, 0.13, n)
    ood_cos = rng.normal(0.55, 0.15, n)
    cosine_auroc = auroc(id_cos, ood_cos)

    scorer = OODScorer().fit(
        weak_id_energy, id_cos, {"near_ood": (weak_ood_energy, ood_cos)}, target_tpr=0.95
    )
    fused_auroc = scorer.calibration.measured_auroc["near_ood"]

    accepted = 1.0 - scorer.is_out_of_distribution(weak_id_energy, id_cos).mean()
    check("keeps ~95% of in-distribution", abs(accepted - 0.95) < 0.03, f"(kept {accepted:.1%})")
    check("fusing two partial signals beats either alone",
          fused_auroc > max(weak_energy_auroc, cosine_auroc),
          f"(fused {fused_auroc:.3f} vs energy {weak_energy_auroc:.3f}, cosine {cosine_auroc:.3f})")
    check("FPR@95TPR reported",
          "near_ood__fpr@95tpr" in scorer.calibration.measured_auroc,
          f"({scorer.calibration.measured_auroc['near_ood__fpr@95tpr']:.3f})")


def test_fusion() -> None:
    print("\ncontext fusion guardrails")
    ids = ["tomato::early_blight", "tomato::late_blight", "tomato::septoria_leaf_spot", "rice::blast"]

    close = np.array([0.40, 0.37, 0.15, 0.08])
    risky = {"tomato::late_blight": 0.95, "tomato::early_blight": 0.20}
    result = fuse(ids, close, risky, FusionConfig(lam=0.35))
    check("re-ranks a genuinely close call", result.top_class == "tomato::late_blight",
          f"({result.top_class} @ {result.top_confidence:.3f})")

    confident = np.array([0.88, 0.07, 0.03, 0.02])
    result = fuse(ids, confident, risky, FusionConfig(lam=0.35))
    check("cannot overturn a confident vision call", result.top_class == "tomato::early_blight")

    uncertain = np.array([0.22, 0.21, 0.20, 0.37])
    result = fuse(ids, uncertain, {"rice::blast": 0.05, "tomato::early_blight": 0.99},
                  FusionConfig(lam=0.9, min_vision_confidence=0.45))
    check("disabled when vision is too uncertain", not result.reordered,
          f"({result.explanation[0][:48]}...)")

    outside = np.array([0.5, 0.3, 0.15, 0.05])
    result = fuse(ids, outside, {"rice::blast": 1.0}, FusionConfig(lam=1.0, top_k=2))
    check("cannot promote a class outside top-k", result.top_class != "rice::blast")

    check("probabilities stay normalised",
          abs(float(result.fused_probabilities.sum()) - 1.0) < 1e-9)


def test_risk_rules() -> None:
    print("\nweather risk rules")
    start = datetime(2026, 7, 1)
    hours = 24 * 14
    stamps = [start + timedelta(hours=h) for h in range(hours)]

    blight_weather = WeatherSeries(
        timestamps=stamps,
        temperature_c=[16.0] * hours,
        relative_humidity=[95.0] * hours,
        precipitation_mm=[0.0] * hours,
        shortwave_radiation=[120.0] * hours,
        wind_speed=[5.0] * hours,
        latitude=12.97, longitude=77.59,
    )
    dry_weather = WeatherSeries(
        timestamps=stamps,
        temperature_c=[34.0] * hours,
        relative_humidity=[35.0] * hours,
        precipitation_mm=[0.0] * hours,
        shortwave_radiation=[600.0] * hours,
        wind_speed=[5.0] * hours,
        latitude=12.97, longitude=77.59,
    )

    model = DiseaseRiskModel()
    observed = start + timedelta(days=12)
    wet = model.assess(["tomato::late_blight"], blight_weather, observed)["tomato::late_blight"].risk
    dry = model.assess(["tomato::late_blight"], dry_weather, observed)["tomato::late_blight"].risk
    check("late blight risk high under Smith conditions", wet > 0.9, f"({wet:.2f})")
    check("late blight risk low when hot and dry", dry < 0.1, f"({dry:.2f})")

    pm_wet = model.assess(["mango::powdery_mildew"], blight_weather, observed)["mango::powdery_mildew"].risk
    pm_dry = model.assess(["mango::powdery_mildew"], dry_weather, observed)["mango::powdery_mildew"].risk
    check("powdery mildew is not simply 'more humidity = more risk'", pm_dry >= pm_wet - 0.05,
          f"(dry {pm_dry:.2f} vs wet {pm_wet:.2f})")

    missing = model.assess(["cotton::leaf_curl_virus"], blight_weather, observed)
    check("unruled class gets the neutral prior",
          abs(missing["cotton::leaf_curl_virus"].risk - 0.35) < 1e-9)
    check("offline returns neutral, never crashes",
          abs(model.assess(["tomato::late_blight"], None, observed)["tomato::late_blight"].risk - 0.35) < 1e-9)


def test_residue() -> None:
    print("\nresidue kinetics")
    record = PesticideRecord(
        active_ingredient="Lambda cyhalothrin", crop="Tomato",
        dt50_days=5.0, dt50_low=3.5, dt50_high=8.0,
        mrl_mg_per_kg=0.3, label_phi_days=5,
        deposit=DepositType.SURFACE, washoff_max=0.55,
        source="FSSAI Version IX (2026); EFSA DAR dissipation range",
    )
    engine = ResidueKineticsEngine({("Lambda cyhalothrin", "Tomato"): record})
    spray = SprayEvent(datetime(2026, 6, 1, 8), "Lambda cyhalothrin", "Tomato", 2.0)

    series = engine.simulate(spray, None, horizon_days=20)
    check("monotone decay with no rain",
          all(series[i].concentration >= series[i + 1].concentration - 1e-12
              for i in range(len(series) - 1)))

    # Half-life sanity: at t = DT50 with T == reference temp, C should be ~C0/2.
    at_dt50 = next(p for p in series if (p.at - spray.applied_at).days >= 5)
    engine_ref = ResidueKineticsEngine({
        ("X", "Tomato"): PesticideRecord("X", "Tomato", 5.0, 5.0, 5.0, 0.3, 5,
                                         reference_temp_c=28.0, source="test")
    })
    ref_series = engine_ref.simulate(SprayEvent(datetime(2026, 6, 1, 8), "X", "Tomato", 1.0), None, 6)
    at_five = next(p for p in ref_series if (p.at - ref_series[0].at).days >= 5)
    check("half-life reproduced at reference temperature",
          abs(at_five.concentration - 0.5) < 0.02, f"(C(DT50)={at_five.concentration:.4f})")

    check("uncertainty band brackets the estimate",
          at_dt50.concentration_low <= at_dt50.concentration <= at_dt50.concentration_high)

    # Rain event must cause a discrete drop, not a smooth one.
    rainy_hours = 24 * 20
    rainy = WeatherSeries(
        timestamps=[datetime(2026, 6, 1, 8) + timedelta(hours=h) for h in range(rainy_hours)],
        temperature_c=[28.0] * rainy_hours,
        relative_humidity=[80.0] * rainy_hours,
        precipitation_mm=[25.0 if 40 <= h <= 42 else 0.0 for h in range(rainy_hours)],
        shortwave_radiation=[200.0] * rainy_hours,
        wind_speed=[4.0] * rainy_hours,
        latitude=12.97, longitude=77.59,
    )
    wet_series = engine.simulate(spray, rainy, horizon_days=18)
    drops = [wet_series[i].concentration - wet_series[i + 1].concentration
             for i in range(len(wet_series) - 1)]
    check("rain causes a discrete wash-off step", max(drops) > 5 * np.median(drops),
          f"(max step {max(drops):.4f} vs median {np.median(drops):.5f})")
    check("rain lowers residue overall", wet_series[-1].concentration < series[-1].concentration)

    # The rule that matters most.
    fast = PesticideRecord("Fast", "Tomato", 0.4, 0.3, 0.5, 0.3, 7, source="test")
    engine2 = ResidueKineticsEngine({("Fast", "Tomato"): fast})
    result = engine2.safe_harvest(SprayEvent(datetime(2026, 6, 1, 8), "Fast", "Tomato", 1.0), None)
    check("label PHI is never shortened by the model",
          result.limiting_factor == "label_phi"
          and result.recommended_date == result.label_phi_date,
          f"({result.recommended_date}, PHI {result.label_phi_date})")

    slow = PesticideRecord("Slow", "Tomato", 22.0, 18.0, 30.0, 0.05, 3, source="test")
    engine3 = ResidueKineticsEngine({("Slow", "Tomato"): slow})
    result = engine3.safe_harvest(SprayEvent(datetime(2026, 6, 1, 8), "Slow", "Tomato", 3.0), None)
    check("model can extend beyond the label PHI",
          result.recommended_date > result.label_phi_date, f"({result.limiting_factor})")
    check("refuses unknown active ingredients",
          _raises(lambda: engine.simulate(SprayEvent(datetime.now(), "Unknown", "Tomato", 1.0))))


def test_metrics() -> None:
    print("\nmetrics")
    rng = np.random.default_rng(3)
    targets = rng.integers(0, 6, 600)
    predictions = targets.copy()
    flip = rng.random(600) < 0.25
    predictions[flip] = rng.integers(0, 6, flip.sum())

    report = classification_report(targets, predictions, [f"c{i}" for i in range(6)])
    check("macro-F1 in range", 0.0 <= report.macro_f1 <= 1.0, f"({report.macro_f1:.3f})")
    check("accuracy roughly matches injected noise", 0.7 < report.accuracy < 0.9,
          f"({report.accuracy:.3f})")

    # Imbalance: macro-F1 must punish ignoring a rare class, accuracy must not.
    imbalanced_t = np.array([0] * 950 + [1] * 50)
    all_majority = np.zeros(1000, dtype=int)
    acc = (imbalanced_t == all_majority).mean()
    f1 = macro_f1(imbalanced_t, all_majority, 2)
    check("macro-F1 exposes what accuracy hides", acc > 0.9 and f1 < 0.55,
          f"(acc={acc:.3f}, macro-F1={f1:.3f})")


def test_quality_and_splits() -> None:
    print("\nquality gate and leakage guard")
    from PIL import Image
    import pandas as pd

    from agrishield.data.splits import assign_group_keys, dhash, group_aware_split, verify_no_leakage
    from agrishield.inference.quality import assess_quality

    black = Image.new("RGB", (400, 400), (4, 4, 4))
    report = assess_quality(black)
    check("rejects a black frame", not report.passed, f"({[i.value for i in report.issues]})")

    tiny = Image.new("RGB", (80, 80), (90, 160, 70))
    check("rejects a tiny image", not assess_quality(tiny).passed)

    leafy = Image.fromarray(
        (np.stack([
            np.full((400, 400), 60), np.full((400, 400), 150), np.full((400, 400), 55)
        ], -1) + np.random.default_rng(0).integers(0, 90, (400, 400, 3))).clip(0, 255).astype(np.uint8)
    )
    check("accepts a plausible leaf photo", assess_quality(leafy).passed,
          f"({[i.value for i in assess_quality(leafy).issues]})")

    img_a = Image.fromarray(np.random.default_rng(7).integers(0, 255, (64, 64, 3)).astype(np.uint8))
    img_b = img_a.resize((60, 60)).resize((64, 64))
    check("dhash treats a resize as a near-duplicate",
          bin(dhash(img_a) ^ dhash(img_b)).count("1") <= 5)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        from PIL import ImageFilter

        rows = []
        for i in range(120):
            # Smooth, structured images - a perceptual hash is designed for
            # photographs, and pure uniform noise is a pathological input for it
            # (dhash compares neighbouring pixels, which are independent in noise).
            rng_i = np.random.default_rng(i)
            low_res = rng_i.normal(0, 1, (8, 8, 3))
            base = Image.fromarray(
                np.clip(np.kron(low_res, np.ones((8, 8, 1))) * 45 + 128, 0, 255).astype(np.uint8)
            ).filter(ImageFilter.GaussianBlur(3))
            for j in range(2):  # two near-identical copies of every specimen
                path = Path(tmp) / f"img_{i}_{j}.png"
                (base if j == 0 else base.resize((60, 60)).resize((64, 64))).save(path)
                rows.append({
                    "path": str(path), "dataset": "synthetic", "raw_label": f"c{i % 4}",
                    "class_id": f"c{i % 4}", "class_index": i % 4, "crop_index": 0,
                    "domain": "field", "mask_path": "",
                })
        df = assign_group_keys(pd.DataFrame(rows), verbose=False)
        df = group_aware_split(df, seed=0, verbose=False)
        verify_no_leakage(df)
        specimen = df.path.str.rsplit("_", n=1).str[0]
        straddling = int((df.groupby(specimen)["split"].nunique() > 1).sum())
        check("near-duplicate pairs stay in the same split", straddling == 0,
              f"({straddling}/120 specimens straddled)")
        check("all three splits are populated", df.split.nunique() == 3,
              f"({df.split.value_counts().to_dict()})")


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def main() -> int:
    print("=" * 66)
    print("AgriShield vision - smoke test")
    print("=" * 66)
    tax = test_taxonomy()
    test_hierarchical_head(tax)
    test_calibration()
    test_ood()
    test_fusion()
    test_risk_rules()
    test_residue()
    test_metrics()
    test_quality_and_splits()

    print("\n" + "=" * 66)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
    print("=" * 66)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

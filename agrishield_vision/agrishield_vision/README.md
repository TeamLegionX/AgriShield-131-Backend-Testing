# AgriShield Vision v2

Field-robust leaf-disease diagnosis with open-set rejection, calibrated
confidence, bounded weather-context fusion, and physically-grounded pesticide
residue kinetics.

Design rationale, trade-off analysis, evaluation protocol and the two-week build
plan live in **[ARCHITECTURE.md](ARCHITECTURE.md)**. Read that first.

---

## Install

```bash
pip install -r requirements.txt
```

`ultralytics`, `onnx`/`onnxruntime` and `litert-torch` are optional (detector and
export tiers). Everything else runs without them.

## Verify the install

```bash
python -m scripts.smoke_test
```

No dataset needed. Checks the parts that are easy to get subtly wrong:
hierarchical normalisation, prototype OOD scoring, calibration monotonicity,
the near-duplicate leakage guard, wash-off physics, and the fusion guardrails.

## Run the pipeline

Stage by stage — each writes an artifact the next reads, so a Colab disconnect
costs you one stage, not the day.

```bash
python -m scripts.run_pipeline manifest   --data-root /content/data
python -m scripts.run_pipeline teacher    --data-root /content/data
python -m scripts.run_pipeline phase1     --data-root /content/data --epochs 15
python -m scripts.run_pipeline phase2     --data-root /content/data --epochs 15
python -m scripts.run_pipeline phase3     --data-root /content/data
python -m scripts.run_pipeline calibrate  --data-root /content/data
python -m scripts.run_pipeline export     --data-root /content/data
```

Expected layout:

```
data/raw/
  plantvillage/<class>/*.jpg          lab; pretraining only, never scored
  plantdoc/<class>/*.jpg
  plantwild/<class>/*.jpg
  plantseg/images/<class>/*.jpg
  plantseg/masks/<class>/*.png        enables background-replacement aug
  rice_leaf_diseases/<class>/*.jpg
  unlabeled_field/*.jpg               your own phone photos, no labels needed
  backgrounds/*.jpg                   soil, sky, canopy, hands, mulch (~200)
  ood/non_leaf/*.jpg                  anything that is not a plant
  ood/unsupported_crop/*.jpg
```

On a free T4, use `--teacher-backbones dinov2_b` and `--batch-size 32`.

---

## Module map

```
configs/taxonomy.yaml          45 canonical classes / 11 crops; alias table;
                               field_support tiers (strong | weak | lab)

agrishield/
  taxonomy.py                  raw label -> canonical id; lab-only quarantine
  data/
    datasets.py                manifest builder, unified + unlabeled datasets
    splits.py                  dhash + LSH banding, group-aware leakage-free split
    transforms.py              background replacement, phone-camera realism
  models/
    heads.py                   prototype (K=4 cosine) + hybrid + hierarchical
    student.py                 MobileNetV4-small + export wrapper
    teacher.py                 frozen DINOv2 bank + feature caching
  train/
    losses.py                  hierarchical NLL, KD (logit + feature), same-crop CutMix
    distill.py                 3-phase loop, EMA, adaptive BatchNorm
  inference/
    quality.py                 rule-based gate with farmer-facing guidance keys
    calibration.py             temperature scaling, ECE, coverage-risk
    ood.py                     energy + prototype cosine + Mahalanobis
    gradcam.py                 Grad-CAM++ with lesion peak boxes
    pipeline.py                G0-G6 orchestration + decision policy
  fusion/
    weather.py                 Open-Meteo client, degrades to None offline
    risk_priors.py             citable epidemiological rules (Smith periods etc.)
    late_fusion.py             bounded top-k Bayesian re-ranking + ablation
  kinetics/
    residue.py                 Q10 + photolysis + rain-event wash-off; PHI floor
  export/
    export_edge.py             ONNX + int8 TFLite via LiteRT Torch
  eval/
    metrics.py                 macro-F1, look-alike confusion blocks, bundle
```

---

## Five non-obvious decisions

**Lab classes stay in the model but are discounted at inference.** Removing them
would push field photos into a wrong *neighbouring* class. Keeping them
undiscounted would surface diagnoses you cannot support. The 20× discount is the
middle path, and `field_support` makes the tiering explicit and auditable.

**The hierarchical head uses a dense masked logsumexp, not `scatter_reduce`.**
G and N are both small, so the `[B, G, N]` tensor is trivial — and every op in
that path survives `torch.export` → TFLite, which scatter-based versions do not
reliably do. Verified: class posteriors marginalise to the crop posterior to
4.5e-08.

**Near-duplicate detection uses LSH banding, not a hash-prefix bucket.**
Bucketing on the top 32 bits of the dhash silently misses most true duplicates —
one flipped bit sends a pair to different buckets and they are never compared.
You get a clean-looking split and inflated test metrics. This bug was live in an
earlier draft of `splits.py` and was only caught by a test that specifically
looked for it.

**Rain wash-off is an event, not a rate term.** A storm strips a fraction of the
surface deposit in an hour. Folding that into an exponential rate constant is
physically wrong and systematically mis-predicts residue in both directions.

**Thresholds live in a JSON sidecar, not in the flatbuffer.** During a hackathon
week, retuning the confident/tentative bar with a config push instead of an
overnight retrain is the difference between fixing a demo and missing one.

---

## What this cannot do

- Reach 99% on field photos. Realistic target is **macro-F1 0.60–0.75**.
- Reliably detect a disease it was never trained on. Near-OOD AUROC is
  **0.70–0.80**, and it is reported separately rather than averaged away.
- Learn image×weather fusion. No public dataset carries paired metadata; the
  weather branch is an explicit rule-based prior with a documented path to a
  trained replacement.
- Measure residue. It estimates from literature half-lives. Only laboratory
  analysis is definitive, and the label PHI always binds.

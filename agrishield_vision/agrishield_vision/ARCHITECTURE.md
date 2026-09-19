# AgriShield Vision v2 — Architecture

**Status:** design + reference implementation
**Replaces:** MobileNetV2 fine-tuned on PlantVillage (38 classes) + Grad-CAM + standalone kinetics microservice

---

## 1. Your current pipeline, as I understand it

```
phone photo → resize 256 → tensor → normalise → MobileNetV2 (depthwise-separable)
            → FC head → softmax(38) → label + confidence
            → Grad-CAM on final conv → heatmap overlay

[separate FastAPI service]  spray log → Open-Meteo → C(t) = C₀·e^(−kt) → safe-harvest countdown
```

Two independent services, one trained on lab images, one not trained at all.

### The five weaknesses that matter

**W1 — The training distribution is not the deployment distribution.** PlantVillage is single leaves on uniform backgrounds. Models trained on it reach ~99.3% on their own split and drop to roughly **31%** under different capture conditions; a separate in-the-wild evaluation put accuracy **below 40%**. Grad-CAM studies on cross-domain transfer show attention shifting from lesion regions to background. Your model has most likely learned "uniform grey background + leaf silhouette", and the field will not supply that.

**W2 — Softmax over a closed 38-class set is a category error.** Every input gets assigned to one of 38 labels. A hand, a screenshot, a cow, an unsupported crop, a disease you never trained on — all become a confident tomato diagnosis. There is no mechanism that can output "I don't know", and that is the output a farmer most needs.

**W3 — Confidence is meaningless.** Raw softmax is systematically over-confident under domain shift. A 2026 benchmark measured a model lose **67.7 accuracy points** on controlled-to-field transfer while mean predicted confidence stayed at **79.76%**. A farmer cannot distinguish a real 92% from a hallucinated one, and a wrong diagnosis here costs a wasted spray, a possible MRL violation, and trust.

**W4 — The kinetics engine and the vision model don't talk, and the physics is incomplete.** Constant-k exponential decay ignores that DT50 is measured under specific trial conditions. More seriously, treating rainfall as part of a smooth rate constant is physically wrong: wash-off is an *event* that strips a fraction of the surface deposit in an hour. A week-long exponential smear systematically under-predicts residue after a dry spell and over-predicts it after a storm.

**W5 — Grad-CAM is being asked to do a job it cannot do.** Your Round-1 feedback asked you to show *where* the disease is. A CAM explains the model, not the pathology, and on cross-domain images it drifts onto backgrounds. It is a useful trust signal; it is not a lesion map, and claiming otherwise in front of an agronomist judge is a risk.

---

## 2. Final architecture

### The one-line bet

> You cannot put a ViT-L on a ₹8,000 Android phone. You *can* distil its domain robustness into a 4M-parameter CNN — including from unlabeled field photos you shoot yourself.

### Diagram

```
                                   ┌──────────────────────────────────────────────┐
                                   │              EDGE TIER (offline)             │
                                   │            target: ~11 MB, <400 ms           │
  ┌─────────┐                      │                                              │
  │  photo  │──────────────────────┼─▶ G0  QUALITY GATE        (rules, ~5 ms)     │
  └─────────┘                      │      blur · exposure · contrast · subject    │
                                   │      size · vegetation presence              │
                                   │         │ fail → REJECT_QUALITY              │
                                   │         │        + "move closer" / "hold      │
                                   │         ▼          steady"  (i18n key)       │
                                   │                                              │
                                   │      G1  LEAF DETECTOR    (YOLOv8n, ~3 MB)   │
                                   │      single class "leaf", int8, 320px        │
                                   │         │ no box → REJECT_NOT_LEAF           │
                                   │         ▼ crop + 8% pad                      │
                                   │                                              │
                                   │      G2  CLASSIFIER       (~8 MB int8)       │
                                   │      MobileNetV4-conv-small @ 224            │
                                   │            │                                 │
                                   │            ├─▶ linear head ──┐               │
                                   │            ├─▶ prototype head (K=4, cosine)  │
                                   │            │        │        ├─▶ 0.5/0.5 →   │
                                   │            │        │        │   class logits│
                                   │            │        │        │               │
                                   │            └─▶ crop head ────┴─▶ hierarchical│
                                   │                     log p(c) = log p(crop)   │
                                   │                              + log p(c|crop) │
                                   │            │                                 │
                                   │            ├─▶ max prototype cosine ──┐      │
                                   │            └─▶ energy = logsumexp ────┤      │
                                   │                                       ▼      │
                                   │      G3  TEMPERATURE SCALING    G4  OPEN-SET │
                                   │          p = softmax(z / T)         z-fused, │
                                   │          T fitted on field val      95% TPR  │
                                   │            │                          │      │
                                   │            │              below thr → REJECT │
                                   │            ▼                          _UNKNOWN│
                                   │      G5  CONTEXT RE-RANK  ◀──── weather prior │
                                   │          bounded, top-k only, off when       │
                                   │          vision is already uncertain         │
                                   │            │                                 │
                                   │            ▼                                 │
                                   │      G6  DECISION POLICY                     │
                                   │          CONFIDENT / TENTATIVE /             │
                                   │          CROP_ONLY / 3× REJECT               │
                                   └────────────┬─────────────────────────────────┘
                                                │
              ┌─────────────────────────────────┼─────────────────────────────────┐
              ▼                                 ▼                                 ▼
   ┌────────────────────┐        ┌──────────────────────────┐      ┌──────────────────────┐
   │ Grad-CAM++ boxes   │        │  treatment suggestion    │      │  SERVER TIER (online)│
   │ "6 spots found"    │        │  (CIB&RC registered      │      │  frozen DINOv2 ViT-L │
   │ severity proxy     │        │   products only)         │      │  + hybrid head + TTA │
   └────────────────────┘        └────────────┬─────────────┘      │  + PlantSeg lesion   │
                                              │                    │    segmentation      │
                                              ▼                    │  second opinion when │
                                 ┌──────────────────────────┐      │  edge says TENTATIVE │
                                 │  SPRAY LOG               │      └──────────────────────┘
                                 │  crop, AI, date, dose    │
                                 └────────────┬─────────────┘
                                              ▼
                                 ┌──────────────────────────────────────────┐
                                 │  RESIDUE KINETICS                        │
                                 │  dC/dt = −k_eff·C,  k = ln2/DT50         │
                                 │  k_eff = k·Q10^((T−T_ref)/10) [·photo]   │
                                 │  rain EVENT: C ← C·(1−W),                │
                                 │              W = W_max(1−e^(−R/R₅₀))     │
                                 │  growth dilution: C ← C·(M₀/M_t)         │
                                 │            │                             │
                                 │            ▼                             │
                                 │  harvest = MAX(model_date, label_PHI)    │
                                 │  ← the model may only ever say "wait     │
                                 │    LONGER". Never shorter. Enforced in   │
                                 │    code, not in a footnote.              │
                                 └──────────────────────────────────────────┘
```

### Why this and not the alternatives

| Option | Verdict | Reasoning |
|---|---|---|
| **(a) YOLOv8n detect → classify** | **Adopted, as stage G1** | Decoupling localisation from classification is the single most effective fix for W1 — it removes the background the model was keying on. A published two-stage pipeline reports ~92.9 mAP@0.5 detection with ~78.5% downstream classification on PlantDoc. Trained as **one class ("leaf")**, not per-disease: PlantDoc's 8,851 boxes plus PlantSeg masks give plenty of leaf boxes, and a single-class detector transfers to crops the classifier has never seen — which is exactly what the "unsupported crop" reject path needs. |
| **(b) Mobile ViT on device** (MobileViT, EfficientFormer) | **Rejected** | Competitive on paper, worse in your deployment. Attention blocks quantise poorly to int8, export less cleanly through `torch.export`, and are slower on CPU at equal accuracy. Your target phones run TFLite on CPU via XNNPACK; GPU/NNAPI delegates are inconsistent on low-end SoCs. Design against the CPU number. |
| **(b') Heavy backbone as teacher** (DINOv2/ConvNeXt) | **Adopted, frozen** | A frozen DINOv2 ViT-L/14 with a light head reaches **77.56%** on PlantWild, above the text-augmented CLIP-prototype baseline (76.18%). Three-backbone fusion (DINOv2+DINOv3+CLIP) reaches **80.23% ± 0.41**. The whole sweep runs in ~a day on one consumer GPU because features cache once. |
| **(c) Knowledge distillation** | **Adopted — the core of the design** | This is how ViT robustness reaches the phone. Two channels: logit KD (temperature 3, ×T² gradient rescale) transfers the differential ranking a farmer needs for "also consider"; cosine feature KD transfers representation geometry, which is where domain robustness lives — and it works on **unlabeled** field photos. |
| **(d) Ensemble + TTA** | **Server tier only** | Real gains, unaffordable latency and size on device. Runs as the "second opinion" path when the edge returns TENTATIVE. |
| **(e) Hierarchical crop → disease** | **Adopted** | Three concrete wins: crop identity is much easier than disease and regularises the trunk; "rice blast on a mango leaf" becomes structurally improbable, not merely unlikely; and the app can degrade to *"this is a rice leaf, disease unclear"* — a genuinely useful answer that a flat 38-way softmax cannot produce. Verified in the smoke test: class posteriors marginalise to the crop posterior to 4.5e-08. |

### Expected trade-offs

| | Edge tier | Server tier |
|---|---|---|
| Backbone | MobileNetV4-conv-small (3.8M) | frozen DINOv2 ViT-L/14 + 1.25M head |
| Input | 224×224 | 224×224 + 3-crop × hflip TTA |
| Size | ~8 MB int8 (+ ~3 MB detector) | ~1.2 GB (server-side) |
| Latency | 150–400 ms, low-end Snapdragon, 1 thread | 200–600 ms + network |
| Expected field macro-F1 | **0.60–0.75** (well-supported classes) | +5–10 points over edge |
| Offline | yes | no |

**Fallback backbone:** `efficientnet_lite0` if TFLite conversion fights you. `mobilenetv3_large_100` for the widest device support.

---

## 3. Data strategy

### Datasets to combine

| Dataset | Size | Domain | Role | Licence note |
|---|---|---|---|---|
| **PlantVillage** | 54,306 / 38 cls | lab | Phase-1 pretraining **only**. Quarantined from every reported metric. | Commonly CC0 — verify your mirror |
| **PlantDoc** | 2,598 / 13 species, 27–30 cls | field | Train + field test. Has 8,851 bounding boxes → detector training. | **CC BY 4.0**, commercial use permitted with attribution |
| **PlantWild** | 18,542 / 89 cls | field | The most valuable single addition. Official split 14,865 / 3,677. | Verify terms — scraped from search engines; research use is safe, commercial is not obvious |
| **PlantSeg** | 115 classes, 69 diseases, 34 hosts | field | **Segmentation masks** → background-replacement augmentation *and* the real lesion-localisation head | Verify (Zenodo) |
| **Cassava Leaf Disease** | 21,367 / 5 cls | field-ish | Extra field data | ⚠️ Kaggle **competition** rules — likely non-commercial. Flag this. |
| **Plant Pathology 2021** | ~23,000, apple | field | Apple classes, expert-annotated | Kaggle competition terms |
| Rice / cotton / sugarcane / chilli / mango sets | varies | field | India-relevant crops | Per-dataset; most are CC BY or CC0 on Kaggle |
| **Your own phone photos** | target 2,000+ | field | **Unlabeled** distillation + representative set for int8 quantisation | Yours |

> **Licensing action item:** competition-sourced data (Cassava, Plant Pathology) may restrict commercial use of derived models. Put a one-line licence audit in the deck. Judges notice, and it costs you nothing to be the team that checked.

### Unifying the taxonomy

Explicit alias table, never string similarity. `configs/taxonomy.yaml` maps every source's raw folder name onto `crop::condition` IDs. An unmapped label is reported loudly, not silently bucketed. 45 classes across 11 crops in the reference config.

Each class carries `field_support`:

- **strong** (≥300 field images) — shown normally
- **weak** (50–300) — shown with a low-certainty banner
- **lab** (PlantVillage only) — kept in the output space so field photos aren't force-fitted into a neighbouring class, but **discounted 20× at inference** and never surfaced as a confident diagnosis

That last mechanism is how you keep 38→45 classes without lying about what you can detect. It also directly answers judge ask #2 ("list all the diseases you detect") with a defensible, tiered list.

### Splitting — the part most projects get wrong

1. **Group by perceptual-hash cluster, not by image.** PlantVillage has many near-identical shots of the same physical leaf. A random split puts siblings on both sides and you measure specimen memorisation.
2. **LSH banding for candidate generation.** The 64-bit dhash is cut into 8 bands; pairs are compared only if a band matches exactly. Pigeonhole guarantees any pair at Hamming ≤5 shares a band. *(The obvious alternative — bucketing on the top 32 bits — silently misses most true duplicates and produces a clean-looking split with inflated metrics. I hit exactly this bug while building; it is invisible unless you test for it.)*
3. **The test split is field-only.** Enforced in code. Lab images may train, may sanity-validate, and never score the headline.
4. **Greedy class-balanced group assignment**, rarest class first, so tail classes survive into val/test. Plain `GroupShuffleSplit` does not guarantee this and silently destroys macro-F1 reporting.

### Class imbalance

Square-root inverse-frequency sampling (`power=0.5`), not full inverse frequency — the latter over-samples 30-image classes hard enough that the model memorises them and macro-F1 gets *worse*. Focal loss is available but **off by default**: in merged scraped data, a large fraction of "hard examples" are mislabelled, and focal will faithfully devote capacity to fitting that noise.

---

## 4. Training recipe

### Augmentation — ranked by value

1. **Background replacement (highest value by a wide margin).** Composite the segmented leaf onto real field backgrounds. Masks from PlantSeg where available; ExG + Otsu fallback on PlantVillage, which works well *precisely because* its background is uniform. This attacks W1 at the root. p=0.5 in phase 1, 0.6 in phase 2.
2. **Phone-camera realism.** Motion blur, exposure error (0.55–1.5×), hard-edged directional shadow, JPEG recompression at quality 30–85.
3. **Aggressive RandomResizedCrop, scale (0.25, 1.0).** Field photos are either a close crop of one lesion or a wide shot where the leaf is 20% of frame.
4. RandAugment(n=2, m=9), ColorJitter, rotation ±25°, RandomErasing p=0.25.
5. **CutMix within crop only** (α=0.2, p=0.3). Blending a mango leaf with a rice leaf produces an image whose label is a lie. Same-crop CutMix is at least physically plausible and proxies multi-infection. No MixUp.

### Losses

- Hierarchical NLL with **label smoothing 0.1** + auxiliary crop CE (weight 0.3). Smoothing does double duty: small accuracy gain, and direct pressure against the over-confidence in W3.
- Prototype separation loss (weight 0.05) — pushes prototypes of different classes apart. Without it, look-alike diseases' prototypes collapse and the cosine OOD score loses its discriminative power.
- Distillation: `KL(teacher‖student)·T²` (T=3, weight 1.0) + `1 − cos(proj(student), teacher_CLS)` (weight 0.5).

### Schedule

AdamW, cosine decay with 2-epoch warmup, discriminative LRs (backbone 3e-4 / head 1e-3), weight decay 0.05, grad clip 1.0, AMP, **EMA of weights (decay 0.999)**. EMA is worth the four lines: on noisy merged data the EMA weights are consistently better *calibrated*, and calibration is a first-class requirement here.

### Three phases

| Phase | Data | Epochs | What it buys |
|---|---|---|---|
| **1 — broad supervised** | everything, lab included | ~15 | a usable feature space, cheaply |
| **2 — distillation** | field-heavy + unlabeled field photos, teacher on every batch | ~15 | **the biggest single jump in field macro-F1** |
| **3 — field-only fine-tune** | field only, LR 5e-5, + adaptive BatchNorm | ~5 | 1–3 points, reliably |

On adaptive BN: re-estimating BatchNorm running statistics on target-domain data needs no gradients and no labels, and is one of the few simple domain-shift mitigations that measurably helps. DANN-style adversarial adaptation has been shown to *degrade* accuracy on this exact controlled-to-field transfer — it is deliberately not offered.

### Post-hoc, on the field **validation** split (never test)

Temperature scaling → decision thresholds from the coverage-risk curve → OOD calibration at 95% TPR.

### Evaluation protocol — report all seven

1. Field-only **macro-F1** + per-class F1 (headline; *not* accuracy, *not* anything on PlantVillage)
2. Confusion matrix restricted to declared look-alike groups (early vs late blight vs Septoria — 3×3 blocks are readable on a slide, a 45×45 matrix is not)
3. ECE + reliability diagram, before and after temperature scaling
4. Coverage-risk curve with the operating point marked
5. OOD AUROC / FPR@95TPR **broken out per rejection case**
6. On-device p50/p95 latency and model size, measured on a real phone
7. With/without context-fusion ablation across λ

### Realistic targets — use these, not 99%

| Metric | Target | Note |
|---|---|---|
| Field macro-F1 (strong+weak classes) | **0.60 – 0.75** | PlantWild SOTA is ~80% with a ViT-L; a 4M-param student will land below that |
| Field accuracy at 60% coverage | 0.85+ | the number the product actually runs on |
| ECE after temperature scaling | **< 0.05** | verified mechanism: 0.260 → 0.024 in the smoke test |
| OOD AUROC — not-a-leaf | 0.90 – 0.97 | easy case, detector does most of it |
| OOD AUROC — unsupported crop | 0.80 – 0.90 | |
| OOD AUROC — **unknown disease, known crop** | **0.70 – 0.80** | near-OOD is genuinely hard. Claim this range. |
| Classifier size (int8) | ≤ 8 MB | + ~3 MB detector |
| Latency, low-end Snapdragon, 1 thread | 150 – 400 ms | measure with `adb` + the TFLite benchmark tool |

---

## 5. Multimodal integration

### Why there is no trainable joint model (state this out loud)

PlantVillage, PlantDoc, PlantWild and the rest carry **no paired weather, GPS or date metadata**. There is no `(image, weather, label)` triple to learn from. Anything you fit that claims to use both is learning from the image while the weather branch memorises noise — and it will validate cleanly, because the noise is consistent within a dataset.

**Trainable now:** vision classifier, hierarchical heads, prototypes, calibration, OOD thresholds, the leaf detector, the lesion segmentation head (PlantSeg masks).
**Needs new data:** any learned image×context fusion; any severity regressor; any yield-loss estimate.

### What runs instead

```
log p_fused(c) = log p_vision(c) + λ·clip(log(risk_c / 0.35), ±1.0)     for c ∈ top-k
```

Three guardrails, all verified in the smoke test:

- **G1 — top-k only.** Weather cannot introduce a disease the image gave no support to. ✓ *verified: cannot promote a class outside top-k*
- **G2 — bounded shift.** Reorders a 0.40/0.37 call; cannot overturn 0.88/0.07. ✓ *verified both directions*
- **G3 — off when vision is uncertain.** Fusing a prior into noise yields confident nonsense. ✓ *verified*

Risk rules are explicit, citable, and agronomist-correctable — which a learned MLP is not. Smith periods for late blight (2 consecutive days, min temp ≥10 °C, ≥11 h at RH ≥90%), IRRI rules for rice blast and BLB, dew+temperature windows for rusts, and — importantly — powdery mildew where **rain *lowers* risk**, since free water inhibits germination. That asymmetry is a good thing to have a judge notice.

Default **λ = 0.35**. Ship the ablation. If the curve is flat, say so and turn fusion off: a defensible negative result beats an unverifiable feature.

**The path to real fusion:** log `(image, gps, timestamp, weather snapshot, prediction, farmer/agronomist confirmation)` on every diagnosis. At ~2,000–5,000 confirmed records with geographic and seasonal spread, fit a real late-fusion model and delete `risk_priors.py`. Present the data-collection loop as a roadmap feature — it is one.

### Closed loop

```
diagnose → treatment (CIB&RC-registered products only, MRL-checked)
        → spray log (crop, active ingredient, date, dose)
        → residue kinetics → safe-harvest window
        → re-diagnose in 7 days → efficacy feedback → training data
```

### Kinetics, made defensible

First-order dissipation `C(t) = C₀·exp(−∫k_eff dt)`, `k = ln2/DT50`, with:

1. **Temperature** — Q10 ≈ 2.0: `k_T = k_ref · Q10^((T−T_ref)/10)`
2. **Photolysis** — proportional to insolation, only for photolabile actives
3. **Rain wash-off — an event, not a rate.** `C ← C·(1−W)`, `W = W_max·(1−e^(−R/R₅₀))`, with `W_max` boosted 1.6× inside the rainfastness window and cut to ~0.12 for systemic actives already inside leaf tissue. *Verified: a 25 mm storm produces a step ~1,700× the median hourly decrement.*
4. **Growth dilution** — `C ← C·(M₀/M_t)` for rapidly expanding tissue

**Outputs are intervals**, from the literature DT50 range (e.g. 3.5–8.0 d), and the *pessimistic* bound decides clearance. Published DT50s span a factor of two; a single number is false precision.

**The rule that overrides everything:**

```python
recommended_harvest = max(model_date_at_MRL, label_PHI_date)
```

The CIB&RC label PHI is the legal instrument. The model may only ever recommend waiting **longer**. This is enforced in `residue.py`, not in a disclaimer — and it is a strong thing to say to a panel. *Verified: fast-degrading actives are correctly PHI-limited; slow ones correctly extend past it.*

Every `PesticideRecord` **requires** a `source` citation — the constructor raises without one. A residue prediction with no citable half-life is a guess with a decimal point.

---

## 6. Two-week build plan (free-tier Colab/Kaggle)

### Week 1 — get an honest number

| Day | Task | Artifact |
|---|---|---|
| 1 | Download PlantDoc, PlantWild, PlantSeg, 2 India-crop sets. Write aliases into `taxonomy.yaml`. | `manifest.csv` |
| 1 | **Shoot 200 background plates + 300 unlabeled field photos on a phone.** Half a day, highest ROI of the fortnight. | `raw/backgrounds`, `raw/unlabeled_field` |
| 2 | `assign_group_keys` + `group_aware_split`. Run `verify_no_leakage`. Print the per-class field test counts and *look at them*. | split manifest |
| 2–3 | Cache DINOv2 features once (~15 min for 60k images at fp16 on a T4). Train the teacher head. | `teacher_head.pt` |
| 3–4 | Phase 1. ~4–6 h on a T4. Checkpoint every epoch — Colab disconnects. | `phase1/student_best.pt` |
| 5 | Phase 2 distillation. ~6 h. **Expect the big jump here.** | `phase2/student_best.pt` |
| 6 | Phase 3 + adaptive BN. ~1 h. | `phase3/student_best.pt` |
| 7 | Calibration, thresholds, OOD fitting, full evaluation. | `calibration.json`, all seven metrics |

### Week 2 — make it real

| Day | Task |
|---|---|
| 8 | YOLOv8n leaf detector: `yolo detect train model=yolov8n.pt data=leaf.yaml imgsz=640 epochs=60` |
| 9 | Export ONNX + int8 TFLite. **Re-fit T on the quantised model** — int8 shifts the logit scale. Measure on a real phone with `adb`. |
| 10 | Wire `/diagnose` v2 into the existing FastAPI app. Wire `residue.py` into `/mrl-check` — this closes the gap where `assess_crop_safety()` takes residue as an input. |
| 11 | Flutter integration, offline path, i18n keys for every decision and quality message |
| 12 | Generate the figures: reliability diagram, coverage-risk curve, look-alike confusion blocks, λ ablation, latency table |
| 13 | Dry-run the demo **on airplane mode**, on the actual phone, in bad light |
| 14 | Buffer. Something will break. |

### What to demo live

1. **Airplane mode on.** Diagnose a real leaf offline. Show lesion boxes and a spot count.
2. **Photograph your own hand.** It says "no leaf detected", not "tomato early blight, 91%".
3. **Photograph a blurry shot.** It says "hold steady and tap to focus" — in Kannada.
4. **Photograph an unsupported crop.** "This looks like a plant we don't cover yet."
5. **Log a spray**, show the safe-harvest countdown, then point out that the label PHI is binding and the model only ever extends it.
6. **Show the reliability diagram** and say: *"most crop-disease demos show you 99%. Here is why that number is a lab number, and here is ours."*

Demo #2 through #4 are your differentiator. Everyone will show a correct diagnosis. Almost nobody will show their model **refusing**.

### Graphs that will land

- **Reliability diagram**, before/after temperature scaling — the most persuasive single slide available to you
- **Coverage-risk curve** with the operating point marked: "at 60% coverage we are wrong 12% of the time"
- **Lab vs field macro-F1 bar pair**, with the domain gap labelled — you are showing the gap everyone else hides
- **Look-alike 3×3 confusion blocks** (early/late blight/Septoria) — this is the case that decides which chemical gets sprayed
- **λ ablation** — proves the weather feature is measured, not asserted
- **Latency/size table** from a real budget phone, not Colab

### Limitations to state upfront (as strength, not apology)

1. "Field macro-F1 is 0.6–0.7, not 99%. Anyone quoting 99% is reporting a lab number. Here is the published evidence that lab models fall to ~31% in the field."
2. "Detecting an *unknown* disease on a *known* crop is the hardest case. Our AUROC there is ~0.75. We report it separately rather than averaging it away."
3. "Weather fusion is a rule-based epidemiological prior, not a learned model, because no public dataset has paired weather labels. Here is our data-collection path to a real one."
4. "Residue is an estimate from literature half-lives, not a measurement. Only lab analysis is definitive, and the legal label PHI always binds."
5. "Classes marked *lab-only* are in the model but never surfaced as confident answers. Here is the tiered list of what we actually detect."

---

## 7. Judge feedback → where it is addressed

| Round-1 feedback | Where |
|---|---|
| Show the spot, not just a label | Grad-CAM++ peak boxes + spot count (G6); PlantSeg-trained segmentation head on the server tier for a defensible lesion mask |
| List all detectable diseases | `taxonomy.yaml` with tiered `field_support` — a defensible list, not a marketing count |
| Add quantitative data | The seven-metric evaluation bundle; lesion counts as a severity proxy; residue curves with uncertainty bands |
| Reduce hardware dependence | Entire edge tier is offline on a commodity phone. The electrochemical sensor stays explicitly on the roadmap. |

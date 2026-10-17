<div align="center">

<img src="assets/banner.svg" width="100%" alt="ANN Customer Churn Predictor banner"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3500&pause=1000&color=00F0FF&center=true&vCenter=true&width=860&height=48&lines=Predicting+customer+churn+with+PyTorch;10+%E2%86%92+32+%E2%86%92+16+%E2%86%92+8+%E2%86%92+1+%C2%B7+ReLU+%C2%B7+BCEWithLogits;Class-weighted+loss+%C2%B7+pos_weight+%E2%89%88+12.8;Recall+87.3%25+at+threshold+0.30" alt="Typing SVG"/></a>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Stage](https://img.shields.io/badge/stage-complete-b4ff3d?style=flat-square)
![Recall](https://img.shields.io/badge/recall-87.31%25-00f0ff?style=flat-square)
![Threshold](https://img.shields.io/badge/threshold-0.30-ff3df2?style=flat-square)
![Imbalance](https://img.shields.io/badge/churners-7.25%25_of_test-faff3d?style=flat-square)
![Dataset](https://img.shields.io/badge/dataset-50k_customers-faff3d?style=flat-square)

</div>

<img src="assets/divider.svg" width="100%" alt=""/>

## 📑 Table of Contents

- [🎯 Project Goal](#-project-goal)
- [📦 Dataset](#-dataset)
- [🗺️ The Journey](#️-the-journey)
- [🧪 Pipeline](#-pipeline)
- [🧱 Architecture](#-architecture)
- [🏋️ Training](#️-training)
- [🔍 Findings](#-findings)
- [🔮 Predicting a New Customer](#-predicting-a-new-customer)
- [📦 Model Package](#-model-package)
- [🧠 Key Decisions](#-key-decisions)
- [✅ Progress & Roadmap](#-progress--roadmap)
- [🚀 Getting Started](#-getting-started)

<img src="assets/divider.svg" width="100%" alt=""/>

## 🎯 Project Goal

Build an **Artificial Neural Network in PyTorch** that answers one question:

> **Given a customer's behaviour and account details, will they churn (`1`) or stay (`0`)?**

The ANN outputs a churn **probability**, and a chosen **threshold** turns that probability into the final class.

> [!NOTE]
> The goal was to learn the **complete deep-learning workflow** end to end (data, tensors, architecture, loss, backprop, imbalance, thresholds, saving, inference) rather than to run extensive EDA.

### 🏁 Final result at a glance

| Recall | Precision | F1 | Accuracy | Threshold |
|:---:|:---:|:---:|:---:|:---:|
| **87.31%** | 15.50% | 26.33% | 64.57% | **0.30** |

Low precision is a deliberate trade-off: the project prioritises **catching churners**. See [Findings](#-findings).

<img src="assets/divider.svg" width="100%" alt=""/>

## 📦 Dataset

A **synthetically generated** customer churn dataset with **50,000 customers**, intentionally built with **nonlinear relationships and feature interactions** so an ANN has real patterns to learn.

| Feature | Meaning |
|---|---|
| `age` | Customer age |
| `tenure_months` | How long the customer has been with the company |
| `monthly_charges` | Monthly amount charged |
| `support_tickets` | Number of support tickets |
| `late_payments` | Number of late payments |
| `usage_hours` | Customer usage |
| `satisfaction` | Customer satisfaction score |
| `contract_monthly` | Whether the customer has a monthly contract |
| `tech_support` | Whether the customer has technical support |
| `auto_payment` | Whether automatic payment is enabled |

| Item | Detail |
|---|---|
| 🧮 Rows | 50,000 customers |
| 🔢 Input features | 10 |
| 🎯 Target | `churn` → `0` = no churn, `1` = churn |
| 🧬 Source | Synthetic (generated) |
| ✂️ Split | 80 / 20 → **40,000 train**, **10,000 test** (`random_state=42`, `stratify=y`) |

### ⚖️ The class imbalance

<div align="center">

| Test set | Customers | Share |
|---|---:|---:|
| 🟢 Non-churners | 9,275 | 92.75% |
| 🔴 Churners | **725** | **7.25%** |

</div>

The training set has roughly **12.8 non-churners for every churner**. This single fact shaped the second half of the project.

> [!WARNING]
> A model that predicts **"nobody churns"** scores **92.75% accuracy** and catches **0 churners**. Accuracy alone is misleading here.

<img src="assets/divider.svg" width="100%" alt=""/>

## 🗺️ The Journey

```mermaid
flowchart LR
    V1["🧪 v1 baseline<br/>10→16→8→1<br/>Sigmoid + BCELoss"] --> P["⚠️ Problem<br/>92.75% of test set is 'no churn'<br/>accuracy can lie"]
    P --> F1["⚖️ Weighted loss<br/>BCEWithLogitsLoss<br/>pos_weight ≈ 12.80"]
    F1 --> F2["🧱 Bigger network<br/>10→32→16→8→1<br/>no final Sigmoid"]
    F2 --> F3["🎚️ Threshold tuning<br/>0.3 · 0.4 · 0.5 · 0.6 · 0.7"]
    F3 --> V2["🏆 v2 final<br/>threshold 0.30<br/>recall 87.31%"]
    V2 --> S["💾 Saved package<br/>model · scaler · config"]

    classDef old fill:#1a1a2e,stroke:#6b6b8a,color:#9a9ab8
    classDef warn fill:#4a0d4a,stroke:#ff3df2,stroke-width:3px,color:#ffffff
    classDef done fill:#0d3b2e,stroke:#b4ff3d,stroke-width:2px,color:#ffffff
    class V1 old
    class P warn
    class F1,F2,F3,V2,S done
```

<details open>
<summary><b>📖 The story in five steps (click to collapse)</b></summary>

1. **v1: get a working network.** Built `10 → 16 → 8 → 1` with `ReLU`, `Sigmoid` and `BCELoss`, trained 100 epochs, and got a clean loss curve (≈ 0.69 → ≈ 0.39).
2. **Spot the trap.** With only 7.25% churners, a low loss and high accuracy do not prove the model finds churners.
3. **Fix the loss.** Switched to `BCEWithLogitsLoss(pos_weight≈12.80)`, which makes a missed churner about 12.8× as costly. The final `Sigmoid` was removed because the loss now includes it, which is also more numerically stable.
4. **Grow the model.** Moved to `10 → 32 → 16 → 8 → 1`. Training lengths were also compared, and **more epochs did not mean a better model**: the weighted model did better at an earlier stage than after running to 10,000 epochs.
5. **Choose the decision rule.** Swept thresholds 0.3 to 0.7 and picked **0.30**, because missing a churner costs more than a false alarm in this project.

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

## 🧪 Pipeline

```mermaid
flowchart LR
    subgraph DATA["📦 Data prep"]
        A["📄 CSV dataset<br/>50,000 rows"] --> B["🐼 Pandas<br/>load"]
        B --> C["✂️ Separate X and y"]
        C --> D["🔀 80/20 split<br/>stratified"]
        D --> E["📏 StandardScaler<br/>fit on train only"]
        E --> F["🔥 float32 tensors"]
    end
    subgraph MODEL["🧠 Model"]
        G["🕸️ ANN<br/>10 → 32 → 16 → 8 → 1"] --> H["⚖️ BCEWithLogitsLoss<br/>pos_weight ≈ 12.80<br/>+ Adam lr=0.001"]
        H --> I["🧪 Test logits"]
    end
    subgraph EVAL["📊 Evaluation"]
        J["🎯 sigmoid → probability"] --> K["🎚️ Threshold 0.30"]
        K --> L["📊 Confusion matrix<br/>Recall · Precision · F1"]
    end
    subgraph SHIP["💾 Package"]
        M["churn_model.pth"]
        N["scaler.pkl"]
        O["config.json"]
    end
    F --> G
    I --> J
    L --> M
    L --> N
    L --> O

    classDef done fill:#0d3b2e,stroke:#b4ff3d,stroke-width:2px,color:#ffffff
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O done
```

🟢 every stage complete

> [!IMPORTANT]
> **No data leakage:** the scaler is fit on the training data only (`fit_transform`), and the test data only gets `transform`.

<img src="assets/divider.svg" width="100%" alt=""/>

## 🧱 Architecture

### 🏆 v2: final model

<div align="center">
  <img src="assets/architecture_v2.svg" width="100%" alt="Animated architecture diagram of the final 10-32-16-8-1 network"/>
</div>

| Layer | Operation | Output | Parameters |
|---|---|---|---|
| Input | 10 scaled features | `(N, 10)` | n/a |
| Hidden 1 | `Linear(10 → 32)` + `ReLU` | `(N, 32)` | 352 |
| Hidden 2 | `Linear(32 → 16)` + `ReLU` | `(N, 16)` | 528 |
| Hidden 3 | `Linear(16 → 8)` + `ReLU` | `(N, 8)` | 136 |
| Output | `Linear(8 → 1)` → **logit** | `(N, 1)` | 9 |
| **Total** | | | **1,025** |

*`N` = number of customers in the batch. In the animation, hidden layer 1 draws 8 of its 32 neurons and hidden layer 2 draws 6 of its 16.*

The output is a **logit**: any real number from −∞ to +∞, not yet a probability. `torch.sigmoid` is applied only when evaluating or predicting.

<details>
<summary><b>🕰️ v1: the first model (click to expand)</b></summary>

<div align="center">
  <img src="assets/architecture.svg" width="100%" alt="Animated architecture diagram of the original 10-16-8-1 network"/>
</div>

| Layer | Operation | Output | Parameters |
|---|---|---|---|
| Input | 10 scaled features | `(N, 10)` | n/a |
| Hidden 1 | `Linear(10 → 16)` + `ReLU` | `(N, 16)` | 176 |
| Hidden 2 | `Linear(16 → 8)` + `ReLU` | `(N, 8)` | 136 |
| Output | `Linear(8 → 1)` + `Sigmoid` | `(N, 1)` | 9 |
| **Total** | | | **321** |

*Hidden layer 1 draws 8 of its 16 neurons.*

<img src="assets/Visual.png" width="100%" alt="Static architecture diagram of the v1 network"/>

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

## 🏋️ Training

```mermaid
flowchart LR
    A["Forward pass"] --> B["Logits"] --> C["Weighted BCE loss"] --> D["Backpropagation"] --> E["Gradients"] --> F["Adam update"] --> G["Repeat"]
    G -.-> A
```

```python
# class weight = negatives / positives  (≈ 12.8026 on the training set)
pos_weight = (y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()

loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(neural_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = neural_model(X_train_scaled_tensor)
    loss   = loss_fn(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
```

> [!TIP]
> **Lesson learned:** more epochs do not automatically mean a better model. The weighted model performed better at an earlier training stage than after continuing to 10,000 epochs.

<img src="assets/divider.svg" width="100%" alt=""/>

## 🔍 Findings

### 🎚️ Threshold tuning (the heart of the project)

The network gives a probability. The threshold decides who gets flagged. Sliding it trades **recall** against **precision**:

<div align="center">
  <img src="assets/threshold.svg" width="100%" alt="Animated grouped bar chart of recall, precision and F1 at thresholds 0.3 to 0.7"/>
</div>

| Threshold | TN | FP | FN | TP | Recall | Precision | F1 | Accuracy |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.3** ⭐ | 5,824 | 3,451 | 92 | 633 | **87.31%** | 15.50% | 26.33% | 64.57% |
| 0.4 | 6,634 | 2,641 | 116 | 609 | 84.00% | 18.74% | 30.64% | 72.43% |
| 0.5 | 7,326 | 1,949 | 164 | 561 | 77.38% | 22.35% | 34.68% | 78.87% |
| 0.6 | 7,875 | 1,400 | 224 | 501 | 69.10% | 26.35% | 38.16% | 83.76% |
| 0.7 | 8,378 | 897 | 271 | 454 | 62.62% | 33.60% | 43.74% | 88.32% |

```mermaid
flowchart LR
    L["⬇️ Lower threshold"] --> L1["More customers flagged"] --> L2["More TP + more FP"] --> L3["📈 Higher recall<br/>📉 Lower precision"]
    H["⬆️ Higher threshold"] --> H1["Fewer customers flagged"] --> H2["Fewer FP + more FN"] --> H3["📈 Higher precision<br/>📉 Lower recall"]

    classDef lo fill:#0d3b2e,stroke:#b4ff3d,color:#ffffff
    classDef hi fill:#4a0d4a,stroke:#ff3df2,color:#ffffff
    class L,L1,L2,L3 lo
    class H,H1,H2,H3 hi
```

<details>
<summary><b>🔬 Explore each threshold (click to expand)</b></summary>

<details>
<summary>Threshold <b>0.3</b> ⭐ chosen</summary>

|  | Predicted stay | Predicted churn |
|---|---:|---:|
| **Actually stayed** | 5,824 (TN) | 3,451 (FP) |
| **Actually churned** | 92 (FN) | 633 (TP) |

Catches 633 of 725 churners, but flags 3,451 loyal customers by mistake.

</details>

<details>
<summary>Threshold <b>0.4</b></summary>

|  | Predicted stay | Predicted churn |
|---|---:|---:|
| **Actually stayed** | 6,634 (TN) | 2,641 (FP) |
| **Actually churned** | 116 (FN) | 609 (TP) |

</details>

<details>
<summary>Threshold <b>0.5</b> (conventional default)</summary>

|  | Predicted stay | Predicted churn |
|---|---:|---:|
| **Actually stayed** | 7,326 (TN) | 1,949 (FP) |
| **Actually churned** | 164 (FN) | 561 (TP) |

</details>

<details>
<summary>Threshold <b>0.6</b></summary>

|  | Predicted stay | Predicted churn |
|---|---:|---:|
| **Actually stayed** | 7,875 (TN) | 1,400 (FP) |
| **Actually churned** | 224 (FN) | 501 (TP) |

</details>

<details>
<summary>Threshold <b>0.7</b></summary>

|  | Predicted stay | Predicted churn |
|---|---:|---:|
| **Actually stayed** | 8,378 (TN) | 897 (FP) |
| **Actually churned** | 271 (FN) | 454 (TP) |

</details>

</details>

### 🏆 Final decision: threshold = 0.30

| Metric | Formula | Result |
|---|---|---:|
| Recall | 633 / (633 + 92) | **87.31%** |
| Precision | 633 / (633 + 3,451) | 15.50% |
| F1-score | harmonic mean | 26.33% |
| Accuracy | (5,824 + 633) / 10,000 | 64.57% |

> [!NOTE]
> The low precision is expected. Choosing an aggressive threshold to maximise churn detection means flagging many customers who would have stayed. If a retention offer is cheap relative to losing a customer, this is the right side of the trade-off. If outreach is costly, a higher threshold from the table above fits better.

### 🕰️ v1 baseline: unweighted model, 100 epochs

<div align="center">
  <img src="assets/chart.svg" width="100%" alt="Animated bar chart of training and test BCE loss for the v1 model"/>
</div>

| Metric (v1, unweighted `BCELoss`) | Value |
|---|---|
| Epochs | 100 |
| Training loss, start | ≈ 0.69 |
| Training loss, end | ≈ 0.39 |
| Test loss | ≈ 0.389 |

> [!NOTE]
> These losses belong to the **v1 baseline** and are not comparable with the weighted loss of the final model, because `pos_weight` changes the loss scale. A low v1 loss also did not guarantee good churn detection, which is why the project moved to a weighted loss and threshold tuning.

<img src="assets/divider.svg" width="100%" alt=""/>

## 🔮 Predicting a New Customer

<div align="center">
  <img src="assets/prediction.svg" width="100%" alt="Animated probability gauge showing 91.73 percent churn probability against a 0.30 threshold"/>
</div>

```mermaid
flowchart LR
    A["New customer<br/>(1, 10)"] --> B["Saved StandardScaler"] --> C["float32 tensor"] --> D["Trained ANN"] --> E["Logit"] --> F["Sigmoid"] --> G["Probability<br/>0.9173"] --> H{"≥ 0.30 ?"}
    H -->|yes| I["🔴 CHURN"]
    H -->|no| J["🟢 STAY"]

    classDef hot fill:#4a0d4a,stroke:#ff3df2,stroke-width:3px,color:#ffffff
    classDef ok fill:#0d3b2e,stroke:#b4ff3d,color:#ffffff
    class I hot
    class J ok
```

<details>
<summary><b>🧑 Example customer and inference code (click to expand)</b></summary>

| Feature | Value |
|---|---:|
| `age` | 35 |
| `tenure_months` | 12 |
| `monthly_charges` | 80 |
| `support_tickets` | 5 |
| `late_payments` | 2 |
| `usage_hours` | 25 |
| `satisfaction` | 6 |
| `contract_monthly` | 1 |
| `tech_support` | 0 |
| `auto_payment` | 0 |

```python
import json, joblib, torch, torch.nn as nn, pandas as pd

model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 8),  nn.ReLU(),
    nn.Linear(8, 1),
)
model.load_state_dict(torch.load("models/churn_model.pth"))
model.eval()

scaler    = joblib.load("models/scaler.pkl")
threshold = json.load(open("models/config.json"))["threshold"]   # 0.3

customer = pd.DataFrame([{
    "age": 35, "tenure_months": 12, "monthly_charges": 80,
    "support_tickets": 5, "late_payments": 2, "usage_hours": 25,
    "satisfaction": 6, "contract_monthly": 1,
    "tech_support": 0, "auto_payment": 0,
}])

x = torch.tensor(scaler.transform(customer), dtype=torch.float32)   # shape (1, 10)

with torch.no_grad():
    prob = torch.sigmoid(model(x)).item()                           # ≈ 0.9173

print("CHURN" if prob >= threshold else "STAY", f"({prob:.2%})")
```

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

## 📦 Model Package

```text
models/
├── churn_model.pth   # state_dict: learned weights and biases
├── scaler.pkl        # fitted StandardScaler (joblib)
└── config.json       # {"threshold": 0.3}
```

| File | Why it matters |
|---|---|
| `churn_model.pth` | What the ANN learned during training |
| `scaler.pkl` | New data **must** be scaled with the same training statistics |
| `config.json` | Stores the chosen decision threshold so inference matches evaluation |

<img src="assets/divider.svg" width="100%" alt=""/>

## 🧠 Key Decisions

<details>
<summary><b>Why an ANN?</b></summary>

The dataset was generated with **nonlinear relationships and feature interactions**. A network with hidden layers and ReLU activations can represent those patterns, which a purely linear model could not.

</details>

<details>
<summary><b>Why 10 → 32 → 16 → 8 → 1?</b></summary>

The first model was `10 → 16 → 8 → 1`. It was later widened to `10 → 32 → 16 → 8 → 1`: one input per feature, three hidden layers that narrow step by step, and a single output for churn. It stays small (1,025 parameters) and easy to inspect.

</details>

<details>
<summary><b>Why BCEWithLogitsLoss instead of Sigmoid + BCELoss?</b></summary>

`BCEWithLogitsLoss` fuses sigmoid and binary cross-entropy in one numerically stable operation, so the model's last layer outputs raw logits. It also accepts `pos_weight`, which was needed for the imbalance.

</details>

<details>
<summary><b>Why pos_weight ≈ 12.80?</b></summary>

Training had about 12.8× more non-churners than churners. Weighting positives by `negatives / positives` makes a missed churner about 12.8× more costly, so the model can no longer ignore the minority class. This significantly increased recall.

</details>

<details>
<summary><b>Why StandardScaler, and why fit on train only?</b></summary>

Features sit on very different scales (for example `usage_hours` vs `support_tickets`), and gradient-based training works better on comparable scales. The scaler learns its mean and variance from the training data only, so nothing from the test set leaks into training.

</details>

<details>
<summary><b>Why Adam (lr = 0.001)?</b></summary>

Adam adapts each weight's step size using the gradients from backpropagation, and it is a solid default for a small feed-forward network.

</details>

<details>
<summary><b>Why threshold 0.30 and not 0.50?</b></summary>

Because the project prioritises catching potential churners. Moving from 0.5 to 0.3 lifts recall from 77.38% to 87.31% (633 vs 561 churners caught) at the cost of precision. The threshold is a business choice, not a fixed model property, which is why it lives in `config.json`.

</details>

<details>
<summary><b>Why not judge by accuracy?</b></summary>

Predicting "no churn" for everyone already gives 92.75% accuracy and detects zero churners. Recall, precision, F1 and the confusion matrix reveal what accuracy hides.

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

## ✅ Progress & Roadmap

- [x] Dataset preparation (50,000 synthetic customers)
- [x] Stratified 80/20 train / test split
- [x] Feature scaling (`StandardScaler`, fit on train only)
- [x] Tensor conversion (`float32`)
- [x] v1 ANN (10 → 16 → 8 → 1) with `Sigmoid` + `BCELoss`
- [x] Identified class imbalance (7.25% churners)
- [x] Class-weighted `BCEWithLogitsLoss` (`pos_weight ≈ 12.80`)
- [x] v2 ANN (10 → 32 → 16 → 8 → 1), logits output
- [x] Backpropagation, gradients and Adam optimisation
- [x] Epoch-length experiments (more epochs ≠ better)
- [x] Threshold tuning (0.3 to 0.7) with confusion matrices
- [x] Final threshold 0.30: recall 87.31%, precision 15.50%, F1 26.33%, accuracy 64.57%
- [x] Saved model, scaler and threshold
- [x] New-customer prediction (91.73% → CHURN)

**💡 Possible next steps:** precision-recall curve, cost-based threshold selection, early stopping, comparison with a gradient-boosting baseline.

<img src="assets/divider.svg" width="100%" alt=""/>

## 🚀 Getting Started

```bash
git clone https://github.com/Kartikshard/ann_customer_churn.git
cd ann_customer_churn

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch pandas numpy scikit-learn joblib jupyter

jupyter notebook ann.ipynb
```

<details>
<summary><b>🧩 Final model definition</b></summary>

```python
import torch
import torch.nn as nn

neural_model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 8),  nn.ReLU(),
    nn.Linear(8, 1),   # raw logit, no Sigmoid
)

loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # ≈ 12.80
optimizer = torch.optim.Adam(neural_model.parameters(), lr=0.001)
```

</details>

<details>
<summary><b>🛠️ Tech stack</b></summary>

| Area | Tools |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| ML | scikit-learn (`train_test_split`, `StandardScaler`, `confusion_matrix`) |
| Deep learning | PyTorch (`nn.Sequential`, `nn.Linear`, `nn.ReLU`, `BCEWithLogitsLoss`, Adam, tensors, backprop) |
| Persistence | PyTorch `.pth`, Joblib `.pkl`, JSON config |
| Environment | Jupyter Notebook |

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=140&section=footer&text=Thanks%20for%20stopping%20by&fontSize=24&fontColor=ffffff&animation=twinkling" width="100%" alt="footer wave"/>
</div>
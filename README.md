<div align="center">

<img src="assets/banner.svg" width="100%" alt="ANN Customer Churn Predictor banner"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3500&pause=1000&color=00F0FF&center=true&vCenter=true&width=780&height=48&lines=Predicting+customer+churn+with+PyTorch;10+%E2%86%92+16+%E2%86%92+8+%E2%86%92+1+%C2%B7+ReLU+%C2%B7+Sigmoid+%C2%B7+BCE;Training+loss+0.69+%E2%86%92+0.39+in+100+epochs;Next+up%3A+accuracy%2C+precision%2C+recall+%26+F1" alt="Typing SVG"/></a>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Stage](https://img.shields.io/badge/stage-model_evaluation-ff3df2?style=flat-square)
![Training](https://img.shields.io/badge/training-100_epochs_done-b4ff3d?style=flat-square)
![Test loss](https://img.shields.io/badge/test_loss-~0.389-00f0ff?style=flat-square)
![Dataset](https://img.shields.io/badge/dataset-50k_customers-faff3d?style=flat-square)

</div>

<img src="assets/divider.svg" width="100%" alt=""/>

## 📑 Table of Contents

- [🎯 Project Goal](#-project-goal)
- [📦 Dataset](#-dataset)
- [🧪 Pipeline](#-pipeline)
- [🧱 Architecture](#-architecture)
- [🔍 Findings](#-findings)
- [🧠 Key Decisions](#-key-decisions)
- [✅ Progress & Roadmap](#-progress--roadmap)
- [🚀 Getting Started](#-getting-started)

<img src="assets/divider.svg" width="100%" alt=""/>

## 🎯 Project Goal

Build an **Artificial Neural Network in PyTorch** that answers one question:

> **Given a customer's behaviour and account details, will they churn (`1`) or stay (`0`)?**

The network outputs a churn probability between 0 and 1 for every customer.

> [!NOTE]
> **Where I am right now:** the model is trained and the test loss is computed. The next step is turning probabilities into `0/1` predictions and scoring them with accuracy, precision, recall and F1.

<img src="assets/divider.svg" width="100%" alt=""/>

## 📦 Dataset

A **synthetically generated** customer churn dataset with **50,000 customers**. It was intentionally built with **nonlinear relationships and feature interactions**, so an ANN has real patterns to learn.

| | | | | |
|---|---|---|---|---|
| `age` | `tenure_months` | `monthly_charges` | `support_tickets` | `late_payments` |
| `usage_hours` | `satisfaction` | `contract_monthly` | `tech_support` | `auto_payment` |

| Item | Detail |
|---|---|
| 🧮 Rows | 50,000 customers |
| 🔢 Input features | 10 (listed above) |
| 🎯 Target | `churn` → `0` = no churn, `1` = churn |
| 🧬 Source | Synthetic (generated) |

<img src="assets/divider.svg" width="100%" alt=""/>

## 🧪 Pipeline

```mermaid
flowchart LR
    subgraph DATA["📦 Data prep"]
        A["📄 CSV dataset<br/>50,000 rows"] --> B["🐼 Pandas<br/>load"]
        B --> C["✂️ Separate X and y"]
        C --> D["🔀 Train / test split"]
        D --> E["📏 StandardScaler"]
        E --> F["🔥 PyTorch tensors"]
    end
    subgraph MODEL["🧠 Model"]
        G["🕸️ ANN<br/>10 → 16 → 8 → 1"] --> H["📉 BCE loss + Adam<br/>100 epochs"]
        H --> I["🧪 Test loss<br/>≈ 0.389"]
    end
    subgraph EVAL["📊 Evaluation"]
        J["🎯 Probabilities → 0/1"] --> K["📊 Accuracy · Precision<br/>Recall · F1"]
    end
    F --> G
    I --> J

    classDef done fill:#0d3b2e,stroke:#b4ff3d,stroke-width:2px,color:#ffffff
    classDef next fill:#4a0d4a,stroke:#ff3df2,stroke-width:3px,stroke-dasharray:5 3,color:#ffffff
    classDef todo fill:#1a1a2e,stroke:#6b6b8a,color:#9a9ab8
    class A,B,C,D,E,F,G,H,I done
    class J next
    class K todo
```

🟢 done &nbsp;·&nbsp; 🟣 current step &nbsp;·&nbsp; ⚪ upcoming

<img src="assets/divider.svg" width="100%" alt=""/>

## 🧱 Architecture

<div align="center">
  <img src="assets/architecture.svg" width="100%" alt="Animated architecture diagram of the 10-16-8-1 network"/>
</div>

| Layer | Operation | Output | Parameters |
|---|---|---|---|
| Input | 10 scaled features | `(N, 10)` | n/a |
| Hidden 1 | `Linear(10 → 16)` + `ReLU` | `(N, 16)` | 176 |
| Hidden 2 | `Linear(16 → 8)` + `ReLU` | `(N, 8)` | 136 |
| Output | `Linear(8 → 1)` + `Sigmoid` | `(N, 1)` | 9 |
| **Total** | | | **321** |

*`N` = number of customers in the batch. In the animation, hidden layer 1 draws 8 of its 16 neurons.*

<details>
<summary><b>🔁 Training loop (click to expand)</b></summary>

```mermaid
flowchart LR
    A["Forward pass"] --> B["Prediction"] --> C["BCE loss"] --> D["Backpropagation"] --> E["Gradients"] --> F["Adam update"] --> G["Repeat x100"]
    G -.-> A
```

</details>

<details>
<summary><b>🖼️ Static diagram (Visual.png)</b></summary>

<img src="assets/Visual.png" width="100%" alt="Static architecture diagram"/>

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

## 🔍 Findings

### 📉 Training results (100 epochs)

<div align="center">
  <img src="assets/chart.svg" width="100%" alt="Animated bar chart of training and test BCE loss"/>
</div>

| Metric | Value |
|---|---|
| Epochs | 100 |
| Training loss, start | ≈ 0.69 |
| Training loss, end | ≈ 0.39 |
| Test loss | ≈ 0.389 |

> [!NOTE]
> Training loss and test loss ended at nearly the same value (≈ 0.39 vs ≈ 0.389). The chart shows start and end values only, not a per-epoch curve.

### 🎯 Classification metrics

| Metric | Result |
|---|---|
| Accuracy | ⏳ not computed yet |
| Precision | ⏳ not computed yet |
| Recall | ⏳ not computed yet |
| F1-score | ⏳ not computed yet |

<img src="assets/divider.svg" width="100%" alt=""/>

## 🧠 Key Decisions

<!-- ✏️ The reasoning below is my wording of standard rationale. Edit it to match your own thinking. -->

<details>
<summary><b>Why an ANN?</b></summary>

The dataset was generated with **nonlinear relationships and feature interactions**. A network with hidden layers and ReLU activations can represent those patterns, which a purely linear model could not.

</details>

<details>
<summary><b>Why 10 → 16 → 8 → 1?</b></summary>

One input per feature, two hidden layers that narrow from 16 to 8 neurons, and a single output neuron for the churn probability. It is small (321 parameters) and easy to inspect.

</details>

<details>
<summary><b>Why ReLU, Sigmoid and BCELoss?</b></summary>

`ReLU` adds nonlinearity in the hidden layers. `Sigmoid` squeezes the output into (0, 1) so it reads as a probability. `BCELoss` is the matching loss for a binary target.

</details>

<details>
<summary><b>Why StandardScaler after the split?</b></summary>

Features sit on very different scales (for example `monthly_charges` vs `satisfaction`), and gradient-based training works better on standardised inputs. The scaler comes after the train/test split in the pipeline.

</details>

<details>
<summary><b>Why Adam?</b></summary>

Adam is a solid default optimiser for this kind of small feed-forward network.

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

## ✅ Progress & Roadmap

- [x] Dataset preparation
- [x] Train / test split
- [x] Feature scaling (`StandardScaler`)
- [x] Tensor conversion
- [x] ANN architecture (10 → 16 → 8 → 1)
- [x] Forward propagation and binary cross-entropy loss
- [x] Backpropagation, gradients and Adam optimisation
- [x] 100-epoch training
- [x] Test-set prediction and test loss
- [ ] 👉 **CURRENT:** convert predicted probabilities into `0/1` predictions
- [ ] Evaluate with accuracy, precision, recall and F1-score

<img src="assets/divider.svg" width="100%" alt=""/>

## 🚀 Getting Started

```bash
git clone https://github.com/Kartikshard/ann_customer_churn.git
cd ann_customer_churn

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch pandas numpy scikit-learn jupyter

jupyter notebook <your-notebook>.ipynb
```

<details>
<summary><b>🧩 Model definition</b></summary>

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 16), nn.ReLU(),
    nn.Linear(16, 8),  nn.ReLU(),
    nn.Linear(8, 1),   nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())  # set your learning rate here
```

</details>

<img src="assets/divider.svg" width="100%" alt=""/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=140&section=footer&text=Thanks%20for%20stopping%20by&fontSize=24&fontColor=ffffff&animation=twinkling" width="100%" alt="footer wave"/>
</div>
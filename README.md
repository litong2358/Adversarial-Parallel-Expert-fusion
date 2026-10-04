# Robust Expert Fusion

Improving adversarial robustness through **parallel sub-network specialization and fusion**, evaluated on two adversarial training frameworks: [TRADES](https://arxiv.org/pdf/1901.08573.pdf) (ICML 2019) and [DKL](https://arxiv.org/pdf/2305.13948) (NeurIPS 2024).

## Motivation

Standard adversarial training uses a single monolithic network to learn robust representations across all classes simultaneously. We hypothesize that **splitting the classification task among specialized expert sub-networks**, each responsible for a semantically coherent subset of classes, and then **fusing their outputs** can yield:

1. **Higher clean accuracy** — each expert focuses on a smaller, more coherent label space.
2. **Improved adversarial robustness** — specialized experts produce more discriminative features that are harder to fool.
3. **Flexible routing** — learning to weight expert opinions based on per-sample confidence can further improve performance under adversarial perturbation.

We systematically evaluate four fusion strategies across two adversarial training losses (TRADES, DKL) and two datasets (CIFAR-10, CIFAR-100).

## Method

### Training Pipeline

**Parallel methods** (concat, routing_ab, routing_conf) — 2 stages:

1. **Stage 1 — Expert Pre-training:** Each WideResNet-34-10 sub-network is trained independently on its own class subset, with an additional *unknown* class for out-of-distribution inputs. EMA is applied for stabilization.
2. **Stage 2 — Fusion + Adversarial Training:** Expert backbones are frozen, a fusion head (FC or routing mechanism) is added, and the full model is fine-tuned end-to-end with adversarial training (TRADES or DKL loss).

**Gating methods** (gated) — 3 stages:

1. **Stage 1 — Expert Pre-training:** Same as above.
2. **Stage 2 — Gate Training:** Expert backbones are frozen, and the gating module is trained to learn the fusion weights.
3. **Stage 3 — Joint Adversarial Fine-tuning:** The full model (experts + gate) is fine-tuned end-to-end with adversarial training, using a reduced backbone learning rate.

### Class Splits

**CIFAR-10 (2 experts):** Vehicles {airplane, automobile, ship, truck} vs. Animals {bird, cat, deer, dog, frog, horse}.

**CIFAR-100 (4 experts):** 100 fine classes grouped into 4 semantically coherent super-groups of 25 classes each (textured organic, smooth organic, rigid man-made, large structures).

### Fusion Strategies

| Method | Abbreviation | Mechanism |
|--------|:---:|-----------|
| **Embedding Concatenation** | `concat` | `emb = [e₁ ; e₂]`, `logits = FC(emb)` |
| **Feature-level Gated Fusion** | `gated` | `α = sigmoid(MLP([e₁ ; e₂]))`, `emb = α · e₁ + (1−α) · e₂`, `logits = FC(emb)` |
| **a,b Routing** | `routing_ab` | `score_i = a · (1 − unk_i) + b · unk_j`, `w = softmax(scores)`, `logits = Σ wᵢ · logitsᵢ` |
| **Confidence-only Routing** | `routing_conf` | `score_i = 1 − unk_i`, `w = softmax(scores)`, `logits = Σ wᵢ · logitsᵢ` |

## Results

### CIFAR-10

#### TRADES (WRN-34-10, ε = 8/255, β = 6.0)

| Method | Params | Clean Acc (%) | PGD-20 Acc (%) | AutoAttack Acc (%) |
|--------|--------|:---:|:---:|:---:|
| TRADES Baseline | 46.2M | 84.62 | 55.30 | 51.31 |
| Concat + FC | 92.4M+ | 88.75 | 59.32 | 53.36 |
| Feature-level Gated | 92.4M+ | 89.13 | 58.71 | 53.02 |
| a,b Routing | 92.4M+ | 87.75 | 58.88 | 53.18 |
| Confidence-only Routing | 92.4M+ | 88.04 | 58.74 | 53.66 |

#### DKL (WRN-34-10, ε = 8/255, α = 4.0, β = 20.0)

| Method | Architecture | Clean Acc (%) | AutoAttack Acc (%) |
|--------|-------------|:---:|:---:|
| DKL Baseline | WRN-34-10 | 84.30 | 56.35 |
| **Concat + FC** | **WRN-34-10 x 2 + FC** | **86.58** | **57.15** |

### CIFAR-100

#### TRADES (WRN-34-10, ε = 8/255, β = 6.0)

| Method | Clean Acc (%) | PGD-20 Acc (%) | AutoAttack Acc (%) |
| --- | :---: | :---: | :---: |
| TRADES Baseline | 60.37 | 32.42 | 27.08 |
| **Concat + FC** | **64.57** | **33.82** | **27.96** |

#### DKL (WRN-34-10, ε = 8/255, α = 4.0, β = 20.0)

| Method | Clean Acc (%) | AutoAttack Acc (%) |
| --- | :---: | :---: |
| DKL Baseline | 65.18 | 31.22 |
| **Concat + FC** | **69.36** | **32.54** |

### Key Findings

- All parallel expert methods outperform the single-model baselines in clean accuracy (+3\~4.5%), PGD-20 robustness (+3\~4%), and AutoAttack accuracy (+0.8\~2.3%).
- **Confidence-only routing** achieves the best AutoAttack accuracy (53.66%) with the simplest routing mechanism, indicating that complex routing may not be necessary.

## Project Structure

```
robust-expert-fusion/
│
├── trades/                              # TRADES-based experiments
│   ├── models/                          # Model architectures
│   │   ├── wideresnet.py                #   WideResNet (original TRADES)
│   │   ├── wideresnet_update.py         #   WideResNet (updated)
│   │   ├── parallel_wrn.py              #   WRNWithEmbedding + ParallelFusionWRN
│   │   └── resnet.py                    #   Standard ResNet
│   ├── losses/
│   │   └── trades.py                    #   TRADES loss function
│   ├── train/
│   │   ├── cifar10/
│   │   │   ├── baseline.py              #   TRADES baseline
│   │   │   ├── concat.py                #   Embedding concatenation + FC
│   │   │   ├── gated.py                 #   Feature-level gated fusion
│   │   │   ├── routing_ab.py            #   a,b routing
│   │   │   └── routing_conf.py          #   Confidence-only routing
│   │   └── cifar100/
│   │       ├── baseline.py              #   TRADES baseline
│   │       ├── concat.py                #   Embedding concatenation + FC
│   │       ├── gated.py                 #   Feature-level gated fusion
│   │       ├── routing_ab.py            #   a,b routing
│   │       └── routing_uniform.py       #   Uniform routing
│   └── eval/
│       ├── cifar10/
│       │   ├── pgd_*.py                 #   PGD-20 evaluation
│       │   └── aa_*.py                  #   AutoAttack evaluation
│       └── cifar100/
│           ├── pgd_*.py                 #   PGD-20 evaluation
│           └── aa_*.py                  #   AutoAttack evaluation
│
├── dkl/                                 # DKL-based experiments
│   ├── baseline/                        #   DKL baseline (self-contained)
│   │   ├── models/                      #     WideResNet definitions
│   │   ├── utils/                       #     Logger, metrics, misc
│   │   ├── dataset/                     #     CIFAR data loading
│   │   ├── auto_attacks/                #     AutoAttack evaluation
│   │   ├── awp.py                       #     Adversarial Weight Perturbation
│   │   ├── augmentation.py              #     AutoAugment + Cutout
│   │   └── swa.py                       #     Stochastic Weight Averaging
│   ├── models/                          #   Fusion model architectures
│   │   ├── cifar10/                     #     ParallelFusionWRN, GatedFusionWRN, SoftRouting
│   │   └── cifar100/                    #     ParallelFusionWRN100 (4-group)
│   ├── train/
│   │   ├── cifar10/
│   │   │   ├── baseline.py              #   DKL baseline
│   │   │   ├── concat.py                #   Embedding concatenation + FC
│   │   │   ├── gated.py                 #   Feature-level gated fusion
│   │   │   ├── routing_ab.py            #   a,b routing
│   │   │   └── routing_conf.py          #   Confidence-only routing
│   │   └── cifar100/
│   │       ├── baseline.py              #   DKL baseline
│   │       └── concat.py                #   Embedding concatenation + FC (4 experts)
│   ├── eval/
│   │   ├── cifar10/
│   │   │   └── aa_*.py                  #   AutoAttack evaluation
│   │   └── cifar100/
│   │       └── aa_concat.py             #   AutoAttack evaluation
│   └── scripts/                         #   SLURM job scripts
│
├── README.md
└── LICENSE
```

## Requirements

- Python 3.8+
- PyTorch 1.x+ with CUDA
- torchvision
- numpy
- [autoattack](https://github.com/fra31/auto-attack) (`pip install autoattack`)

## Hyperparameters

### TRADES

| Parameter | Value |
|-----------|-------|
| Architecture | WideResNet-34-10 per expert |
| ε (L∞) | 8/255 ≈ 0.031 |
| PGD steps (train) | 10 |
| Step size | 0.007 |
| β (TRADES) | 6.0 |
| Batch size | 128 |
| Learning rate | 0.1 (cosine / multi-step decay) |
| Weight decay | 5 x 10⁻⁴ |
| EMA decay | 0.999 |

### DKL

| Parameter | Value |
|-----------|-------|
| Architecture | WideResNet-34-10 per expert |
| ε (L∞) | 8/255 ≈ 0.031 |
| PGD steps (train) | Progressive (2 → 5) |
| α (DKL SCE weight) | 4.0 |
| β (DKL MSE weight) | 20.0 |
| T (temperature) | 4.0 |
| AWP γ | 0.005 |
| Batch size | 128 |
| Learning rate | 0.1 (cosine annealing) |
| Weight decay | 5 x 10⁻⁴ |
| EMA decay | 0.999 |

## References

```bibtex
@inproceedings{zhang2019theoretically,
    author    = {Hongyang Zhang and Yaodong Yu and Jiantao Jiao and
                 Eric P. Xing and Laurent El Ghaoui and Michael I. Jordan},
    title     = {Theoretically Principled Trade-off between Robustness and Accuracy},
    booktitle = {International Conference on Machine Learning (ICML)},
    year      = {2019}
}

@inproceedings{cui2024decoupled,
    author    = {Jiequan Cui and Zhuotao Tian and Zhisheng Zhong and
                 Xiaojuan Qi and Bei Yu and Hanwang Zhang},
    title     = {Decoupled Kullback-Leibler Divergence Loss},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

# Related Work

> This document summarizes existing research and projects most relevant to Castle-SeedFormer, and clarifies where this project sits within the broader landscape.

---

## Table of Contents

- [Most Relevant Research](#most-relevant-research)
- [Japanese Heritage Digitization Projects](#japanese-heritage-digitization-projects)
- [Positioning of This Research](#positioning-of-this-research)
- [General Point Cloud Completion Models](#general-point-cloud-completion-models)

---

## Most Relevant Research

### Building-PCC: Building Point Cloud Completion Benchmarks (2024)

The closest existing work to this project.

- **What it is**: A large-scale benchmark dataset specifically for urban building point cloud completion
- **Scale**: ~50,000 buildings (30,000 train / 10,000 val / 10,000 test)
- **Baselines evaluated**: PCN, FoldingNet, TopNet, GRNet, SnowflakeNet, PoinTr, AdaPoinTr, AnchorFormer
- **Paper**: [arXiv 2404.15644](https://arxiv.org/abs/2404.15644) — ISPRS Annals 2024
- **GitHub**: [tudelft3d/Building-PCC](https://github.com/tudelft3d/Building-PCC-Building-Point-Cloud-Completion-Benchmarks)

**Difference from this project:**
Building-PCC focuses on modern Western-style urban buildings captured by aerial LiDAR. It does not address traditional Japanese architectural geometry (curved rooflines, ishigaki stone walls) or smartphone-captured data with large occlusion gaps.

---

### Self-Supervised Large Scale Point Cloud Completion for Archaeological Site Restoration (2025)

The most similar in terms of scale and purpose.

- **What it is**: Point cloud completion applied to a real archaeological site (Mawchu Llacta, Peru)
- **Scale**: 15+ million points, 600+ incomplete structures
- **Method**: Self-supervised learning — no paired ground truth required
- **Paper**: [arXiv 2503.04030](https://arxiv.org/abs/2503.04030) / [IEEE Xplore](https://ieeexplore.ieee.org/document/11094182/)

**Difference from this project:**
Targets stone ruins in an open outdoor environment. The architectural geometry is simpler than Japanese castle structures (no layered rooflines, decorative elements). Data is captured by professional equipment, not consumer smartphones.

---

### CPDC-MFNet: Conditional Point Cloud Completion for Cultural Relics (2024)

The most similar in terms of domain (cultural heritage).

- **What it is**: A conditional diffusion completion network designed for cultural heritage artifacts
- **Target objects**: Terracotta warriors (兵馬俑), ceramic vessels, sculptures
- **Paper**: [Nature Scientific Reports](https://www.nature.com/articles/s41598-024-58956-1)

**Difference from this project:**
Targets small, tabletop-scale artifacts (sculptures, pottery). The objects are isolated and scannable from all angles. Castle architecture is orders of magnitude larger and has fundamentally different occlusion patterns.

---

## Japanese Heritage Digitization Projects

### OUR Shurijo — Shurijo Castle Digital Reconstruction

- **Background**: Shurijo Castle (Okinawa), a UNESCO World Heritage site, was largely destroyed by fire in October 2019
- **Method**: Structure from Motion (SfM) from ~80,000 photos contributed by ~3,000 people worldwide
- **Partners**: Google Arts & Culture
- **Website**: [our-shurijo.org](https://www.our-shurijo.org/en/)
- **Physical reconstruction**: Scheduled for completion in autumn 2026

**Relevance**: Demonstrates strong public and institutional interest in Japanese castle 3D digitization. Notably, this project relied entirely on photogrammetry — **no completion was applied to missing regions**.

---

### Eiheiji Temple Digital Twin (Shimizu Corporation, 2024)

- **Target**: 19 nationally designated cultural properties within Eiheiji Temple (800-year-old Soto Zen headquarters, Fukui)
- **Method**: Ground LiDAR point cloud surveying
- **Output**: Precise digital twin enabling extraction of floor plans, elevations, and cross-sections
- **Source**: [Shimizu Corporation News Release](https://www.shimz.co.jp/en/company/about/news-release/2024/2024004.html)

**Relevance**: Demonstrates professional-grade heritage digitization using high-cost LiDAR equipment — the approach this project aims to complement with a low-cost alternative.

---

## Positioning of This Research

The table below summarizes how this project compares to the most relevant existing work.

| Project | Target | Scale | Capture method | Completion? |
|---------|--------|-------|----------------|-------------|
| Building-PCC | Modern urban buildings | ~50,000 buildings | Aerial LiDAR | Yes |
| Archaeological Site Restoration | Stone ruins (Peru) | 15M+ points | Professional scanner | Yes |
| CPDC-MFNet | Cultural artifacts (sculptures) | Tabletop scale | Lab scanner | Yes |
| OUR Shurijo | Japanese castle | Single structure | Smartphone (SfM) | No |
| Eiheiji Digital Twin | Japanese temple | 19 structures | Ground LiDAR | No |
| **Castle-SeedFormer (ours)** | **Japanese castle** | **50–100M points** | **Smartphone (Luma AI)** | **Yes** |

**Key observations:**

- Projects that apply completion to architectural structures use professional equipment, not smartphones
- Projects using smartphones for Japanese heritage do not apply completion
- No existing work combines **smartphone capture + point cloud completion** for **Japanese castle architecture** at this scale

---

## General Point Cloud Completion Models

For reference, the major completion models in the literature, ordered roughly by recency:

| Model | Year | Key idea |
|-------|------|----------|
| FoldingNet | 2018 | Folds a 2D grid into 3D shape |
| TopNet | 2019 | Hierarchical tree-based point generation |
| PCN | 2019 | Coarse-to-fine with FoldingNet decoder |
| GRNet | 2020 | 3D grid as intermediate representation |
| PoinTr | 2021 | Transformer with geometry-aware tokens |
| SeedFormer | 2022 | Seed points + UpTransformer (base of this project) |
| AdaPoinTr | 2023 | Adaptive query generation for PoinTr |
| DAPoinTr | 2024 | Domain-adaptive point transformer |

For a comprehensive list, see [Awesome Point Cloud Completion](https://github.com/hitcslj/Awesome-Point-Cloud-Completion).

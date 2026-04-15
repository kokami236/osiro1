# Why Point Cloud Completion? — Motivation & Background

> This document explains why point cloud **completion** (AI-based restoration) is necessary for Japanese castle digitization, rather than simply improving the scanning process itself.

---

## Table of Contents

- [The Problem: Castles Are Hard to Scan](#the-problem-castles-are-hard-to-scan)
- [Why Not Just Use Better Equipment?](#why-not-just-use-better-equipment)
- [Why Existing Completion Models Don't Work](#why-existing-completion-models-dont-work)
- [Our Approach](#our-approach)
- [Who This Is For](#who-this-is-for)

---

## The Problem: Castles Are Hard to Scan

Scanning a castle with a smartphone (via photogrammetry tools like Luma AI) is accessible and low-cost, but it inevitably produces **incomplete point clouds** — regions where geometry is simply missing.

These missing regions are not random. They are structurally predictable:

```
Typical occlusion patterns in castle scans:

  ┌──────────────────────────────────────┐
  │          Roof underside              │  ← Cannot be photographed from below
  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │                                      │
  │  Wall surfaces (rear/high angle)     │  ← Obstructed by trees, terrain
  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  │
  │                                      │
  │  Stone wall (ishigaki) joints        │  ← Fine geometry lost at distance
  │  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓                │
  └──────────────────────────────────────┘

  ▓ = missing / incomplete regions
```

These gaps are a fundamental consequence of **viewpoint limitations** — no matter how many photos are taken, a smartphone camera cannot see through walls, under eaves, or into occluded geometry.

---

## Why Not Just Use Better Equipment?

The obvious alternative is to use professional-grade LiDAR scanners, which produce far denser and more complete scans.

| Method | Completeness | Cost | Accessibility |
|--------|-------------|------|---------------|
| Smartphone + Luma AI | △ Incomplete | Free–Low | Anyone |
| Drone LiDAR | ○ Good | ¥1M–¥5M+ | Specialist only |
| Ground LiDAR (FARO, Leica) | ◎ Excellent | ¥3M–¥10M+ | Specialist only |

**Cost is a critical barrier.** Japan has hundreds of castles, castle ruins, and related heritage structures. Deploying professional LiDAR teams to every site is financially and logistically unrealistic.

Furthermore, even professional LiDAR scans suffer from occlusion in complex architectural structures — the problem is reduced, not eliminated.

> The goal of this research is not to replace professional digitization,  
> but to make **meaningful 3D reconstruction accessible at low cost**.

---

## Why Existing Completion Models Don't Work

Point cloud completion is an active research area, but existing models are not directly applicable to castle architecture.

### Trained on the Wrong Data

The dominant benchmark dataset — **ShapeNet55** — consists of approximately 51,000 models across 55 categories:

```
ShapeNet55 categories (examples):
  airplane, car, chair, table, sofa, lamp, guitar, pistol, vessel ...
```

These are smooth, manufactured objects with simple geometry. Castle architecture has fundamentally different characteristics:

| Property | ShapeNet objects | Castle architecture |
|----------|-----------------|---------------------|
| Surface type | Smooth, uniform | Irregular stone, curved tile |
| Scale | Tabletop (~1 m) | Full structure (10–30 m) |
| Geometry complexity | Low–Medium | High (layered roofs, ishigaki) |
| Occlusion pattern | Simple | Complex, multi-level |

A model trained on chairs and airplanes does not generalize well to the curved eaves of a Japanese castle turret or the irregular surface of an ishigaki stone wall.

### Scale Mismatch

The original SeedFormer model was designed for inputs of **2,048 points** — suitable for ShapeNet-scale objects.

A single smartphone scan of a castle can contain **50–100 million points**. No existing completion model was designed to operate at this scale directly.

---

## Our Approach

We address both problems with a two-part solution:

### 1. Domain-Specific Fine-tuning

Rather than using ShapeNet-pretrained weights as-is, we fine-tune the model on **castle diorama data** — physical scale models of Japanese castles that closely replicate the actual architectural geometry (rooflines, walls, stone bases).

```
ShapeNet pretrained weights
        ↓ fine-tuning
Castle diorama dataset
        ↓
Model specialized for Japanese castle geometry
```

### 2. Patch-based Large-Scale Pipeline

To bridge the scale gap, we decompose the full castle scan into small overlapping patches (0.25 m³ cubes), process each independently, then merge the results:

```
Full castle scan (50–100M points)
        ↓ decompose into 0.25m patches
Individual patches (2,048 points each)
        ↓ SeedFormer completion
Completed patches (16,384 points each)
        ↓ merge + post-process
Completed full castle scan
```

This makes it possible to apply a model designed for small objects to real-world architectural-scale data.

---

## Who This Is For

| User | Use case |
|------|----------|
| Cultural heritage researchers | Low-cost 3D digitization of castles and ruins |
| Local governments / boards of education | Archiving regional heritage structures |
| Reconstruction projects | Restoring digital models of partially destroyed structures |
| 3D city model projects (e.g. PLATEAU) | Filling gaps in urban heritage 3D data |

The broader vision is that **anyone with a smartphone** should be able to produce a reasonably complete 3D model of a heritage structure — without specialist equipment, without a large budget, and without professional training.

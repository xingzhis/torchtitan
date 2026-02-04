# Principled Guide to Training Configuration

This guide summarizes what to keep constant and what to adjust when scaling models (Model Size) or hardware (GPUs).

## 1. Constants (Mostly Fixed)
These hyperparameters rarely change regardless of scale.
*   **Weight Decay**: `0.1` (Standard for AdamW).
*   **Max Norm (Gradient Clipping)**: `1.0`.
*   **Beta1, Beta2**: `0.9`, `0.95`.
*   **Eps**: `1e-8`.
*   **Precision**: `bfloat16` (Preferred over fp16).

## 2. Model-Dependent (Adjust with Model Size)
*   **Total Tokens**: $\approx 20 \times \text{Model Params}$ (Chinchilla Optimal).
    *   *Example*: 1B model $\rightarrow$ 20B tokens.
*   **Learning Rate**: **Decreases** as Model Size **Increases**.
    *   100M - 500M: `3e-4` to `6e-4`
    *   1B - 7B: `1e-4` to `3e-4`
    *   30B - 70B: `1e-5` to `1e-4`
    *   *Rule*: Large models are more unstable; use lower LR.
*   **Global Batch Size (GBS)**: **Increases** slightly with Model Size / Loss.
    *   Small (<1B): 0.5M - 1M tokens.
    *   Medium (1B - 10B): 2M - 4M tokens.
    *   Large (>70B): 4M+ tokens.
    *   *Note*: `1M tokens` (e.g. BS 256 @ seq 4096) is a very safe baseline for most small-to-medium models.

## 3. Hardware-Dependent (Adjust with Hardware)
*   **Local Batch Size**: Maximize this to fit in VRAM (Per-GPU).
    *   *Goal*: Maximize GPU Utilization.
    *   *Constraint*: Memory (OOM).
    *   *Formula*: Depends on static model memory + dynamic activation memory.
*   **Gradient Accumulation Steps**: Derived Variable.
    *   $\text{AccumSteps} = \frac{\text{Global Batch Size}}{\text{Local Batch Size} \times \text{N\_GPUs}}$
    *   *Role*: Bridges the gap between what hardware *can* run (Local) and what math *needs* (Global).

### Memory Estimation Formula
$$M_{total} \approx M_{static} + M_{dynamic}$$

**1. Static Memory (Model + Optimizer)**
$$M_{static} \approx 12 \text{ bytes} \times \text{NumParams}$$
*   For 0.6B output: $\approx 7.2 \text{ GB}$

**2. Dynamic Memory (Activations)**
$$M_{activations} \approx \text{BatchSize} \times \text{SeqLen} \times \text{HiddenDim} \times \text{Layers} \times \text{Factor}$$
*   Rough rule of thumb: ~0.5 MB per token-layer (without extensive checkpointing).
*   **Data Parallel Replicate Degree**: equal to `N_GPUs` (usually).

## 4. Time-Dependent (Derived from Training Duration)
*   **Total Steps**: Derived.
    *   $\text{Steps} = \frac{\text{Total Tokens}}{\text{Global Batch Size (in tokens)}}$
*   **Warmup Steps**:
    *   **Long Runs**: ~1% of Total Steps (or 2000 steps max).
    *   **Short/Test Runs**: ~5-10% of Total Steps (or 100-200 steps).

## Summary Table
| Parameter | Behavior | Typical Value |
| :--- | :--- | :--- |
| **LR** | ⬇️ as Model ⬆️ | `3e-4` (Small), `1e-4` (Medium) |
| **GBS (Tokens)** | ⬆️ as Model ⬆️ | 1M (Small), 4M (Large) |
| **Local BS** | ⬆️ as VRAM ⬆️ | Max possible w/o OOM |
| **Accum Steps** | Derived | Adjusts to match GBS |
| **Total Tokens** | 20x Params | Chinchilla Law |

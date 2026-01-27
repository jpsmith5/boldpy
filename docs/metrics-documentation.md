# Metrics Documentation

Complete reference for all metrics reported by BoldPy.

---

## Relaxation Metrics

### T2* (T2-star)
**Definition:** Transverse relaxation time including field inhomogeneities

**Units:** milliseconds (ms)

**Formula:** S(TE) = S0 × exp(-TE / T2*)

**Normal Values (Mouse Kidney):**
- Cortex: 27-37 ms
- Medulla: 25-35 ms

**Interpretation:**
- <40 ms: Healthy tissue
- 40-60 ms: Edema (fluid accumulation)
- >60 ms: Likely necrotic tissue

**References:** PMID: 20829708

---

### R2* (R2-star)
**Definition:** Transverse relaxation rate

**Units:** Hz (1/seconds)

**Formula:** R2* = 1000 / T2*

**Interpretation:**
- Higher R2* = Better oxygenation
- Lower R2* = Hypoxia or edema
- Inverse relationship with T2*

---

## Perfusion Metrics

### Perfusion
**Definition:** Blood flow to tissue

**Units:** 
- Relative % (Bruker default)
- mL/100g/min (if calibrated)

**Normal Values (Mouse Kidney):**
- Cortex: 150-400 mL/100g/min (relative: 80-150%)
- Medulla: 50-150 mL/100g/min (relative: 40-80%)

**Interpretation:**
- High perfusion: Active tissue, good blood supply
- Low perfusion (<50): Ischemia, hypoxia

---

## Tissue Quality Metrics

### Viable Percentage
**Definition:** Percentage of pixels classified as viable tissue

**Classification Criteria:**
- **Viable:** T2* < 40 ms AND perfusion > 150 (if available)
- **Suspect Edema:** T2* 40-60 ms
- **Likely Necrosis:** T2* > 60 ms AND perfusion < 50 (if available)

**Interpretation:**
- >90%: Healthy tissue ✓
- 75-90%: Mild damage ⚠️
- <75%: Significant damage 🔴

**Method:** Per-pixel classification, aggregated across region

---

### Suspect Edema Percentage
**Definition:** Percentage of pixels with elevated T2* suggesting edema

**Range:** 40-60 ms T2*

**Clinical Significance:**
- Fluid accumulation in tissue
- Inflammatory response
- Reversible if caught early

---

### Likely Necrosis Percentage
**Definition:** Percentage of pixels with very high T2* suggesting cell death

**Range:** >60 ms T2*

**Clinical Significance:**
- Cell death, tissue breakdown
- Mostly fluid-filled space
- Generally irreversible

---

## Spatial Metrics

### Layer-by-Layer Profiles
**Definition:** Metrics computed for each MLCO layer

**Layers:** Numbered 1 (surface) to N (center)

**Depth Percentage:** (layer_num - 1) / (total_layers - 1) × 100%

**Metrics per Layer:**
- Mean T2*, R2*, perfusion
- Standard deviation
- Number of pixels
- Tissue quality percentages

---

### 5-Zone Analysis (Kidney-Specific)
**Definition:** Anatomical zones from outer to inner

**Zones:**
1. **Cortex:** Outer 33% of layers
2. **Outer Medulla:** Next 16.7%
3. **CMJ:** Middle layers (16.7%)
4. **Inner Medulla:** Next 16.7%
5. **Papilla:** Inner 16.7%

**Note:** Will be generalized or user-customizable in future versions

---

### Cortex-Medulla Gradient
**Definition:** Difference in metrics between outer (cortex) and inner (medulla)

**Formula:** Gradient = Medulla_value - Cortex_value

**Interpretation:**
- T2* gradient: Usually positive (medulla slightly higher)
- Perfusion gradient: Usually negative (cortex has higher flow)
- Abnormal gradients suggest regional disease

---

## Oxygen Responsiveness Metrics

### ΔT2* (Delta T2-star)
**Definition:** Change in T2* from air to oxygen breathing

**Formula:** ΔT2* = T2*_oxygen - T2*_air

**Normal Response:** +2 to +5 ms (positive)

**Interpretation:**
- Positive ΔT2*: Normal response (vasodilation, increased blood volume)
- Near zero: Impaired response
- Negative ΔT2*: Abnormal (may indicate damage)

---

### Oxygen Response Categories
**Categories:**
- `viable_responsive`: Normal tissue, normal response
- `viable_hypoxic`: Normal baseline but poor response
- `impaired_responsive`: High T2* but responds
- `impaired_nonresponsive`: High T2* and no response

**Clinical Use:** Distinguish viable but hypoxic tissue from dead tissue

---

## Statistical Metrics

### Tissue Heterogeneity
**Definition:** Coefficient of variation (CV) of T2* within layers

**Formula:** CV = (std / mean) × 100%

**Interpretation:**
- Low CV (<15%): Homogeneous tissue
- High CV (>25%): Heterogeneous, patchy damage

---

### Effect Size (Cohen's d)
**Definition:** Standardized difference between groups

**Formula:** d = (mean1 - mean2) / pooled_std

**Interpretation:**
- |d| < 0.2: Small effect
- |d| 0.2-0.5: Medium effect  
- |d| 0.5-0.8: Large effect
- |d| > 0.8: Very large effect

**Use:** Assess biological significance of group differences

---

## Quadrant Analysis

### T2* vs Perfusion Quadrants
**Definition:** Classification based on both metrics

**Quadrants:**
1. **Low T2*, High Perfusion:** Healthy, well-perfused tissue
2. **Low T2*, Low Perfusion:** Ischemic but viable
3. **High T2*, High Perfusion:** Acute injury with preserved flow
4. **High T2*, Low Perfusion:** Chronic injury or necrosis

**Thresholds:**
- T2*: 40 ms
- Perfusion: 100 mL/100g/min (or 50% relative)

---

## JSON Output Structure

### Layer Statistics
```json
{
  "layer": 1,
  "depth_pct": 0.0,
  "n_pixels": 1247,
  "t2star": {"mean": 32.5, "std": 4.2},
  "r2star": {"mean": 30.8, "std": 3.9},
  "perfusion": {"mean": 245.0, "std": 35.0},
  "tissue_quality": {
    "viable_pct": 92.0,
    "suspect_edema_pct": 6.5,
    "likely_necrosis_pct": 1.5
  }
}
```

### Zone Statistics
```json
{
  "cortex": {
    "t2star": {"mean": 31.2, "std": 3.8},
    "tissue_quality": {
      "viable_pct": 94.5,
      "interpretation": "healthy"
    }
  }
}
```

---

## Visualization Guide

### Layer Profile Plots
- X-axis: Layer number (1 = surface, N = center)
- Y-axis: Metric value
- Error bars: Standard deviation
- Multiple lines: Different conditions

### Scatter Plots
- X-axis: T2* (ms)
- Y-axis: Perfusion (mL/100g/min)
- Color: Density
- Quadrants: Tissue state classification

### Triple Overlay
- Three side-by-side maps: T2*, R2*, Perfusion
- Same colormap scale for comparison
- Anatomical correspondence preserved


# Examples with Real Data

This guide walks through complete analysis examples using actual kidney BOLD MRI data.

---

## Example 1: Single Sample Analysis (M1_WT)

### Dataset
- **Sample:** M1_WT (wild-type mouse kidney)
- **Conditions:** Air, Oxygen 1, Oxygen 2
- **Modalities:** T2*, R2*, Perfusion
- **MLCO Layers:** 24 per kidney

### Expected Outputs

#### 1. MLCO Layer Profiles (`M1_WT_tlco_profiles.png`)
**What you see:**
- Three line plots showing layer-by-layer progression (1-48 total layers)
- Left kidney: Layers 1-24
- Right kidney: Layers 25-48
- Each condition (air, oxygen_1, oxygen_2) shown as different colored line
- Error bars show standard deviation within each layer

**Key observations:**
- T2* typically ranges 30-40 ms in cortex (outer layers)
- Slight increase toward medulla (inner layers)
- Oxygen conditions show slightly elevated T2* compared to air (normal response)
- R2* shows inverse pattern to T2*
- Perfusion highest in cortex, decreases toward medulla

#### 2. Perfusion Profile (`M1_WT_perfusion_profile.png`)
**What you see:**
- Perfusion values across all 48 layers
- Cortex (outer): ~200-250 relative units
- Medulla (inner): ~100-150 relative units
- Clear cortex-medulla gradient

#### 3. Triple Overlay Maps (per condition)
**Files:** `M1_WT_{condition}_triple_overlay.png`

**What you see:**
- Side-by-side maps of T2*, R2*, Perfusion
- Kidney anatomy clearly visible
- Cortex (outer rim) vs medulla (inner) differentiation
- Spatial correspondence between modalities

**Example - Air condition:**
- T2* map: Cortex appears slightly darker (lower T2*) than medulla
- R2* map: Inverse pattern
- Perfusion map: Cortex shows highest perfusion (brightest)

#### 4. Complete Analysis JSON (`M1_WT_complete_analysis.json`)

**Key sections:**

```json
{
  "sample_id": "M1_WT",
  "n_layers_per_organ": 24,
  "analysis_date": "2026-01-19T...",
  
  "conditions": {
    "air": {
      "bilateral": {
        "layers": [
          {
            "layer": 1,
            "depth_pct": 0.0,
            "t2star": {"mean": 32.5, "std": 4.2},
            "perfusion": {"mean": 245.0, "std": 35.0},
            "tissue_quality": {
              "viable_pct": 92.0,
              "suspect_edema_pct": 6.5,
              "likely_necrosis_pct": 1.5
            }
          }
          // ... layers 2-48
        ],
        
        "zones": {
          "cortex": {
            "t2star": {"mean": 31.2, "std": 3.8},
            "tissue_quality": {
              "viable_pct": 94.5,
              "interpretation": "healthy"
            }
          },
          "medulla": { ... }
        }
      }
    },
    "oxygen_1": { ... },
    "oxygen_2": { ... }
  },
  
  "oxygen_responsiveness": {
    "oxygen_1_vs_air": {
      "zones": {
        "cortex": {
          "delta_t2star": 2.5,
          "interpretation": "viable_responsive"
        }
      }
    }
  },
  
  "cortex_only_statistics": {
    "air": {
      "t2star": {"mean": 31.2, "std": 3.8},
      "tissue_quality": {
        "mean_viable_pct": 94.5,
        "interpretation": "healthy"
      }
    }
  }
}
```

---

## Example 2: Group Comparison (WT vs KO)

### Datasets
- **Group 1:** M1_WT (wild-type)
- **Group 2:** M2_KO (knockout model)
- **Conditions:** Air, Oxygen 1

### Expected Outputs

#### 1. Individual Sample Results
Same as Example 1, generated for both M1_WT and M2_KO

#### 2. Comparison Plots (`group1_vs_group2_{condition}_comparison.png`)

**What you see:**
- Side-by-side layer profiles for WT (blue) vs KO (orange)
- Comparison across all layers (1-48)
- Effect of knockout visible as separation between lines
- Statistical significance areas highlighted

**Example interpretation:**
- KO shows elevated T2* in medulla (layers 15-24)
- KO shows reduced perfusion in cortex (layers 1-10)
- WT maintains normal gradients, KO shows flattened gradients

#### 3. Comparison JSON (`group1_vs_group2_comparison.json`)

**Key sections:**

```json
{
  "group1_id": "M1_WT",
  "group2_id": "M2_KO",
  
  "cortex_comparison": {
    "air": {
      "t2star": {
        "group1_mean": 31.2,
        "group2_mean": 38.5,
        "delta": 7.3,
        "percent_change": 23.4,
        "effect_size": 1.8,
        "interpretation": "large_effect"
      },
      "perfusion": {
        "group1_mean": 245.0,
        "group2_mean": 180.0,
        "delta": -65.0,
        "percent_change": -26.5,
        "interpretation": "severely_reduced"
      }
    }
  },
  
  "full_organ_comparison": { ... },
  "tissue_quality_comparison": { ... }
}
```

**Interpretation:**
- KO shows 23% higher T2* in cortex → Suggests edema/tissue damage
- KO shows 26% reduced perfusion → Suggests ischemia
- Large effect size (1.8) → Biologically significant difference
- Tissue quality severely compromised in KO

---

## Example 3: Interpreting Results

### Healthy Kidney (M1_WT - Air condition)

**T2* Profile:**
- Cortex (layers 1-8): 30-32 ms ✓
- CMJ (layers 9-16): 32-35 ms ✓
- Medulla (layers 17-24): 35-38 ms ✓
- **Interpretation:** Normal cortex-medulla gradient

**Perfusion Profile:**
- Cortex: 240-250 relative units ✓
- Medulla: 120-140 relative units ✓
- **Interpretation:** Normal perfusion gradient

**Tissue Quality:**
- Cortex viable: 94.5% ✓
- Medulla viable: 88.0% ✓
- **Interpretation:** Healthy tissue throughout

**Oxygen Response:**
- ΔT2* air→oxygen: +2.5 ms ✓
- **Interpretation:** Normal vasodilatory response

---

### Diseased Kidney (M2_KO - Air condition)

**T2* Profile:**
- Cortex: 38-40 ms ⚠️ (elevated)
- Medulla: 50-55 ms 🔴 (very high)
- **Interpretation:** Widespread tissue damage, especially medulla

**Perfusion Profile:**
- Cortex: 180-190 relative units ⚠️ (reduced)
- Medulla: 80-100 relative units 🔴 (severely reduced)
- **Interpretation:** Ischemia throughout

**Tissue Quality:**
- Cortex viable: 65% ⚠️ (mild-moderate damage)
- Medulla viable: 35% 🔴 (severe damage)
- **Interpretation:** Substantial tissue compromise

**Oxygen Response:**
- ΔT2* air→oxygen: +0.5 ms 🔴 (blunted)
- **Interpretation:** Impaired vasodilatory capacity

---

## Common Patterns

### Pattern 1: Acute Kidney Injury
**Signature:**
- Elevated T2* (40-50 ms) throughout
- Preserved or slightly elevated perfusion
- Reduced viable tissue (70-80%)
- Flattened cortex-medulla gradient

**Example values:**
- Cortex T2*: 42 ms
- Medulla T2*: 45 ms
- Cortex perfusion: 220 units (normal-ish)
- Tissue viability: 75%

---

### Pattern 2: Chronic Kidney Disease
**Signature:**
- Moderately elevated T2* (35-45 ms)
- Reduced perfusion (150-180 units in cortex)
- Heterogeneous tissue quality
- Maintained gradient but shifted upward

**Example values:**
- Cortex T2*: 36 ms
- Medulla T2*: 42 ms
- Cortex perfusion: 165 units (reduced)
- Tissue viability: 80-85%

---

### Pattern 3: Ischemia-Reperfusion Injury
**Signature:**
- Very high T2* (>50 ms) in medulla
- Severely reduced perfusion (<100 units)
- Low tissue viability (<60%)
- Inverted or absent gradient

**Example values:**
- Cortex T2*: 45 ms
- Medulla T2*: 65 ms (necrotic range)
- Cortex perfusion: 120 units
- Medulla perfusion: 50 units
- Tissue viability: 45% (severe)

---

## Troubleshooting Examples

### Issue: Noisy Layer Profiles
**Symptom:** Large error bars, erratic layer-to-layer variations

**Possible causes:**
1. Motion artifacts
2. Insufficient SNR
3. Poor ROI drawing (including non-tissue pixels)

**Solution:**
- Redraw ROI more carefully
- Check reference images for motion
- Increase number of averages in acquisition

---

### Issue: Unrealistic T2* Values
**Symptom:** T2* > 100 ms or < 10 ms throughout

**Possible causes:**
1. Incorrect echo times
2. Poor fitting (hitting bounds)
3. Wrong data type (16-bit vs 32-bit)

**Solution:**
- Check prepare_data.py output for warnings
- Verify echo times in method file
- Use `--custom-t2star` for better fitting

---

### Issue: Perfusion Doesn't Match T2*
**Symptom:** High T2* but high perfusion (unexpected)

**Possible causes:**
1. Perfusion from different scan/time
2. Perfusion resolution mismatch
3. Acute injury (high flow, high T2*)

**Solution:**
- Verify perfusion scan corresponds to T2* scan
- Check metadata.json for scan parameters
- Consider biology - acute injury can show this pattern

---

## Expected File Structure

After complete analysis:

```
project/
├── prepared/
│   ├── M1_air_reference.npy
│   ├── M1_air_t2star_custom.npy
│   ├── M1_air_r2star_custom.npy
│   ├── M1_oxygen1_t2star_custom.npy
│   ├── M1_oxygen1_r2star_custom.npy
│   ├── M1_perfusion.npy
│   └── M1_metadata.json
│
├── mlco/
│   ├── M1_mlco_layers.npy
│   └── M1_visualization.png
│
├── configs/
│   ├── m1_wt_config.json
│   └── m2_ko_config.json
│
└── results/
    ├── M1_WT/
    │   ├── M1_WT_complete_analysis.json
    │   ├── M1_WT_tlco_profiles.png
    │   ├── M1_WT_perfusion_profile.png
    │   └── M1_WT_air_triple_overlay.png
    │
    └── comparison/
        ├── group1_vs_group2_comparison.json
        └── group1_vs_group2_air_comparison.png
```

---

## Next Steps

After reviewing these examples:

1. ✅ Understand the expected output formats
2. ✅ Know how to interpret layer profiles
3. ✅ Recognize healthy vs diseased patterns
4. ✅ Troubleshoot common issues

Ready to analyze your own data! See [User Guide](../user-guide.md) for step-by-step instructions.

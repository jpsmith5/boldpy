# Tissue Viability Threshold Configurations

This directory contains threshold definition files for tissue viability classification in BOLD MRI analysis.

## Overview

Threshold configurations define how BoldPy classifies tissue based on T2* and perfusion values. Each configuration specifies:
- Species and organ type
- Tissue classification categories (viable, suspect edema, likely necrosis)
- T2* thresholds (region-specific)
- Perfusion thresholds
- High-confidence classification rules

## Included Configurations

### Default Configuration
- **`kidney_mouse_default.yaml`** - Mouse kidney (reference implementation)
  - Based on published literature (PMID: 20829708)
  - Viable: T2* 5-40 ms
  - Suspect edema: T2* 40-60 ms
  - Likely necrosis: T2* >60 ms

### Example Configurations
- **`kidney_human_example.yaml`** - Human kidney example
  - Illustrative values for 3T imaging
  - **Must be adapted** based on your specific protocol
  - Includes guidance on clinical considerations

## Tissue Classification Categories

### 1. Viable Tissue
- Normal, healthy tissue
- T2* values in expected physiological range
- Normal perfusion levels

### 2. Suspect Edema
- Elevated T2* (increased water content)
- May indicate fluid accumulation
- Intermediate perfusion

### 3. Likely Necrosis
- Very high T2* values
- Often combined with low perfusion
- Indicates tissue damage or cell death

## Region-Specific Thresholds

Different kidney regions have different baseline T2* values:
- **Cortex**: Typically higher perfusion, intermediate T2*
- **CMJ**: Transition zone, intermediate values
- **Medulla**: Normally lower perfusion, can have lower T2*

Thresholds can be customized per region to account for these physiological differences.

## Creating Custom Threshold Configurations

### Basic Structure

```yaml
metadata:
  species: mouse  # or human, rat, etc.
  organ: kidney
  reference: "PMID or citation"
  description: "Brief description of source/context"

thresholds:
  viable:
    cortex_t2star: [min, max]     # ms
    cmj_t2star: [min, max]        # ms
    medulla_t2star: [min, max]    # ms
    perfusion: [min, max]         # ml/100g/min
  
  suspect_edema:
    cortex_t2star: [min, max]
    cmj_t2star: [min, max]
    medulla_t2star: [min, max]
    perfusion: [min, max]
  
  likely_necrosis:
    cortex_t2star: [min, max]
    cmj_t2star: [min, max]
    medulla_t2star: [min, max]
    perfusion: [min, max]

high_confidence_rules:  # Optional
  necrosis:
    t2star_min: 60
    perfusion_max: 50
```

## Important Considerations

### Species Differences
- **Mouse** (7T-11T): Higher field strength → different T2* values
- **Human** (1.5T-3T): Lower field strength → different ranges
- **Rat**: Intermediate, depends on field strength

### Field Strength Effects
- Higher field → shorter T2* values overall
- Different tissue contrast
- Adjust thresholds accordingly

### Sequence Parameters
- Echo times (TE)
- Multi-echo vs single-echo
- Spatial resolution
- All affect measured T2* values

### Physiological Factors
- **Age**: Affects baseline perfusion and T2*
- **Disease state**: May alter "normal" ranges
- **Medications**: Can affect perfusion
- **Hydration status**: Influences T2*

## Validating Your Thresholds

Before finalizing custom thresholds:

1. **Review literature** for your species/field strength/organ
2. **Analyze healthy controls** to establish baseline ranges
3. **Compare with published values** from similar protocols
4. **Validate classifications** against histology or other gold standards
5. **Consult experts** in your imaging modality and organ system

## Usage in BoldPy

```bash
python boldpy_analyze.py \
    --config sample.json \
    --n-layers 24 \
    --zone-config configs/zones/kidney_24layer.yaml \
    --threshold-config configs/thresholds/kidney_mouse_default.yaml \
    --output-dir results/
```

## Default Behavior

If no threshold config is specified, BoldPy will use `kidney_mouse_default.yaml`.

## Common Threshold Ranges (Reference)

### Mouse Kidney (7T-11T)
- **Normal cortex**: 25-40 ms
- **Normal medulla**: 20-35 ms
- **Edema**: >40 ms
- **Necrosis**: >60 ms

### Human Kidney (3T) - Approximate
- **Normal cortex**: 30-50 ms (varies widely by protocol)
- **Normal medulla**: 25-45 ms
- **Pathological**: >50-70 ms

**Important**: These are rough guidelines. Always validate with your specific imaging protocol.

## Example: Adapting for Rat at 7T

```yaml
metadata:
  species: rat
  organ: kidney
  reference: "Your pilot data"
  description: "Rat kidney at 7T, multi-echo sequence"

thresholds:
  viable:
    cortex_t2star: [10, 35]     # Adjust based on your data
    cmj_t2star: [10, 35]
    medulla_t2star: [10, 30]    # Typically lower
    perfusion: [180, 450]       # Rat-specific perfusion
```

## Questions?

For questions about threshold configurations:
- Check documentation: `docs/threshold-configuration.md`
- Review literature for your imaging protocol
- Analyze pilot data to establish ranges
- Consult with imaging experts

## Warning

⚠️ **Do not blindly use default thresholds for different species, field strengths, or protocols!**

Thresholds are highly specific to:
- Species
- Field strength (1.5T, 3T, 7T, 11T, etc.)
- Sequence parameters
- Imaging protocol
- Clinical/research context

Always validate thresholds with appropriate controls and literature references.

# Zone Configuration Files

Zone configurations define anatomical regions and subzones for MLCO-based BOLD MRI analysis.

## 📋 **Available Configurations**

### **Single-Region Configs**

Standard configs for single MLCO objects:

- **`kidney_24layer.yaml`** ⭐ Reference implementation
  - 24 layers total, 5 zones
  - Mouse kidney at 7-11T
  
- **`kidney_12layer.yaml`** - Lower resolution
  - 12 layers total, 3 zones
  - Faster analysis

- **`heart_12layer.yaml`** - Example template
  - 12 layers from epicardium to endocardium

### **Multi-Region Configs** 🆕

For independently analyzed anatomical regions:

- **`kidney_multiregion_8_8_6.yaml`** - 3 regions
  - Cortex: 8 layers (1001-1008)
  - Medulla: 8 layers (2001-2008)
  - Papilla: 6 layers (3001-3006)
  
- **`kidney_multiregion_12_12.yaml`** - 2 regions
  - Cortex: 12 layers (1001-1012)
  - Medulla: 12 layers (2001-2012)

---

## 🔧 **Configuration Formats**

### **Single-Region** (Traditional)

```yaml
metadata:
  organ: kidney
  n_layers: 24
  
zones:
  zone_name:
    layers: [1, 2, 3, 4, 5]  # Simple numbers
    percentage: 20
```

### **Multi-Region** (Phase 3) 🆕

```yaml
metadata:
  mode: multi_region
  
regions:
  cortex:
    region_id: 1
    n_layers: 8
    zones:
      outer:
        layers: [1001, 1002, 1003, 1004]  # Encoded!
```

**Encoding:** `region_id * 1000 + layer_num`  
**Example:** Region 2, Layer 5 → `2005`

---

## ✅ **Validation Rules**

**All configs:**
- No overlapping layers
- No missing layers
- Percentages sum to ~100%

**Single-region:**
- Layers: 1 to n_layers
- All consecutive

**Multi-region:**
- Encoded values: `region_id * 1000 + layer_num`
- Per-region validation
- No cross-region overlaps

---

## 🎯 **When to Use Each**

### **Single-Region**
- One homogeneous organ
- Bilateral treated as replicates
- Zones defined within MLCO layers

**Example:** Both kidneys, analyzing cortex vs medulla zones

### **Multi-Region** 🆕
- Manually outlined distinct regions
- Independent MLCO per region
- Separate statistics per region

**Example:** Cortex/medulla/papilla drawn separately, each with independent layers

---

## 📝 **Creating Custom Configs**

### **Single-Region Template**
```yaml
metadata:
  organ: liver
  n_layers: 16
  
zones:
  peripheral:
    layers: [1, 2, 3, 4, 5, 6, 7, 8]
    percentage: 50
  central:
    layers: [9, 10, 11, 12, 13, 14, 15, 16]
    percentage: 50
```

### **Multi-Region Template** 🆕
```yaml
metadata:
  mode: multi_region
  
regions:
  region1:
    region_id: 1
    n_layers: 8
    zones:
      outer:
        layers: [1001, 1002, 1003, 1004]
      inner:
        layers: [1005, 1006, 1007, 1008]
```

---

## 🧪 **Testing**

```bash
# Test single-region
python tissue_zones.py --zone-config configs/zones/kidney_24layer.yaml

# Test multi-region
python tissue_zones.py --zone-config configs/zones/kidney_multiregion_8_8_6.yaml
```

---

## 💡 **Tips**

- **Single-region:** 12-24 layers typical
- **Multi-region:** More layers for larger regions
- **Percentages:** Should sum to 100%
- **Encoding:** Use formula for multi-region: `region_id * 1000 + layer_num`

---

## 📚 **Related Docs**

- Multi-region ROI: `docs/unified_roi_drawer_guide.md`
- Multi-region MLCO: `docs/phase2_guide.md`
- Phase 3 guide: `docs/phase3_guide.md`
- Thresholds: `configs/thresholds/README.md`

---

**Need help?** Validation will catch errors automatically!

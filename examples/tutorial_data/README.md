# Tutorial Data - Run Your First Analysis in 30 Seconds!

This directory contains pre-processed example data you can analyze **immediately** without any setup.

---

## 🚀 Quick Start (30 seconds)

```bash
# Navigate to tutorial directory
cd examples/tutorial_data/

# Run analysis (works out of the box!)
python ../../boldpy_analyze.py \
    --config sample_config.json \
    --n-layers 12 \
    --output-dir tutorial_results/
```

**That's it!** Your results are in `tutorial_results/`

---

## 📊 What You'll Get

The analysis will create **19 output files**:

```
tutorial_results/
├── tutorial_sample_complete_analysis.json              # All metrics (JSON)
│
├── tutorial_sample_tlco_profiles.png/svg/pdf          # Layer-by-layer profiles
├── tutorial_sample_perfusion_profile.png/svg/pdf      # Perfusion across layers
│
├── tutorial_sample_air_t2star_perfusion_scatter.png/svg/pdf        # Scatter (air)
├── tutorial_sample_oxygen1_t2star_perfusion_scatter.png/svg/pdf    # Scatter (oxygen)
│
├── tutorial_sample_air_triple_overlay.png/svg/pdf                  # Triple overlay (air)
└── tutorial_sample_oxygen1_triple_overlay.png/svg/pdf              # Triple overlay (oxygen)
```

**All formats:** PNG (for viewing), SVG (vector graphics), PDF (publication-ready)

### Expected Results:
- **T2* profiles:** Cortex ~32ms → Medulla ~36ms (normal gradient)
- **Oxygen response:** +2-3ms increase from air to oxygen (healthy response)
- **Perfusion:** Cortex ~250 → Medulla ~150 (normal gradient)
- **Tissue quality:** >90% viable (healthy kidney)

---

## 📁 What's in This Tutorial?

### Pre-Processed Maps:
```
prepared_maps/
├── sample_air_t2star_custom.npy       # T2* map - air breathing
├── sample_air_r2star_custom.npy       # R2* map - air breathing
├── sample_oxygen1_t2star_custom.npy   # T2* map - oxygen breathing
├── sample_oxygen1_r2star_custom.npy   # R2* map - oxygen breathing
└── sample_perfusion.npy               # Perfusion map
```

### MLCO Mask:
```
sample_mlco_layers.npy                 # 12-layer concentric mask
```

### Configuration:
```
sample_config.json                     # Points to all the maps above
```

---

## 🧪 Tutorial Data Details

- **Sample:** Synthetic healthy kidney (realistic values)
- **Size:** 150×150 pixels, one kidney
- **Layers:** 12 (cortex → medulla)
- **Conditions:** Air breathing + Oxygen breathing
- **Modalities:** T2*, R2*, Perfusion

**Values based on:** Mouse kidney literature (PMID: 20829708)

---

## 📖 Next Steps

After running this tutorial:

1. ✅ **Explore the results**
   - Open the `.png` plots
   - Check the `.json` file for metrics

2. ✅ **Learn the full workflow**
   - See `../../docs/quick-start.md` for processing your own data
   - Read `../../docs/user-guide.md` for comprehensive guide

3. ✅ **Understand the metrics**
   - Check `../../docs/metrics-documentation.md`
   - Review `../../docs/examples-with-data.md`

4. ✅ **Process your own data**
   ```bash
   # Step 0: Extract T2*/R2* from your scans
   python ../../prepare_data.py --input your_scan.PvDatasets --both-t2star
   
   # Step 1: Draw ROI
   python ../../roi_drawer.py --image prepared/scan_reference.npy --output roi.npy
   
   # Step 2: Generate MLCO
   python ../../generate_mlco.py --mask roi.npy --split --n-layers 12
   
   # Step 3: Create config
   # (pointing to your processed maps)
   
   # Step 4: Analyze
   python ../../boldpy_analyze.py --config your_config.json --n-layers 12 --output-dir results/
   ```

---

## 🔍 Troubleshooting

### "Config missing required fields"
- Make sure you're running from the `tutorial_data/` directory
- The config uses relative paths that work from this location

### "File not found"
- Run `ls prepared_maps/` to verify files exist
- Check you extracted the full package

### Analysis runs but no plots?
- Check `tutorial_results/` directory
- Look for error messages in console

---

## ✨ Why This Tutorial?

**Traditional "quick start":**
```bash
# Step 1: Get raw data
# Step 2: Process data (30 min)
# Step 3: Draw ROIs (10 min)
# Step 4: Generate layers (5 min)
# Step 5: Create config
# Step 6: Finally analyze!
```

**This tutorial:**
```bash
cd tutorial_data/
python ../../boldpy_analyze.py --config sample_config.json --n-layers 12 --output-dir results/
# Done in 30 seconds!
```

**Learn by doing** - See real results immediately, then learn the workflow!

---

## 📚 Additional Resources

- **User Guide:** `../../docs/user-guide.md`
- **Metrics Reference:** `../../docs/metrics-documentation.md`
- **Script Reference:** `../../docs/scripts-reference.md`
- **Real Data Examples:** `../../docs/examples-with-data.md`

---

**Ready to analyze? Run the command above and see BoldPy in action!** 🚀

# Changelog

All notable changes to BoldPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.1] - 2026-01-26

### Added - Enhanced T2* Detection & Visualization

#### **Tiered T2* Frame Detection (prepare_data.py)**
- **Intelligent frame identification:** Three-tier approach for robust T2* frame detection
  - **Tier 1:** Metadata parsing (`VisuCoreFrameType` from Bruker visu_pars)
  - **Tier 2:** Enhanced multi-factor scoring heuristic (100-point system)
  - **Tier 3:** Manual override via `--t2-frame N` option
- **Multi-factor scoring criteria:**
  - Mean value (40 pts): Optimal range 10-30 ms for tissue
  - Max value (20 pts): Should be <100 ms to exclude fluid
  - visu_pars range (15 pts): Validates against metadata
  - Distribution shape (15 pts): Ensures reasonable spread
  - Pixel coverage (10 pts): Requires >10% non-zero pixels
- **Confidence reporting:** HIGH (70-100 pts), MEDIUM (50-69 pts), LOW (<50 pts)
- **New CLI option:** `--t2-frame N` for manual frame specification (1-indexed)
- **Detailed diagnostics:** Frame-by-frame scoring breakdown with recommendations

#### **Continuous Whole-Kidney Plotting (boldpy_plots.py)**
- **New function:** `plot_whole_kidney_continuous()` - Single sample visualization
  - Cortex → Medulla → Papilla as one continuous profile
  - 2-3 panels: T2*, R2*, Perfusion (if available)
  - Region boundaries with color-coded shading
  - Configurable tissue viability threshold lines
  - Whole-kidney gradient calculation
- **New function:** `plot_whole_kidney_comparison()` - Group comparison
  - Overlaid profiles (WT vs KO) for direct visual comparison
  - 2 panels: T2* comparison, Perfusion comparison
  - Group 1: Blue solid line with circles
  - Group 2: Orange dashed line with squares
  - Threshold reference lines for context
- **Configurable thresholds:** `DEFAULT_THRESHOLDS` dictionary
  - `t2star_fluid`: 40.0 ms (above = likely fluid/degraded)
  - `t2star_ischemic`: 8.0 ms (below = severe ischemia)
  - `perfusion_cortex_min`: 250.0 ml/100g/min
  - `perfusion_medulla_min`: 100.0 ml/100g/min
  - `perfusion_severe`: 50.0 ml/100g/min
  - `gradient_abnormal`: 40.0 ms (|gradient| threshold)
- **Automated workflow integration:** `boldpy_analyze.py` now generates continuous plots automatically
  - Single sample: Generates continuous profile for each condition
  - Group comparison: Generates N continuous comparisons (one per condition)

### Fixed - Robustness & Data Quality

#### **Missing Layer Handling**
- **Problem:** Array size mismatches when MLCO layers have 0 pixels
- **Solution:** Fill missing layers with NaN values, display as natural gaps
- **Benefits:**
  - No more crashes on incomplete data
  - Gaps show where tissue wasn't captured (informative!)
  - Works automatically for all samples
- **Enhanced diagnostics:** Reports which layers are missing and why

#### **Layer Number Inference**
- **Problem:** Averaged bilateral data doesn't have `layer_number` field (KeyError)
- **Solution:** Infer layer numbers from position in list
- **Implementation:** `layer.get('layer_number', idx)` - uses index as fallback
- **Backwards compatible:** Works with both raw and averaged data

#### **Integer X-Axis Labels**
- **Problem:** Layer axes sometimes showed fractional labels (2.5, 3.0, etc.)
- **Solution:** Force integer-only ticks using `MaxNLocator(integer=True)`
- **Applied to:** All layer profile plots (5 functions, 8 axes total)
- **Result:** Clean, professional plots with only whole numbers

### Changed - Visualization Improvements

#### **Perfusion Integration**
- **Complete integration:** All Phase 2 comparison plots now include perfusion
- **Automatic upsampling:** Perfusion maps (80×80) upsampled to match T2*/R2* (200×200)
- **Method:** Bilinear interpolation via `scipy.ndimage.zoom`
- **4-panel layout:** T2*, R2*, Perfusion, Gradient (when perfusion available)
- **Backward compatible:** Falls back to 3 panels if perfusion unavailable

#### **Module Consolidation**
- **Before:** 4 separate plotting files (3,277 lines total)
- **After:** 1 unified `boldpy_plots.py` (2,820 lines)
- **Savings:** 532 lines eliminated (16% reduction)
- **Organization:** 4 logical sections with 18 total functions
  - Whole-kidney functions (5)
  - Phase 1 multiregion (6)
  - Phase 2 multiregion (5)
  - Phase 2 continuous (2 NEW!)
- **Benefits:** Single import location, no code duplication, easier maintenance

### Improved - Code Quality & Usability

#### **Error Handling**
- **Per-region validation:** Identifies which specific region has issues
- **Clear error messages:** Explains what went wrong and how to fix it
- **Graceful degradation:** Continues with available data when possible
- **User guidance:** Suggests manual override when auto-detection uncertain

#### **Diagnostic Output**
- **Frame detection:**
  ```
  Tier 1: Checking Bruker metadata...
    ⚠️  Metadata uninformative (frames labeled REAL_IMAGE)
  Tier 2: Using enhanced scoring heuristic...
       ★ Frame 3: 95/100 pts
           Mean 12.5 ms: 40/40 pts
           Max 89.1 ms: 20/20 pts
    ✓ Selected frame 3 (confidence: HIGH)
  ```
- **Missing layers:**
  ```
  cortex: expected=6, actual=5 ⚠ missing 1 layer(s)
    Note: 1 layer(s) missing (will be padded with NaN)
  ```

### Documentation
- **New guides:**
  - `MASTER_SESSION_SUMMARY.md` - Complete development session tracking
  - `ENHANCED_BRUKER_FRAME_DETECTION.md` - Tiered detection system
  - `INTEGER_X_AXIS_FORMATTING.md` - Visualization improvements
  - `FIX_MISSING_LAYER_NUMBER_FIELD.md` - Robustness fixes
  - `PREPARE_DATA_INTEGRATION_SUMMARY.md` - prepare_data.py enhancements
- **Updated guides:**
  - `COMPARISON_PLOTTING_ADDITION.md` - Group comparison features
  - `PERFUSION_INTEGRATION_COMPLETE.md` - Perfusion in all plots
  - `CONTINUOUS_PROFILES_USAGE_GUIDE.md` - Whole-kidney visualization
  - `BOLD_MRI_INTERPRETATION_GUIDE.md` - Updated with new features

### Technical Details
- **Total additions:** ~800 lines of new functionality
- **Bug fixes:** 5 major issues resolved
- **Functions added:** 4 (tiered detection) + 2 (continuous plotting)
- **Functions enhanced:** 8 (perfusion, missing layers, integer axes)
- **Backwards compatibility:** 100% maintained
- **Testing:** All features validated with real experimental data (M1/M2 samples)

---

## [2.2.0] - 2026-01-20

### Added - Configuration System
- **YAML-based configurations:** Introduced modular zone and threshold configuration system
- **Zone configs** (`configs/zones/`):
  - `kidney_24layer.yaml` - Reference implementation (default)
  - `kidney_12layer.yaml` - Lower resolution alternative
  - `heart_12layer.yaml` - Example for users to adapt
  - Comprehensive README explaining when MLCO analysis is appropriate
- **Threshold configs** (`configs/thresholds/`):
  - `kidney_mouse_default.yaml` - Mouse kidney thresholds (default)
  - `kidney_human_example.yaml` - Template for human applications
  - Comprehensive README with guidance on species/field strength adaptation
- **CLI arguments:**
  - `--zone-config` - Specify custom zone configuration YAML
  - `--threshold-config` - Specify custom threshold configuration YAML
- **Dynamic config loading:** Configs can be specified per-analysis without modifying code

### Changed
- **Scope clarification:** Explicitly documented that BoldPy is for MLCO-amenable organs
- **Kidney-focused positioning:** Framework optimized for kidneys and organs with concentric architecture
- **Module structure:** Updated `tissue_zones.py` and `mlco_analysis.py` to dynamically load configs
- **User customization:** Thresholds and zone definitions now easily adjustable without code changes

### Improved
- **Organ applicability documentation:** Clear guidance on when MLCO analysis is appropriate:
  - ✅ Kidney (cortex → medulla → papilla)
  - ✅ Heart (epicardium → endocardium)
  - ✅ Organs with concentric architecture
  - ❌ Brain (requires 3D parcellation, not concentric)
  - ❌ Lung (lobar/airway architecture, not concentric)
- **Configuration validation:** Automatic checking of zone coverage and threshold structure
- **Example configurations:** Multiple templates for users to adapt to their specific needs

### Documentation
- New documentation files:
  - `configs/zones/README.md` - Creating custom zone configurations
  - `configs/thresholds/README.md` - Customizing tissue viability thresholds
- Updated existing documentation to reflect configuration system
- Added examples showing how to use custom configs

### Notes
- **Backwards compatible:** Default configs match previous hardcoded values
- **No breaking changes:** Existing scripts work without modification
- **Future-ready:** System designed for easy extension to other organs with concentric architecture

---

## [2.1.1] - 2026-01-19

### Fixed
- **Critical:** Fixed `n_layers_per_organ` parameter name in `boldpy_analyze.py` (was incorrectly `n_layers_per_kidney`)
- Removed unused BoldPy imports that could cause import failures
- Improved perfusion resampling to use bilinear interpolation (order=1) instead of nearest neighbor

### Changed
- Perfusion resampling warning message now suggests running prepare_data.py with proper resolution
- Cleaned up import statements to remove dependencies on unused functions

### Documentation
- Added comprehensive tissue quality assessment documentation
- Created quick reference guides for metrics
- Updated all documentation to reflect config-based workflow

---

## [2.1.0] - 2026-01-15

### Major Refactoring - Config-Based Workflow

#### Changed - Breaking
- **boldpy_analyze.py:** Complete rewrite to use pre-computed `.npy` files instead of PvDatasets
- **Config format:** New structure using `t2star_maps`, `r2star_maps`, `perfusion_map` instead of `bold_scans`
- **Removed `--source` argument:** No longer choose between `bruker` and `nonlinear_fit` at analysis time
- **Function renaming:** More generic, modular naming
  - `load_animal_data()` → `load_data()`
  - `analyze_animal()` → `analyze_sample()`
  - `compare_animals()` → `compare_groups()`
  - `animal_*` variables → `sample_*` or `data`
  - `wt_config/ko_config` → `group1_config/group2_config`

#### Added - boldpy_analyze.py
- Comprehensive config validation with clear error messages
- File existence checking for all referenced maps
- Automatic shape validation between T2* and R2* maps
- Condition matching validation
- Support for optional perfusion maps

#### Removed - boldpy_analyze.py
- PvDatasets loading functions (`load_bruker_maps()`, `load_and_fit_our_maps()`)
- Echo time extraction functions (no longer needed)
- Command-line mode with `--animal-id` and scan arguments
- All Bruker-specific data extraction code (~288 lines removed)

### Added - prepare_data.py

#### T2* Fitting Improvements
- **100% pixel fitting success rate** achieved
- Correct T2* bounds: [5, 2000] ms (previously incorrect [1, 100] ms)
- Fixed Python list vs numpy array conversion for scipy.curve_fit
- Smart frame detection for Bruker T2* maps (auto-detects Frame 3)
- Multi-frame pdata/2 handling with automatic T2* frame identification

#### Perfusion Extraction
- **Bruker perfusion extraction** from pdata/2, Frame 5
- Automatic resampling to match T2* resolution
- Validates perfusion values (0-200% relative range)
- Stores perfusion statistics in metadata.json
- Works in both single-scan and batch modes

#### Batch Processing
- Process entire directories of PvDatasets files
- Pattern matching with `--pattern` argument
- Automatic sample naming from filenames
- Comprehensive error handling per scan

#### Output Options
- 2D slice extraction (default)
- Full 3D volume support with `--3d`
- Specific slice selection with `--slice N`
- Both Bruker and custom T2* with `--both-t2star`

### Added - generate_mlco.py

#### Generic Bilateral Organ Support
- **Completely redesigned** for any bilateral organ (not just kidneys)
- New `split_mask()` function replaces kidney-specific code
- MRI left-right flip convention with configurable `--no-mri-flip`
- Custom component naming with `--component-names`
- Minimum component size filtering with `--min-component-size`

#### Simplified Arguments
- `--mask` for single mask input (replaces `--bilateral`, `--left`, `--right`)
- `--split` flag to indicate bilateral organ
- `--component-names` for custom naming (default: left, right)
- Works for single organs (without `--split`) or bilateral

#### Examples
- Bilateral kidneys with MRI convention
- Single organ analysis
- Custom component names (medial/lateral, lung1/lung2, etc.)
- Non-MRI data with `--no-mri-flip`

### Documentation

#### Added
- Complete quick-start guide with all 4 steps
- New config format documentation with examples
- Updated workflow examples (prepare → ROI → MLCO → analyze)
- MRI flip convention explained
- Perfusion extraction guide

#### Updated
- Installation instructions
- Example configs with all conditions (air, oxygen_1, oxygen_2)
- Removed `--source` references throughout
- Updated to `--group1-config` / `--group2-config` naming

---

## Version Numbering

- **Major (X.0.0):** Breaking changes, major new features
- **Minor (0.X.0):** New features, non-breaking changes
- **Patch (0.0.X):** Bug fixes, documentation updates

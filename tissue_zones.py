#!/usr/bin/env python3
"""
Tissue Zone Definitions and Masking
====================================

Configurable zone and threshold system for MLCO-based BOLD MRI analysis.

Supports custom zone definitions and tissue viability thresholds via YAML configs.
Default configurations provided for kidney analysis (24-layer).

Zone definitions specify anatomical regions (e.g., cortex, medulla)
Threshold definitions specify tissue quality criteria (viable, edema, necrosis)
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# ============================================================================
# MULTI-REGION ENCODING/DECODING
# ============================================================================

def decode_multiregion_value(encoded_value: int) -> Tuple[int, int]:
    """
    Decode multi-region MLCO encoded value
    
    Parameters:
    -----------
    encoded_value : int
        Encoded value (region_id * 1000 + layer_num)
        
    Returns:
    --------
    region_id : int
        Region ID (1, 2, 3, ...)
    layer_num : int
        Layer number within region (1, 2, 3, ...)
    """
    region_id = encoded_value // 1000
    layer_num = encoded_value % 1000
    return region_id, layer_num


def encode_multiregion_value(region_id: int, layer_num: int) -> int:
    """
    Encode multi-region MLCO value
    
    Parameters:
    -----------
    region_id : int
        Region ID (1, 2, 3, ...)
    layer_num : int
        Layer number within region (1, 2, 3, ...)
        
    Returns:
    --------
    encoded_value : int
        Encoded value (region_id * 1000 + layer_num)
    """
    return region_id * 1000 + layer_num


def is_multiregion_config(config: Dict) -> bool:
    """
    Check if zone config is multi-region format
    
    Parameters:
    -----------
    config : dict
        Zone configuration
        
    Returns:
    --------
    is_multiregion : bool
        True if config has multi-region format
    """
    # Check for multi-region indicators
    if 'regions' in config:
        return True
    if 'metadata' in config and config['metadata'].get('mode') == 'multi_region':
        return True
    return False


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_zone_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load zone configuration from YAML file
    
    Parameters:
    -----------
    config_path : Path, optional
        Path to zone configuration YAML file.
        If None, loads default kidney_24layer.yaml
    
    Returns:
    --------
    config : dict
        Zone configuration with 'metadata', 'zones', and 'aggregate_zones'
    """
    if config_path is None:
        # Use default kidney configuration
        script_dir = Path(__file__).parent
        config_path = script_dir / 'configs' / 'zones' / 'kidney_24layer.yaml'
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Zone config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_zone_config(config)
    
    return config


def load_threshold_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load tissue viability threshold configuration from YAML file
    
    Parameters:
    -----------
    config_path : Path, optional
        Path to threshold configuration YAML file.
        If None, loads default kidney_mouse_default.yaml
    
    Returns:
    --------
    config : dict
        Threshold configuration with 'metadata' and 'thresholds'
    """
    if config_path is None:
        # Use default kidney configuration
        script_dir = Path(__file__).parent
        config_path = script_dir / 'configs' / 'thresholds' / 'kidney_mouse_default.yaml'
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Threshold config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_threshold_config(config)
    
    return config


def validate_zone_config(config: Dict) -> None:
    """
    Validate zone configuration structure
    
    Supports both single-region and multi-region formats.
    
    Single-region checks:
    - Required keys present
    - Layer numbers valid (positive integers)
    - No gaps or overlaps in layer assignments
    - Percentages sum to 100 (approximately)
    
    Multi-region checks:
    - Each region has valid structure
    - Encoded layer values match region_id * 1000 + layer_num
    - No overlaps within or across regions
    """
    # Check required keys
    if 'metadata' not in config:
        raise ValueError("Zone config missing required key: metadata")
    
    # Detect format
    if is_multiregion_config(config):
        _validate_multiregion_config(config)
    else:
        _validate_singleregion_config(config)


def _validate_singleregion_config(config: Dict) -> None:
    """Validate single-region zone configuration"""
    if 'zones' not in config:
        raise ValueError("Zone config missing required key: zones")
    
    if 'n_layers' not in config['metadata']:
        raise ValueError("Zone config metadata must specify n_layers")
    
    n_layers = config['metadata']['n_layers']
    
    # Validate each zone
    all_layers = set()
    total_percentage = 0
    
    for zone_name, zone_info in config['zones'].items():
        if 'layers' not in zone_info:
            raise ValueError(f"Zone '{zone_name}' missing 'layers' key")
        
        layers = zone_info['layers']
        
        # Check for valid layer numbers
        for layer in layers:
            if not isinstance(layer, int) or layer < 1 or layer > n_layers:
                raise ValueError(f"Invalid layer number in zone '{zone_name}': {layer}")
            
            if layer in all_layers:
                raise ValueError(f"Layer {layer} assigned to multiple zones")
            
            all_layers.add(layer)
        
        # Check percentage if provided
        if 'percentage' in zone_info:
            total_percentage += zone_info['percentage']
    
    # Check that all layers are assigned
    expected_layers = set(range(1, n_layers + 1))
    if all_layers != expected_layers:
        missing = expected_layers - all_layers
        extra = all_layers - expected_layers
        if missing:
            raise ValueError(f"Layers not assigned to any zone: {sorted(missing)}")
        if extra:
            raise ValueError(f"Invalid layer numbers: {sorted(extra)}")
    
    # Check percentages sum to 100 (with 1% tolerance)
    if total_percentage > 0 and abs(total_percentage - 100) > 1:
        raise ValueError(f"Zone percentages sum to {total_percentage}, should be ~100")


def _validate_multiregion_config(config: Dict) -> None:
    """Validate multi-region zone configuration"""
    if 'regions' not in config:
        raise ValueError("Multi-region config missing 'regions' key")
    
    all_encoded_layers = set()
    
    for region_name, region_config in config['regions'].items():
        # Check required keys
        if 'region_id' not in region_config:
            raise ValueError(f"Region '{region_name}' missing 'region_id'")
        if 'n_layers' not in region_config:
            raise ValueError(f"Region '{region_name}' missing 'n_layers'")
        if 'zones' not in region_config:
            raise ValueError(f"Region '{region_name}' missing 'zones'")
        
        region_id = region_config['region_id']
        n_layers = region_config['n_layers']
        
        # Validate zones within this region
        region_layers = set()
        total_percentage = 0
        
        for zone_name, zone_info in region_config['zones'].items():
            if 'layers' not in zone_info:
                raise ValueError(f"Zone '{zone_name}' in region '{region_name}' missing 'layers'")
            
            layers = zone_info['layers']
            
            for encoded_layer in layers:
                # Check encoding is correct
                decoded_region, decoded_layer = decode_multiregion_value(encoded_layer)
                
                if decoded_region != region_id:
                    raise ValueError(f"Layer {encoded_layer} in region '{region_name}' "
                                   f"has wrong region encoding (expected region {region_id}, "
                                   f"got {decoded_region})")
                
                if decoded_layer < 1 or decoded_layer > n_layers:
                    raise ValueError(f"Layer {encoded_layer} in region '{region_name}' "
                                   f"has invalid layer number {decoded_layer} "
                                   f"(must be 1-{n_layers})")
                
                # Check for duplicates within region
                if encoded_layer in region_layers:
                    raise ValueError(f"Layer {encoded_layer} assigned to multiple zones "
                                   f"in region '{region_name}'")
                
                # Check for duplicates across regions
                if encoded_layer in all_encoded_layers:
                    raise ValueError(f"Layer {encoded_layer} assigned to multiple regions")
                
                region_layers.add(encoded_layer)
                all_encoded_layers.add(encoded_layer)
            
            # Check percentage if provided
            if 'percentage' in zone_info:
                total_percentage += zone_info['percentage']
        
        # Check that all layers in this region are assigned
        expected_encoded = set([encode_multiregion_value(region_id, i) 
                               for i in range(1, n_layers + 1)])
        if region_layers != expected_encoded:
            missing = expected_encoded - region_layers
            extra = region_layers - expected_encoded
            if missing:
                missing_layers = [decode_multiregion_value(l)[1] for l in sorted(missing)]
                raise ValueError(f"Region '{region_name}': layers not assigned to any zone: {missing_layers}")
            if extra:
                raise ValueError(f"Region '{region_name}': invalid layer numbers: {sorted(extra)}")
        
        # Check percentages sum to 100 (with 1% tolerance)
        if total_percentage > 0 and abs(total_percentage - 100) > 1:
            raise ValueError(f"Region '{region_name}': zone percentages sum to {total_percentage}, should be ~100")


def validate_threshold_config(config: Dict) -> None:
    """
    Validate threshold configuration structure
    
    Checks:
    - Required keys present
    - Threshold values valid (tuples of numbers)
    - Min < Max for all ranges
    """
    # Check required keys
    required_keys = ['metadata', 'thresholds']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Threshold config missing required key: {key}")
    
    # Validate each threshold category
    for category, thresholds in config['thresholds'].items():
        for threshold_name, value in thresholds.items():
            if threshold_name == 'description':
                continue
            
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Threshold '{threshold_name}' in '{category}' must be [min, max]")
            
            min_val, max_val = value
            if min_val >= max_val:
                raise ValueError(f"Threshold '{threshold_name}' in '{category}': min ({min_val}) >= max ({max_val})")


# ============================================================================
# LEGACY COMPATIBILITY - Convert configs to old format
# ============================================================================

def get_zone_definitions(zone_config: Optional[Dict] = None) -> Dict:
    """
    Get zone definitions in legacy format for backwards compatibility
    
    Parameters:
    -----------
    zone_config : dict, optional
        Loaded zone configuration. If None, loads default.
    
    Returns:
    --------
    zone_definitions : dict
        Dictionary mapping zone names to layer ranges
    """
    if zone_config is None:
        zone_config = load_zone_config()
    
    zone_defs = {}
    for zone_name, zone_info in zone_config['zones'].items():
        layers = zone_info['layers']
        zone_defs[zone_name] = {
            'layers': range(min(layers), max(layers) + 1),
            'description': zone_info.get('description', ''),
            'percentage': zone_info.get('percentage', 0)
        }
    
    return zone_defs


def get_aggregate_zones(zone_config: Optional[Dict] = None) -> Dict:
    """
    Get aggregate zones in legacy format
    
    Parameters:
    -----------
    zone_config : dict, optional
        Loaded zone configuration. If None, loads default.
    
    Returns:
    --------
    aggregate_zones : dict
        Dictionary mapping aggregate zone names to layer ranges
    """
    if zone_config is None:
        zone_config = load_zone_config()
    
    agg_zones = {}
    if 'aggregate_zones' in zone_config:
        for zone_name, zone_info in zone_config['aggregate_zones'].items():
            layers = zone_info['layers']
            agg_zones[zone_name] = range(min(layers), max(layers) + 1)
    
    return agg_zones


def get_tissue_thresholds(threshold_config: Optional[Dict] = None) -> Dict:
    """
    Get tissue thresholds in legacy format
    
    Parameters:
    -----------
    threshold_config : dict, optional
        Loaded threshold configuration. If None, loads default.
    
    Returns:
    --------
    thresholds : dict
        Dictionary with threshold values
    """
    if threshold_config is None:
        threshold_config = load_threshold_config()
    
    return threshold_config['thresholds']


# Default configs (loaded once at module import)
_DEFAULT_ZONE_CONFIG = None
_DEFAULT_THRESHOLD_CONFIG = None


def get_default_configs():
    """Load default configs once and cache"""
    global _DEFAULT_ZONE_CONFIG, _DEFAULT_THRESHOLD_CONFIG
    
    if _DEFAULT_ZONE_CONFIG is None:
        _DEFAULT_ZONE_CONFIG = load_zone_config()
    
    if _DEFAULT_THRESHOLD_CONFIG is None:
        _DEFAULT_THRESHOLD_CONFIG = load_threshold_config()
    
    return _DEFAULT_ZONE_CONFIG, _DEFAULT_THRESHOLD_CONFIG


# Load defaults for backwards compatibility
ZONE_CONFIG, THRESHOLD_CONFIG = get_default_configs()
ZONE_DEFINITIONS = get_zone_definitions(ZONE_CONFIG)
AGGREGATE_ZONES = get_aggregate_zones(ZONE_CONFIG)
TISSUE_THRESHOLDS = get_tissue_thresholds(THRESHOLD_CONFIG)


def update_configs(zone_config_path: Optional[Path] = None,
                   threshold_config_path: Optional[Path] = None) -> None:
    """
    Update module-level zone and threshold configurations
    
    Call this function to use custom configurations instead of defaults.
    This will update ZONE_CONFIG, THRESHOLD_CONFIG, ZONE_DEFINITIONS,
    AGGREGATE_ZONES, and TISSUE_THRESHOLDS module-level variables.
    
    Parameters:
    -----------
    zone_config_path : Path, optional
        Path to custom zone configuration YAML.
        If None, keeps current zone config.
    threshold_config_path : Path, optional
        Path to custom threshold configuration YAML.
        If None, keeps current threshold config.
    
    Example:
    --------
    >>> from tissue_zones import update_configs
    >>> update_configs(
    ...     zone_config_path='configs/zones/kidney_12layer.yaml',
    ...     threshold_config_path='configs/thresholds/kidney_human_example.yaml'
    ... )
    """
    global ZONE_CONFIG, THRESHOLD_CONFIG
    global ZONE_DEFINITIONS, AGGREGATE_ZONES, TISSUE_THRESHOLDS
    
    if zone_config_path is not None:
        ZONE_CONFIG = load_zone_config(zone_config_path)
        
        # Only update legacy definitions for single-region configs
        # Multi-region configs use a different structure accessed via helper functions
        if not is_multiregion_config(ZONE_CONFIG):
            ZONE_DEFINITIONS = get_zone_definitions(ZONE_CONFIG)
            AGGREGATE_ZONES = get_aggregate_zones(ZONE_CONFIG)
        else:
            # For multi-region, leave legacy definitions as is
            # Analysis will use get_region_zones() and similar functions instead
            print("  Note: Multi-region config loaded (legacy zone definitions not updated)")
    
    if threshold_config_path is not None:
        THRESHOLD_CONFIG = load_threshold_config(threshold_config_path)
        TISSUE_THRESHOLDS = get_tissue_thresholds(THRESHOLD_CONFIG)

# ============================================================================
# TISSUE CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_tissue_viability(t2star: float, 
                              perfusion: Optional[float] = None,
                              region: str = 'cortex',
                              threshold_config: Optional[Dict] = None) -> str:
    """
    Classify tissue viability based on T2* and perfusion
    
    Parameters:
    -----------
    t2star : float
        T2* value in ms
    perfusion : float, optional
        Perfusion in ml/100g/min
    region : str
        'cortex', 'cmj', or 'medulla'
    threshold_config : dict, optional
        Loaded threshold configuration. If None, uses default.
        
    Returns:
    --------
    classification : str
        'viable', 'suspect_edema', or 'likely_necrosis'
    """
    if threshold_config is None:
        threshold_config = THRESHOLD_CONFIG
    
    thresholds = threshold_config['thresholds']
    t2_key = f'{region}_t2star'
    
    # Check necrosis first (most severe)
    necrosis_range = thresholds['likely_necrosis'][t2_key]
    if t2star >= necrosis_range[0]:
        # High confidence if perfusion also low
        if 'combination_rules' in threshold_config:
            rules = threshold_config['combination_rules'].get('high_confidence_necrosis', {})
            if perfusion is not None and perfusion < rules.get('max_perfusion', 50):
                return 'likely_necrosis_high_conf'
        elif perfusion is not None and perfusion < 50:
            return 'likely_necrosis_high_conf'
        return 'likely_necrosis'
    
    # Check edema
    edema_range = thresholds['suspect_edema'][t2_key]
    if t2star >= edema_range[0]:
        return 'suspect_edema'
    
    # Viable tissue
    else:
        return 'viable'


def calculate_tissue_quality(t2_map: np.ndarray,
                            mask: np.ndarray,
                            perfusion_map: Optional[np.ndarray] = None,
                            region: str = 'cortex',
                            threshold_config: Optional[Dict] = None) -> Dict:
    """
    Calculate tissue quality statistics for a masked region
    
    Parameters:
    -----------
    t2_map : ndarray
        T2* map
    mask : ndarray
        Boolean mask for the region
    perfusion_map : ndarray, optional
        Perfusion map
    region : str
        'cortex', 'cmj', or 'medulla'
    threshold_config : dict, optional
        Loaded threshold configuration. If None, uses default.
        
    Returns:
    --------
    quality_stats : dict
        Percentage of pixels in each category
    """
    # Get T2* values in region
    t2_values = t2_map[mask]
    n_total = len(t2_values)
    
    if n_total == 0:
        return {
            'viable_pct': 0,
            'suspect_edema_pct': 0,
            'likely_necrosis_pct': 0,
            'n_pixels': 0
        }
    
    # Get perfusion if available
    perf_values = None
    if perfusion_map is not None:
        perf_values = perfusion_map[mask]
    
    # Classify each pixel
    classifications = []
    for i, t2 in enumerate(t2_values):
        perf = perf_values[i] if perf_values is not None else None
        classifications.append(
            classify_tissue_viability(t2, perf, region, threshold_config)
        )
    
    # Count each category
    n_viable = sum(1 for c in classifications if 'viable' in c)
    n_suspect = sum(1 for c in classifications if 'suspect' in c)
    n_necrosis = sum(1 for c in classifications if 'necrosis' in c)
    
    return {
        'viable_pct': (n_viable / n_total) * 100,
        'suspect_edema_pct': (n_suspect / n_total) * 100,
        'likely_necrosis_pct': (n_necrosis / n_total) * 100,
        'n_pixels': n_total,
        'tissue_quality_score': n_viable / n_total  # 0-1 score
    }


def get_zone_name(layer_num: int, zone_config: Optional[Dict] = None) -> str:
    """
    Get zone name for a given layer number
    
    Parameters:
    -----------
    layer_num : int
        Layer number (1-based)
    zone_config : dict, optional
        Loaded zone configuration. If None, uses default.
    
    Returns:
    --------
    zone_name : str
        Name of the zone containing this layer
    """
    if zone_config is None:
        zone_config = ZONE_CONFIG
    
    for zone_name, zone_info in zone_config['zones'].items():
        if layer_num in zone_info['layers']:
            return zone_name
    return 'unknown'


def get_zone_layers(zone_name: str, zone_config: Optional[Dict] = None) -> range:
    """
    Get layer range for a given zone name
    
    Parameters:
    -----------
    zone_name : str
        Name of the zone
    zone_config : dict, optional
        Loaded zone configuration. If None, uses default.
    
    Returns:
    --------
    layers : range
        Range of layer numbers in this zone
    """
    if zone_config is None:
        zone_config = ZONE_CONFIG
    
    # Check primary zones
    if zone_name in zone_config['zones']:
        layers = zone_config['zones'][zone_name]['layers']
        return range(min(layers), max(layers) + 1)
    
    # Check aggregate zones
    if 'aggregate_zones' in zone_config and zone_name in zone_config['aggregate_zones']:
        layers = zone_config['aggregate_zones'][zone_name]['layers']
        return range(min(layers), max(layers) + 1)
    
    raise ValueError(f"Unknown zone: {zone_name}")


# ============================================================================
# MULTI-REGION SUPPORT FUNCTIONS
# ============================================================================

def get_region_zones(region_id: int, zone_config: Dict) -> Dict:
    """
    Get zone definitions for a specific region in multi-region config
    
    Parameters:
    -----------
    region_id : int
        Region ID (1, 2, 3, ...)
    zone_config : dict
        Multi-region zone configuration
        
    Returns:
    --------
    zones : dict
        Dictionary mapping zone names to layer lists for this region
    """
    if not is_multiregion_config(zone_config):
        raise ValueError("Config is not multi-region format")
    
    # Convert to plain int to avoid numpy type issues
    region_id = int(region_id)
    
    # Find region by ID
    for region_name, region_config in zone_config['regions'].items():
        if int(region_config['region_id']) == region_id:
            return region_config['zones']
    
    raise ValueError(f"Region ID {region_id} not found in config")


def get_all_region_ids(zone_config: Dict) -> List[int]:
    """
    Get list of all region IDs in multi-region config
    
    Parameters:
    -----------
    zone_config : dict
        Multi-region zone configuration
        
    Returns:
    --------
    region_ids : list of int
        List of region IDs
    """
    if not is_multiregion_config(zone_config):
        raise ValueError("Config is not multi-region format")
    
    region_ids = []
    for region_name, region_config in zone_config['regions'].items():
        region_ids.append(region_config['region_id'])
    
    return sorted(region_ids)


def get_region_name(region_id: int, zone_config: Dict) -> str:
    """
    Get region name from region ID
    
    Parameters:
    -----------
    region_id : int
        Region ID (1, 2, 3, ...)
    zone_config : dict
        Multi-region zone configuration
        
    Returns:
    --------
    region_name : str
        Name of the region (e.g., 'cortex', 'medulla')
    """
    if not is_multiregion_config(zone_config):
        raise ValueError("Config is not multi-region format")
    
    # Convert to plain int to avoid numpy type issues
    region_id = int(region_id)
    
    for region_name, region_config in zone_config['regions'].items():
        if int(region_config['region_id']) == region_id:
            return region_name
    
    raise ValueError(f"Region ID {region_id} not found in config")


def get_multiregion_zone_mask(mlco_mask: np.ndarray,
                               region_id: int,
                               zone_name: str,
                               zone_config: Dict) -> np.ndarray:
    """
    Create zone mask for a specific region
    
    Parameters:
    -----------
    mlco_mask : ndarray
        Multi-region MLCO mask with encoded values
    region_id : int
        Region ID (1, 2, 3, ...)
    zone_name : str
        Name of the zone within the region
    zone_config : dict
        Multi-region zone configuration
        
    Returns:
    --------
    zone_mask : ndarray
        Boolean mask for the specified zone
    """
    region_zones = get_region_zones(region_id, zone_config)
    
    if zone_name not in region_zones:
        raise ValueError(f"Zone '{zone_name}' not found in region {region_id}")
    
    zone_layers = region_zones[zone_name]['layers']
    
    # Create mask for these encoded layers
    zone_mask = np.isin(mlco_mask, zone_layers)
    
    return zone_mask


def extract_region_from_mlco(mlco_mask: np.ndarray, region_id: int) -> np.ndarray:
    """
    Extract all pixels belonging to a specific region
    
    Parameters:
    -----------
    mlco_mask : ndarray
        Multi-region MLCO mask with encoded values
    region_id : int
        Region ID to extract
        
    Returns:
    --------
    region_mask : ndarray
        Boolean mask for all pixels in this region
    """
    # Region pixels are those with values in range [region_id*1000, (region_id+1)*1000)
    return (mlco_mask >= region_id * 1000) & (mlco_mask < (region_id + 1) * 1000)


def get_region_type(zone_name: str, zone_config: Optional[Dict] = None) -> str:
    """
    Get region type for a zone (cortex, cmj, or medulla)
    Used to map zones to appropriate thresholds
    
    Parameters:
    -----------
    zone_name : str
        Name of the zone
    zone_config : dict, optional
        Loaded zone configuration. If None, uses default.
    
    Returns:
    --------
    region_type : str
        'cortex', 'cmj', or 'medulla'
    """
    if zone_config is None:
        zone_config = ZONE_CONFIG
    
    if zone_name in zone_config['zones']:
        return zone_config['zones'][zone_name].get('region_type', 'cortex')
    
    # Default to cortex if not specified
    return 'cortex'


def interpret_tissue_state(t2star_mean: float,
                           perfusion_mean: Optional[float],
                           tissue_quality: Dict,
                           zone: str) -> str:
    """
    Generate interpretation text for tissue state
    
    Parameters:
    -----------
    t2star_mean : float
        Mean T2* in zone
    perfusion_mean : float or None
        Mean perfusion in zone
    tissue_quality : dict
        Tissue quality statistics
    zone : str
        Zone name
        
    Returns:
    --------
    interpretation : str
        Human-readable interpretation
    """
    viable_pct = tissue_quality['viable_pct']
    necrosis_pct = tissue_quality['likely_necrosis_pct']
    
    # Check for severe damage
    if necrosis_pct > 30:
        if perfusion_mean is not None and perfusion_mean < 50:
            return f"🔴 Severe tissue damage - {necrosis_pct:.0f}% necrotic (↑↑T2*, ↓↓perfusion)"
        else:
            return f"🔴 Severe tissue damage - {necrosis_pct:.0f}% likely necrotic/fluid-filled"
    
    # Check for moderate damage
    elif necrosis_pct > 10 or tissue_quality['suspect_edema_pct'] > 20:
        return f"⚠️ Tissue stress - {necrosis_pct:.0f}% necrotic, {tissue_quality['suspect_edema_pct']:.0f}% edematous"
    
    # Check for mild hypoxia (based on perfusion)
    elif perfusion_mean is not None and perfusion_mean < 150:
        return f"⚠️ Reduced perfusion ({perfusion_mean:.0f} ml/100g/min) - mild ischemia"
    
    # Viable tissue
    elif viable_pct > 90:
        return f"✓ Viable tissue ({viable_pct:.0f}% viable)"
    
    else:
        return f"Tissue quality: {viable_pct:.0f}% viable, {tissue_quality['suspect_edema_pct']:.0f}% suspect"


# ============================================================================
# ZONE COMPARISON FUNCTIONS
# ============================================================================

def calculate_effect_size(group1_mean: float, group1_std: float,
                         group2_mean: float, group2_std: float) -> float:
    """
    Calculate Cohen's d effect size
    
    Parameters:
    -----------
    group1_mean, group1_std : float
        Mean and std of group 1
    group2_mean, group2_std : float
        Mean and std of group 2
        
    Returns:
    --------
    cohens_d : float
        Effect size (0.2=small, 0.5=medium, 0.8=large)
    """
    # Pooled standard deviation
    pooled_std = np.sqrt((group1_std**2 + group2_std**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (group2_mean - group1_mean) / pooled_std
    return cohens_d


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    direction = "↑" if d > 0 else "↓"
    return f"{magnitude} {direction}"


if __name__ == "__main__":
    print("BoldPy Tissue Zone Configuration System")
    print("=" * 70)
    
    # Load and display zone configuration
    zone_config = load_zone_config()
    print(f"\nZone Configuration: {zone_config['metadata'].get('organ', 'unknown')}")
    print(f"Total layers per organ: {zone_config['metadata']['n_layers']}")
    print(f"Description: {zone_config['metadata'].get('description', '')}")
    
    print(f"\nZone Breakdown:")
    for zone_name, zone_info in zone_config['zones'].items():
        layers = zone_info['layers']
        print(f"  {zone_name:25s}: Layers {min(layers):2d}-{max(layers):2d} "
              f"({zone_info.get('percentage', 0):2.0f}%) - {zone_info.get('description', '')}")
    
    # Load and display threshold configuration
    threshold_config = load_threshold_config()
    print(f"\nThreshold Configuration: {threshold_config['metadata'].get('species', 'unknown')} "
          f"{threshold_config['metadata'].get('organ', 'unknown')}")
    print(f"Reference: {threshold_config['metadata'].get('reference', '')}")
    
    print(f"\nTissue Classification Thresholds (T2*, ms):")
    thresholds = threshold_config['thresholds']
    print(f"  Viable:          {thresholds['viable']['cortex_t2star'][0]}-{thresholds['viable']['cortex_t2star'][1]} ms")
    print(f"  Suspect Edema:   {thresholds['suspect_edema']['cortex_t2star'][0]}-{thresholds['suspect_edema']['cortex_t2star'][1]} ms")
    print(f"  Likely Necrosis: >{thresholds['likely_necrosis']['cortex_t2star'][0]} ms")
    
    if 'combination_rules' in threshold_config:
        rules = threshold_config['combination_rules'].get('high_confidence_necrosis', {})
        if rules:
            print(f"\n  High confidence necrosis: T2* >{rules.get('min_t2star', 60)} ms "
                  f"AND perfusion <{rules.get('max_perfusion', 50)} ml/100g/min")

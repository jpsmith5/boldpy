"""
Bruker ParaVision Dataset Loader
=================================

Load multi-echo BOLD data from Bruker ParaVision format.
Supports PvDatasets with method files and 2dseq images.
Also handles .PvDatasets ZIP archives (extracts automatically).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import struct
import logging
import zipfile
import tempfile
import shutil

logger = logging.getLogger(__name__)


class PvDatasetLoader:
    """
    Load Bruker ParaVision datasets
    
    Supports typical BOLD acquisition structure:
    - Method file with parameters
    - 2dseq files with image data
    - visu_pars with visualization parameters
    
    Parameters
    ----------
    pdata_path : str or Path
        Path to pdata directory (e.g., .../pdata/1/)
        
    Example
    -------
    >>> loader = PvDatasetLoader('/path/to/study/1/pdata/1')
    >>> echo_images, echo_times, metadata = loader.load()
    >>> print(f"Loaded {len(echo_times)} echoes: {echo_images.shape}")
    """
    
    def __init__(self, pdata_path: Union[str, Path]):
        """Initialize loader with pdata path"""
        self.pdata_path = Path(pdata_path)
        
        if not self.pdata_path.exists():
            raise FileNotFoundError(f"PvDataset not found: {self.pdata_path}")
        
        # Key files
        self.method_file = self._find_method_file()
        self.visu_pars_file = self.pdata_path / 'visu_pars'
        self.twodseq_file = self.pdata_path / '2dseq'
        
        logger.info(f"Initialized PvDataset loader: {self.pdata_path}")
    
    def _find_method_file(self) -> Path:
        """Find method file (in parent directory)"""
        # Method file is typically in parent of pdata
        parent = self.pdata_path.parent
        method_file = parent / 'method'
        
        if not method_file.exists():
            # Try going up one more level
            parent = parent.parent
            method_file = parent / 'method'
        
        if not method_file.exists():
            logger.warning("Method file not found")
            return None
        
        return method_file
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load complete dataset
        
        Returns
        -------
        echo_images : np.ndarray
            Echo images, shape (n_echoes, height, width) or (n_echoes, n_slices, height, width)
        echo_times : np.ndarray
            Echo times in milliseconds
        metadata : dict
            Complete metadata (voxel size, TR, etc.)
        """
        logger.info("Loading PvDataset...")
        
        # Load images
        echo_images, image_metadata = self._load_2dseq()
        
        # Load echo times
        echo_times = self._load_echo_times()
        
        # Load voxel dimensions
        voxel_size = self._load_voxel_size()
        
        # Compile metadata
        metadata = {
            **image_metadata,
            'echo_times_ms': echo_times.tolist(),
            'voxel_size_mm': voxel_size,
            'pdata_path': str(self.pdata_path),
            'method_file': str(self.method_file) if self.method_file else None
        }
        
        # Load additional parameters
        if self.method_file and self.method_file.exists():
            method_params = self._parse_method_file()
            metadata.update(method_params)
        
        logger.info(f"Loaded {len(echo_times)} echoes: {echo_images.shape}")
        logger.info(f"Echo times: {echo_times} ms")
        logger.info(f"Voxel size: {voxel_size} mm")
        
        return echo_images, echo_times, metadata
    
    def _load_2dseq(self) -> Tuple[np.ndarray, Dict]:
        """Load 2dseq image file"""
        if not self.twodseq_file.exists():
            raise FileNotFoundError(f"2dseq file not found: {self.twodseq_file}")
        
        # Parse visu_pars for image dimensions
        visu_pars = self._parse_visu_pars()
        
        # Get image dimensions
        dim = visu_pars.get('VisuCoreSize', [256, 256])
        data_type = visu_pars.get('VisuCoreWordType', '_16BIT_SGN_INT')
        n_images = visu_pars.get('VisuCoreFrameCount', 1)
        
        # Determine numpy dtype
        dtype_map = {
            '_8BIT_UNSGN_INT': np.uint8,
            '_16BIT_SGN_INT': np.int16,
            '_32BIT_SGN_INT': np.int32,
            '_32BIT_FLOAT': np.float32,
        }
        dtype = dtype_map.get(data_type, np.int16)
        
        # Read binary data
        with open(self.twodseq_file, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
        
        # Reshape based on dimensions
        if len(dim) == 2:
            width, height = dim
            expected_size = width * height * n_images
        elif len(dim) == 3:
            width, height, n_slices = dim
            expected_size = width * height * n_slices * n_images
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
        
        if data.size != expected_size:
            logger.warning(
                f"Data size mismatch: got {data.size}, expected {expected_size}"
            )
        
        # Reshape
        if len(dim) == 2:
            # 2D images
            images = data.reshape(n_images, height, width)
        else:
            # 3D images
            images = data.reshape(n_images, n_slices, height, width)
        
        metadata = {
            'image_dimensions': dim,
            'n_images': n_images,
            'data_type': data_type,
            'dtype': str(dtype)
        }
        
        return images.astype(np.float32), metadata
    
    def _load_echo_times(self) -> np.ndarray:
        """Load echo times from method file"""
        if not self.method_file or not self.method_file.exists():
            logger.warning("Method file not found, using default echo times")
            # Try to infer from number of images
            visu_pars = self._parse_visu_pars()
            n_images = visu_pars.get('VisuCoreFrameCount', 1)
            # Default: start at 3.5ms, 3.5ms spacing
            return np.arange(1, n_images + 1) * 3.5
        
        # Parse method file for echo times
        method_params = self._parse_method_file()
        
        # Look for echo time parameters
        if 'PVM_EchoTime' in method_params:
            echo_times = method_params['PVM_EchoTime']
            if isinstance(echo_times, (int, float)):
                echo_times = [echo_times]
            return np.array(echo_times)
        
        # Alternative: EffectiveEchoTime
        if 'EffectiveEchoTime' in method_params:
            echo_times = method_params['EffectiveEchoTime']
            if isinstance(echo_times, (int, float)):
                echo_times = [echo_times]
            return np.array(echo_times)
        
        # Fallback: try to find any echo time array
        for key in method_params:
            if 'echo' in key.lower() and 'time' in key.lower():
                value = method_params[key]
                if isinstance(value, (list, np.ndarray)):
                    return np.array(value)
        
        logger.warning("Echo times not found in method file, using defaults")
        visu_pars = self._parse_visu_pars()
        n_images = visu_pars.get('VisuCoreFrameCount', 1)
        return np.arange(1, n_images + 1) * 3.5
    
    def _load_voxel_size(self) -> Tuple[float, float, float]:
        """Load voxel dimensions in mm"""
        visu_pars = self._parse_visu_pars()
        
        # VisuCoreExtent gives FOV in mm
        extent = visu_pars.get('VisuCoreExtent', [23.04, 23.04])
        
        # VisuCoreSize gives matrix size
        size = visu_pars.get('VisuCoreSize', [256, 256])
        
        # Calculate voxel size
        if len(extent) == 2 and len(size) == 2:
            voxel_x = extent[0] / size[0]
            voxel_y = extent[1] / size[1]
            return (voxel_x, voxel_y, 1.0)  # Default slice thickness
        elif len(extent) == 3 and len(size) == 3:
            voxel_x = extent[0] / size[0]
            voxel_y = extent[1] / size[1]
            voxel_z = extent[2] / size[2]
            return (voxel_x, voxel_y, voxel_z)
        
        return (0.09, 0.09, 1.0)  # Default fallback
    
    def _parse_visu_pars(self) -> Dict:
        """Parse visu_pars file"""
        if not self.visu_pars_file.exists():
            logger.warning(f"visu_pars not found: {self.visu_pars_file}")
            return {}
        
        params = {}
        
        with open(self.visu_pars_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parameter line starts with ##$
            if line.startswith('##$'):
                # Extract parameter name and value
                if '=' in line:
                    name, value = line[3:].split('=', 1)
                    name = name.strip()
                    value = value.strip()
                    
                    # Check if value is on next line(s)
                    if value == '' or value.startswith('('):
                        # Array parameter
                        i += 1
                        value_lines = []
                        while i < len(lines) and not lines[i].startswith('##'):
                            value_lines.append(lines[i].strip())
                            i += 1
                        value = ' '.join(value_lines)
                        i -= 1
                    
                    # Parse value
                    params[name] = self._parse_param_value(value)
            
            i += 1
        
        return params
    
    def _parse_method_file(self) -> Dict:
        """Parse method file"""
        if not self.method_file or not self.method_file.exists():
            return {}
        
        params = {}
        
        with open(self.method_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('##$'):
                if '=' in line:
                    name, value = line[3:].split('=', 1)
                    name = name.strip()
                    value = value.strip()
                    
                    # Multi-line values
                    if value == '' or value.startswith('('):
                        i += 1
                        value_lines = []
                        while i < len(lines) and not lines[i].startswith('##'):
                            value_lines.append(lines[i].strip())
                            i += 1
                        value = ' '.join(value_lines)
                        i -= 1
                    
                    params[name] = self._parse_param_value(value)
            
            i += 1
        
        return params
    
    def _parse_param_value(self, value: str):
        """Parse parameter value from string"""
        value = value.strip()
        
        # Remove parentheses for arrays
        if value.startswith('(') and ')' in value:
            # Array notation like (3, 5) or (2)
            value = value.split(')', 1)[1].strip()
        
        # Try to parse as number
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try array of numbers
        if ' ' in value:
            try:
                parts = value.split()
                numbers = [float(p) if '.' in p else int(p) for p in parts]
                return numbers
            except ValueError:
                pass
        
        # String value
        value = value.strip('<>')
        return value


def load_pvdataset(
    pdata_path: Union[str, Path],
    select_slice: Optional[int] = None,
    use_brkraw: Optional[bool] = None,
    keep_extracted: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Bruker ParaVision dataset
    
    Handles both .PvDatasets ZIP archives and extracted directories.
    
    Parameters
    ----------
    pdata_path : str or Path
        Path to:
        - .PvDatasets ZIP archive file (auto-extracts)
        - Extracted pdata directory (e.g., .../1/pdata/1/)
    select_slice : int, optional
        If 3D data, select specific slice (0-indexed)
    use_brkraw : bool, optional
        Use BrkRaw backend if available (experimental):
        - None (default): Use extraction method for archives
        - True: Try BrkRaw (requires installation from GitHub)
        - False: Use custom parser only
    keep_extracted : bool
        Keep extracted files after loading (default: False)
        
    Returns
    -------
    echo_images : np.ndarray
        Echo images, shape (n_echoes, height, width)
    echo_times : np.ndarray
        Echo times in milliseconds
    metadata : dict
        Complete metadata
        
    Example
    -------
    >>> # Load .PvDatasets ZIP archive (auto-extracts)
    >>> images, times, meta = load_pvdataset('BOLD.PvDatasets')
    >>> 
    >>> # Load extracted directory
    >>> images, times, meta = load_pvdataset('/path/to/pdata/1')
    """
    pdata_path = Path(pdata_path)
    
    # Check if this is a .PvDatasets ZIP archive
    is_archive = pdata_path.is_file() and pdata_path.suffix == '.PvDatasets'
    
    if is_archive:
        logger.info(f"Detected .PvDatasets archive: {pdata_path.name}")
        
        # Try BrkRaw if explicitly requested
        if use_brkraw is True:
            try:
                from .pvdataset_brkraw import load_pvdataset_brkraw
                logger.info("Using BrkRaw backend for archive")
                return load_pvdataset_brkraw(pdata_path, select_slice=select_slice)
            except ImportError:
                logger.warning(
                    "BrkRaw not available (install from GitHub: "
                    "pip install git+https://github.com/BrkRaw/bruker.git)"
                )
                logger.info("Falling back to extraction method")
        
        # Default: Extract and use custom parser
        logger.info("Extracting archive...")
        extracted_path, temp_dir = _extract_pvdatasets_archive(pdata_path)
        
        try:
            # Load from extracted directory
            loader = PvDatasetLoader(extracted_path)
            echo_images, echo_times, metadata = loader.load()
            
            # Add archive info to metadata
            metadata['source_archive'] = str(pdata_path)
            metadata['extracted_from_archive'] = True
            
        finally:
            # Clean up temporary extraction unless user wants to keep it
            if not keep_extracted and temp_dir:
                logger.info(f"Cleaning up extracted files: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif temp_dir:
                logger.info(f"Extracted files kept at: {temp_dir}")
                metadata['extracted_path'] = str(temp_dir)
    
    else:
        # Regular directory - use custom parser
        # (BrkRaw option available but not default for extracted dirs)
        if use_brkraw is True:
            try:
                from .pvdataset_brkraw import load_pvdataset_brkraw
                logger.info("Using BrkRaw backend")
                return load_pvdataset_brkraw(pdata_path, select_slice=select_slice)
            except ImportError:
                logger.warning("BrkRaw not available, using custom parser")
        
        loader = PvDatasetLoader(pdata_path)
        echo_images, echo_times, metadata = loader.load()
    
    # Handle slice selection
    if echo_images.ndim == 4:
        n_echoes, n_slices, height, width = echo_images.shape
        
        if select_slice is not None:
            if select_slice >= n_slices:
                raise ValueError(
                    f"Slice {select_slice} out of range (0-{n_slices-1})"
                )
            echo_images = echo_images[:, select_slice, :, :]
            logger.info(f"Selected slice {select_slice}/{n_slices-1}")
        else:
            # Default: middle slice
            middle_slice = n_slices // 2
            echo_images = echo_images[:, middle_slice, :, :]
            logger.info(f"Auto-selected middle slice {middle_slice}/{n_slices-1}")
            metadata['selected_slice'] = middle_slice
    
    return echo_images, echo_times, metadata


def _extract_pvdatasets_archive(
    pvdatasets_path: Path,
    output_dir: Optional[Path] = None
) -> Tuple[Path, Optional[Path]]:
    """
    Extract .PvDatasets ZIP archive to directory
    
    Parameters
    ----------
    pvdatasets_path : Path
        Path to .PvDatasets ZIP file
    output_dir : Path, optional
        Output directory for extraction
        If None, uses temporary directory
        
    Returns
    -------
    pdata_path : Path
        Path to extracted pdata/1 directory
    temp_dir : Path or None
        Temporary directory (if used), None otherwise
    """
    if not pvdatasets_path.exists():
        raise FileNotFoundError(f"Archive not found: {pvdatasets_path}")
    
    # Determine output directory
    if output_dir is None:
        # Use temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix='boldpy_pvdatasets_'))
        output_dir = temp_dir
        logger.info(f"Extracting to temporary directory: {temp_dir}")
    else:
        temp_dir = None
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting to: {output_dir}")
    
    # Extract ZIP archive
    try:
        with zipfile.ZipFile(pvdatasets_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"✓ Extracted {pvdatasets_path.name}")
    except zipfile.BadZipFile:
        raise ValueError(
            f"Invalid .PvDatasets file (not a valid ZIP archive): {pvdatasets_path}"
        )
    except Exception as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract archive: {e}")
    
    # Find the pdata/1 directory in extracted files
    pdata_dirs = list(output_dir.rglob('pdata/1'))
    
    if not pdata_dirs:
        # Try looking for just pdata directories
        pdata_dirs = list(output_dir.rglob('pdata'))
        if pdata_dirs:
            # Use first pdata directory found
            pdata_path = pdata_dirs[0]
            # Look for subdirectory with 2dseq
            subdirs = [d for d in pdata_path.iterdir() if d.is_dir()]
            for subdir in subdirs:
                if (subdir / '2dseq').exists():
                    logger.info(f"Found pdata directory: {subdir}")
                    return subdir, temp_dir
        
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(
            f"No valid pdata directory found in {pvdatasets_path}\n"
            f"Expected structure: .../pdata/1/2dseq"
        )
    
    pdata_path = pdata_dirs[0]
    logger.info(f"Found pdata directory: {pdata_path}")
    
    return pdata_path, temp_dir


def main():
    """Command-line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Load Bruker ParaVision dataset'
    )
    parser.add_argument('pdata_path', help='Path to pdata directory')
    parser.add_argument('--slice', type=int, help='Select slice (if 3D)')
    parser.add_argument('--save', help='Save as .npy file')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Load dataset
    echo_images, echo_times, metadata = load_pvdataset(
        args.pdata_path,
        select_slice=args.slice
    )
    
    print("\n" + "="*70)
    print("PVDATASET LOADED")
    print("="*70)
    print(f"Shape: {echo_images.shape}")
    print(f"Echo times: {echo_times} ms")
    print(f"Voxel size: {metadata.get('voxel_size_mm')} mm")
    print(f"Data range: [{np.min(echo_images):.1f}, {np.max(echo_images):.1f}]")
    
    if args.save:
        output = Path(args.save)
        np.save(output / 'echo_images.npy', echo_images)
        np.save(output / 'echo_times.npy', echo_times)
        
        # Save metadata as JSON
        import json
        with open(output / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved to {output}")


if __name__ == '__main__':
    main()

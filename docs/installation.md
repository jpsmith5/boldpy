# Installation Guide

Complete instructions for installing BoldPy and its dependencies.

---

## System Requirements

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+)
- Windows (10/11 with WSL2 recommended)

### Python Version
- Python 3.8 or higher
- Python 3.10+ recommended for best performance

### Hardware
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 10 GB free space for data and results
- **CPU:** Multi-core processor recommended for faster fitting

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Clone the Repository

```bash
git clone https://github.com/jpsmith5/boldpy.git
cd boldpy
```

#### 2. Create Virtual Environment (Optional but Recommended)

**Using venv:**
```bash
python -m venv boldpy_env
source boldpy_env/bin/activate  # On Linux/macOS
# OR
boldpy_env\Scripts\activate  # On Windows
```

**Using conda:**
```bash
conda create -n boldpy python=3.10
conda activate boldpy
```

#### 3. Install BoldPy

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode, allowing you to modify the code if needed.

#### 4. Verify Installation

```bash
python -c "from boldpy.fitting.t2star_fitter import fit_t2star_map; print('✓ BoldPy installed successfully!')"
```

You should see: `✓ BoldPy installed successfully!`

---

### Method 2: Development Installation

For contributors who want to run tests and build documentation:

```bash
# Clone repository
git clone https://github.com/jpsmith5/boldpy.git
cd boldpy

# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/
```

---

### Method 3: PyPI Installation (Future)

Once BoldPy is published to PyPI:

```bash
pip install boldpy
```

---

## Dependencies

### Required Dependencies

BoldPy automatically installs these core dependencies:

```txt
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-image>=0.18.0
scikit-learn>=1.0.0
Pillow>=8.0.0
tqdm>=4.60.0
```

### Optional Dependencies

For development and documentation:

```txt
pytest>=7.0.0
pytest-cov>=3.0.0
mkdocs>=1.4.0
mkdocs-material>=9.0.0
mkdocstrings[python]>=0.20.0
```

---

## Verifying Your Installation

### Quick Test

Run this command to verify all core functionality:

```bash
python -c "
from boldpy.fitting.t2star_fitter import fit_t2star_map
from boldpy.analysis.perfusion_analysis import load_bruker_perfusion
import numpy as np

# Test T2* fitting
test_data = np.random.randn(8, 100, 100) * 100 + 1000
test_times = np.array([3, 7, 11, 15, 19, 23, 27, 31])
result = fit_t2star_map(test_data, test_times, show_progress=False)

print('✓ All core modules loaded successfully!')
print(f'✓ T2* fitting works (result shape: {result[\"t2star\"].shape})')
"
```

### Run Example Analysis

Test with provided example data (if available):

```bash
cd examples/
python run_example_analysis.py
```

---

## Troubleshooting Installation

### Issue: pip install fails with "No module named 'numpy'"

**Solution:** Install numpy first:
```bash
pip install numpy
pip install -e .
```

### Issue: "ImportError: cannot import name 'fit_t2star_map'"

**Solution:** Make sure you're running from the boldpy directory:
```bash
cd /path/to/boldpy
pip install -e .
```

### Issue: Permission errors on Linux/macOS

**Solution:** Use `--user` flag:
```bash
pip install --user -e .
```

Or install in a virtual environment (recommended).

### Issue: matplotlib backend errors

**Solution:** Install GUI backend:

**Linux:**
```bash
sudo apt-get install python3-tk
```

**macOS:**
```bash
brew install python-tk
```

**Windows:**
```bash
pip install pyqt5
```

### Issue: scikit-image compilation errors

**Solution:** Install binary wheels:
```bash
pip install --only-binary :all: scikit-image
```

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv
sudo apt-get install build-essential  # For compiling extensions
```

### macOS

Install Xcode Command Line Tools:
```bash
xcode-select --install
```

Install Homebrew (if not installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Windows

**Recommended:** Use Windows Subsystem for Linux (WSL2):

1. Install WSL2: Follow [Microsoft's guide](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install Ubuntu from Microsoft Store
3. Follow Linux installation instructions above

**Alternative:** Native Windows installation may work but is not officially supported.

---

## Updating BoldPy

### From Git Repository

```bash
cd /path/to/boldpy
git pull origin main
pip install -e . --upgrade
```

### From PyPI (Future)

```bash
pip install --upgrade boldpy
```

---

## Uninstalling BoldPy

```bash
pip uninstall boldpy
```

To also remove the cloned repository:
```bash
rm -rf /path/to/boldpy
```

---

## Docker Installation (Advanced)

For reproducible environments, use Docker:

**Dockerfile** (create this in boldpy directory):
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install BoldPy
COPY . /app
RUN pip install -e .

CMD ["bash"]
```

**Build and run:**
```bash
docker build -t boldpy:latest .
docker run -it -v $(pwd)/data:/data boldpy:latest
```

---

## Next Steps

After installation:

1. **[Quick Start Guide](quick-start.md)** - Run your first analysis
2. **[User Guide](user-guide.md)** - Complete workflow and feature reference
3. **[Examples with Data](examples-with-data.md)** - Expected outputs and interpretation

---

## Getting Help

If you encounter installation issues:

1. Check the troubleshooting section in [User Guide](user-guide.md)
2. Search [GitHub Issues](https://github.com/jpsmith5/boldpy/issues)
3. Ask in [GitHub Discussions](https://github.com/jpsmith5/boldpy/discussions)
4. Email: jasonsmith@virginia.edu

---

**Installation complete!** Ready to [get started](quick-start.md)? 🚀

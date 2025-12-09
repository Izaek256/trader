# MetaTrader5 Installation Guide

## Problem
The `MetaTrader5` package is not available for Python 3.12+ (including Python 3.14). The package officially supports Python 3.8-3.11.

## Solutions

### Option 1: Use Python 3.11 (Recommended)

1. **Install Python 3.11**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use pyenv: `pyenv install 3.11.9`

2. **Create a virtual environment with Python 3.11**:
   ```bash
   # Using venv
   py -3.11 -m venv venv
   venv\Scripts\activate
   
   # Or using conda
   conda create -n mt5 python=3.11
   conda activate mt5
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install MetaTrader5
   ```

### Option 2: Try Installing from Wheel File

1. **Download the wheel file**:
   - Visit: https://www.mql5.com/en/docs/integration/python_metatrader5
   - Or search for "MetaTrader5 python wheel download"

2. **Install the wheel**:
   ```bash
   pip install MetaTrader5-5.0.45-cp311-cp311-win_amd64.whl
   # (adjust version and Python version in filename)
   ```

### Option 3: Install from MT5 Terminal Directory

If you have MetaTrader 5 terminal installed:

1. **Locate your MT5 installation** (typically):
   - `C:\Program Files\MetaTrader 5\`
   - Or `C:\Users\<YourUser>\AppData\Roaming\MetaQuotes\Terminal\<TerminalID>\`

2. **Install from terminal directory**:
   ```bash
   pip install "C:\Program Files\MetaTrader 5\MQL5\Scripts\include\python\MetaTrader5"
   ```

### Option 4: Use Docker (Advanced)

Create a Docker container with Python 3.11 and install MT5 there.

## Verify Installation

After installation, verify it works:

```python
import MetaTrader5 as mt5
print(mt5.version())
```

If this runs without errors, installation was successful!

## Current Python Version Issue

You're currently using **Python 3.14.0**, which is not supported. Please use Python 3.11 or earlier.


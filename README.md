# Zeeman Effect Analysis Software

A Python-based application for analyzing the Zeeman effect in spectral lines.

## Overview

This software simplifies the analysis of the Zeeman effect by automating the measurement of spectral line splitting and calculating fundamental physical constants such as the Bohr magneton and specific charge of the electron.

## Features

- User-friendly graphical interface
- Image input from file
- Automated spectral line analysis
- Calculation of Bohr magneton and specific charge
- Data export as CVS

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Source the .venv
```bash
source .venv/bin/activate
```
Run the application:
```bash
python src/main.py
```

1. Load an image of the spectral lines
2. Input the applied current, the wavelenght, and magnetic field values
3. Adjust image processing parameters if needed
4. Export data as needed

## Details

The software uses:
- PyQt6 for the graphical interface
- OpenCV for image processing
- NumPy for numerical computations
- Pandas for data management
- Matplotlib for visualization


# Homework 1 - CSDS 313

## Overview
This folder contains all code, data, and analysis for Homework 1 in CSDS 313. The assignment covers data cleaning, visualization, and statistical modeling for several datasets, including algorithm performance, anime ratings, and best-selling albums.

## Contents
- **src/**: Python scripts for analysis and visualization
    - `algorithm_performance.py`: Cleans and analyzes algorithm trials, generates error bar, barcode, and histogram plots.
    - `anime.py`: Cleans and analyzes anime ratings, generates three main plots.
    - `best_selling_albums.py`: Cleans and analyzes album sales, generates three main plots.
- **data/**: Datasets used for analysis
- **graphs/**: Output plots and visualizations
- **Report.pdf**: Final report for Homework 1

## How to Run
1. Install dependencies:
   ```sh
   pip install pandas numpy matplotlib seaborn
   ```
2. Run analysis scripts:
   ```sh
   python src/algorithm_performance.py
   python src/anime.py
   python src/best_selling_albums.py
   ```
3. View output plots in the `graphs/` subfolders.

## Notes
- Data cleaning and outlier removal steps are included in each script.
- Plots are saved automatically in the appropriate subfolders.
- See the report for discussion and interpretation of results.

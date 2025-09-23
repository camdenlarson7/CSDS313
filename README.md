# CSDS 313 Coursework Repository

## Overview
This repository contains code, data, and analysis for assignments in CSDS 313. It covers data cleaning, visualization, and statistical modeling for multiple real-world datasets, including music albums, anime ratings, algorithm performance, running times, city populations, and call details. The code demonstrates best practices in data science, including reproducible analysis, clear visualizations, and model fitting.

## Directory Structure
```
CSDS 313/
├── Homework1/             # Homework 1 code, data, and report
│   ├── src/
│   ├── data/
│   ├── graphs/
│   ├── README.md
│   └── Report.pdf
├── src/                   # Python scripts for analysis and visualization
├── data/                  # Raw and processed datasets (CSV files)
├── graphs/                # Output plots and visualizations
├── README.md              # This file
```

## Datasets
- **10k_times.csv**: 10K race times (seconds)
- **algorithm_trials.csv**: Algorithm accuracy results
- **Anime.csv**: Anime ratings and types
- **BRAZIL_CITIES.csv**: Brazilian city populations and metadata
- **call_details.csv**: Customer call time details
- **Top_10_Albums_By_Year.csv**: Album sales by year and genre
- **uniform_random_results.csv**: Simulated uniform random numbers

## Scripts
- **best_selling_albums.py**: Cleans album data, generates error bar, barcode, and histogram plots by genre.
- **anime.py**: Cleans anime data, generates error bar, barcode, and histogram plots by type.
- **algorithm_performance.py**: Cleans algorithm trials, normalizes names, filters accuracy, generates three plots by algorithm.
- **normal_distribution.py**: Loads running times, cleans data, generates histograms with various bin sizes, and formats axes.
- **exponential_distribution.py**: Loads call details, filters outliers, plots call time distribution.
- **power_law.py**: Loads city population data, plots histogram with log bins.
- **uniform_distribution.py**: Simulates uniform random numbers, saves results, and plots distribution.
- **fit_distributions.py**: Fits normal, uniform, power law, and exponential models to each dataset, generates synthetic samples, compares distributions, and visualizes results.

## How to Run
1. **Install dependencies:**
   ```sh
   pip install pandas numpy matplotlib seaborn scipy powerlaw
   ```
2. **Run analysis scripts:**
   ```sh
   python src/best_selling_albums.py
   python src/anime.py
   python src/algorithm_performance.py
   python src/normal_distribution.py
   python src/exponential_distribution.py
   python src/power_law.py
   python src/uniform_distribution.py
   python src/fit_distributions.py
   ```
3. **View output plots:**
   - Plots are saved in the `graphs/` subfolders by topic.

## Model Fitting & Evaluation
- All datasets are fit to four models (normal, uniform, power law, exponential) using maximum likelihood estimation or standard fitting methods.
- Synthetic samples are generated for each model and compared to real data using visualizations and the Kolmogorov-Smirnov test.
- Results and reflections are printed and visualized in `fit_distributions.py`.

## Reporting
- The LaTeX report and PDF summarize the analysis, results, and conclusions.

## Notes
- Data cleaning steps are included in each script for reproducibility.
- Outlier removal and bin size selection are documented in code comments.
- All code is written in Python 3 and uses open-source libraries.

## License
This repository is for educational use in CSDS 313. Please cite appropriately if reusing code or data.

# Signal Separation and Analysis Project

This project demonstrates the methodology of separating mixed audio signals into their original sources using Independent Component Analysis (ICA). It applies advanced signal processing techniques to recover original signals from their mixed forms accurately.

## Dependencies
To successfully run the project, ensure you have the following Python libraries installed:
- NumPy
- SciPy
- Matplotlib
- Pandas
- Scikit-learn
- Soundfile

## Setup and Installation
Make sure Python is installed on your system. You can install the required dependencies using pip:
```bash
pip install numpy scipy matplotlib pandas scikit-learn soundfile
```
##  Data Preparation
The project uses three mixed audio signals (x1.wav, x2.wav, x3.wav) and their corresponding original sources (s1.wav, s2.wav, s3.wav). The signals are read and standardized using Soundfile, ensuring zero mean and unit variance.


## Signal Whitening
Signal whitening is applied to make the observed signals suitable for ICA by removing any correlation. This step involves covariance matrix computation, eigenvalue decomposition, and transformation to the whitened space.

## Independent Component Analysis
The ICA process seeks to find basis vectors that maximize statistical independence among separated signals. This iterative process involves orthogonalizing, normalizing, and updating basis vectors based on kurtosis until convergence.

## Signal Separation
Separated signals are obtained by projecting the mixed signals back to the source space using derived basis vectors.


## Evaluation
- Correlation Matrix: Computes similarities between original and separated signals.
- Cross-Correlation: Analyzes cross-correlation to match separated signals with originals.
- Mean Squared Error (MSE): Quantifies the separation accuracy between original and separated signals.
- Visualization: Plots for visual inspection and comparison of original vs. separated signals.

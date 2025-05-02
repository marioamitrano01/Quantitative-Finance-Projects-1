# Yield Curve Spline Fitter

Project for analyzing the U.S. Treasury yield curve using  interpolation techniques.

## Overview

This code provides tools for analyzing and visualizing yield curves, which represent the relationship between interest rates and time to maturity. Understanding yield curves is crucial for financial analysis, economic forecasting, and fixed-income portfolio management.

## The Mathematics Behind Yield Curve Analysis

### Natural Cubic Splines

The core of our library uses natural cubic spline interpolation to model yield curves. Cubic splines are piecewise polynomial functions that:

- Create a smooth curve passing through all observed data points
- Maintain continuous first and second derivatives at knot points
- Have natural boundary conditions (zero second derivative at endpoints)

For a set of yields at discrete maturities, the spline constructs cubic polynomials between each pair of adjacent points:

### Gaussian Process Regression (GPR)

As an alternative approach, I implement Gaussian Process Regression with a Matérn kernel to model yield curves as probabilistic functions. This method:

- Produces smooth interpolations with uncertainty estimates
- Handles noisy observations more robustly than splines
- Provides confidence intervals around predictions
- Automatically determines the optimal smoothing parameters

The Matérn kernel balances smoothness with flexibility, making it well-suited for yield curve modeling.

### Neural Network Approach

For capturing complex, non-linear relationships, I employ a Multi-Layer Perceptron (MLP) that:

- Learns patterns directly from the data
- Can capture regime-dependent yield curve shapes
- May generalize better to unusual market conditions
- Adapts to changing relationships over time

## Why Use Multiple Models?

Different interpolation techniques excel in different market environments:

- **Splines**: Best for smooth, well-behaved yield curves with sufficient data points
- **GPR**: Ideal when uncertainty quantification is needed or data is noisy
- **MLP**: Powerful for capturing complex non-linear relationships or regime changes

By comparing multiple approaches, analysts can gain greater confidence in their yield curve interpretations and better understand model risk.

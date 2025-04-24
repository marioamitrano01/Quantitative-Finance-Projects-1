# U.S. Treasury Yield Curve Interpolation

This project demonstrates yield curve interpolation techniques by fetching U.S. Treasury yield data from FRED and comparing three different interpolation methods:

- **Natural Cubic Spline Interpolation**  
  A deterministic mathematical approach that provides a smooth interpolation between yield data points.

- **Gaussian Process Regression (GPR) with a Matérn Kernel**  
  A probabilistic machine learning method that not only interpolates the data but also provides uncertainty estimates. In this project, the GPR model uses feature scaling and a Matérn kernel (with ν=2.5) for improved modeling of the yield curve.

- **Multi-Layer Perceptron (MLP) Regressor**  
  A deep neural network approach that captures nonlinear relationships. Input and output feature scaling is applied to enhance numerical stability and convergence.

## Goal

The main goal of this project is to:
- **Collect U.S. Treasury yield curve data** for selected maturities (6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, and 30 Yr) from the Federal Reserve Economic Data (FRED) API.
- **Apply interpolation techniques** (Cubic Spline, GPR, and MLP) to estimate the yield curve.
- **Visualize the results** interactively, comparing the performance of the three methods.

## Techniques and Mathematical Improvements

1. **Natural Cubic Spline Interpolation:**  
   - Utilizes a classical mathematical formulation to ensure smoothness of the yield curve.
   - Efficiently computes coefficients for piecewise cubic polynomials.

2. **Gaussian Process Regression (GPR):**  
   - Employs a Matérn kernel (ν=2.5) combined with a constant kernel and a white noise kernel.
   - Applies feature scaling using scikit‑learn's `StandardScaler` for improved hyperparameter optimization.
   - Provides uncertainty quantification through confidence intervals.

3. **Multi-Layer Perceptron (MLP) Regressor:**  
   - Uses a neural network with two hidden layers (50 and 25 neurons) to capture complex nonlinear relationships.
   - Incorporates feature scaling for both inputs and outputs to enhance model performance.
   - Serves as an alternative machine learning approach to compare against GPR.

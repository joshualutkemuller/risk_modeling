# risk_modeling

This repository contains a sample mean-variance optimization model implemented in
Python.  The implementation lives in [`mean_variance.py`](mean_variance.py) and
provides a small utility class for constructing efficient portfolios under the
classic Markowitz framework.

## Quick start

1. Ensure you have Python 3.9+ with NumPy installed.
2. Run the example script:

   ```bash
   python mean_variance.py
   ```

   The script prints the weights and risk/return characteristics of the global
   minimum variance portfolio, an efficient portfolio for a chosen target
   return, and a few points sampled from the efficient frontier.

## Extending the model

The `MeanVarianceOptimizer` class offers the following methods you can reuse in
other workflows:

* `global_minimum_variance_weights()` – Compute the minimum-variance portfolio
  satisfying the full-investment constraint.
* `efficient_weights(target_return)` – Generate the weights that minimize
  variance for a requested expected return.
* `efficient_frontier(num_points=20)` – Sample points along the efficient
  frontier to visualize the trade-off between risk and return.
* `portfolio_performance(weights)` – Evaluate the expected return and
  volatility of any portfolio weights.

These building blocks can be integrated into more sophisticated investment or
risk analysis pipelines.

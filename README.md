# risk_modeling

This repository contains two complementary components:

1. A classic mean-variance optimization module implemented in
   [`mean_variance.py`](mean_variance.py).
2. An agentic AI workflow for quantitative portfolio construction located in
   the [`agentic_quant`](agentic_quant) package.  The workflow demonstrates how
   multiple specialized agents (data, signal, risk, optimization, and
   reporting) can collaborate through a shared blackboard to produce an
   institutional-style investment report.

## Quick start

1. Ensure you have Python 3.9+ with NumPy installed.
2. Run the classic optimization example:

   ```bash
   python mean_variance.py
   ```

   The script prints the weights and risk/return characteristics of the global
   minimum variance portfolio, an efficient portfolio for a chosen target
   return, and a few points sampled from the efficient frontier.

3. Run the agentic workflow to see the autonomous pipeline in action:

   ```bash
   python -m agentic_quant.main
   ```

   You will receive a multi-section report summarizing simulated data, signal
   estimates, risk analytics, and the resulting portfolio after a risk overlay
   agent applies leverage and concentration limits.

   If you prefer to integrate the workflow into your own script or notebook
   without invoking the CLI entry point, import and call
   `agentic_quant.run_workflow()` directly:

   ```python
   from agentic_quant import run_workflow

   report = run_workflow(
       tickers=("TECH", "HEALTH", "ENERGY", "UTIL"),
       periods=504,
       target_return=0.12,
   )
   print(report)
   ```

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
risk analysis pipelines.  The agentic toolkit builds directly on top of this
optimizer, giving you a template for orchestrating autonomous research
workflows in quantitative finance.  You can customize the agents in
[`agentic_quant/agents.py`](agentic_quant/agents.py) to plug in live data,
alternative alpha models, or bespoke risk overlays.

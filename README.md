# Online Learning for Mean-Variance Portfolio Selection in Adversarial Environments with Side Information

## Abstract

Portfolio management relies heavily on Markowitz's mean-variance optimization based on predictive signals about asset returns. However, this approach suffers from a lack of robustness during adversarial periods such as stock market crashes because predictions of returns and covariance become ineffective. Conversely, online learning algorithms offer performance guarantees in the worst cases, but rarely exploit predictive signals. This project seeks to reconcile these two approaches through the framework of Optimistic Online Convex Optimization. We introduce online portfolio selection algorithms that incorporate financial forecasts as hints to accelerate learning in calm periods while maintaining theoretical robustness in adversarial conditions. We establish new regret bounds that explicitly model the transition from a predictable regime to a market crisis. Empirically, our experiments on real market data demonstrate that optimistic strategies and meta-algorithms combining online learning and Markowitz portfolios outperform the Markowitz benchmark in terms of resilience to stock market crashes (Calmar ratio) and Sharpe ratio. Finally, we highlight the sensitivity of these approaches to the learning rate tuning, identifying a trade-off between theoretical performance and practical feasibility.

## Project Overview

This repository contains the implementation and analysis of online portfolio selection algorithms designed to be robust to regime shifts (e.g., market crashes). The core objective is to maximize the risk-adjusted return (Sharpe Ratio) while minimizing the Maximum Drawdown during adversarial periods.

The framework utilizes Optimistic Follow-The-Regularized-Leader (OFTRL) algorithms, treating financial forecasts (returns and covariance) as "hints" to guide the optimization process.


## Reproductibility and repository structure

To recreate the results of the experiments: 
- The notebook containing the code producing the results is available under the name "notebook.ipynb". The results of the experiments can be obtained by running this notebook along with the dataset "oxfordmanrealizedvolatilityindices.csv" and the "functions.py" file. 
- The code is fully made by myself, the various algorithms that are implemented (OGD, OOGD, EG, OEG, ...) are based on the update formulas given in the paper in sections 2.3, 2.4 and 2.5. I used LLMs to debug my code. 


A detailed analysis and new regret bounds for our optimistic algorithms are available in the "Online Learning for portfolio selection project report.pdf" file.


## Algorithms implemented

The list of algorithms that are implemented and compared is given below: 

1. Benchmarks

* Markowitz Rolling: Classical Mean-Variance optimization using a sliding window for return and covariance forecasts.
* OGD Standard: Online Gradient Descent with adaptive learning rates based on past gradients.
* EG Standard: Exponentiated Gradient with adaptive learning rates.

2. Optimistic Strategies (Proposed Approach)

* OGD Opt (Grad LR): Optimistic OGD with learning rate based on gradient volatility.
* EG Opt (Grad LR): Optimistic EG with learning rate based on gradient volatility.
* OGD Opt (Pred LR): Optimistic OGD with learning rate based on prediction error.
* EG Opt (Pred LR): Optimistic EG with learning rate based on prediction error.

3. Meta-Algorithms (Ensemble Online Learning based on the exponential weights algorithm)

* Meta All Experts: Aggregation of all available strategies.
* Meta Marko + EG Std: Aggregation of Markowitz and Standard EG.
* Meta Marko + OGD Std: Aggregation of Markowitz and Standard OGD.

## Dataset

The experiments use historical returns and intraday realized volatility from the Oxford-Man Institute's Realised Library.

* Assets: 22 major stock market indices (tracked via ETFs in practice).
* Period: 2002 to 2018.
* Features: Daily realized returns and realized covariance matrices estimated from 5-minute intraday data.

## Key Results

Performance metrics evaluated include the Annualized Sharpe Ratio, Maximum Drawdown, and Calmar Ratio.

1. Robustness: Optimistic strategies demonstrated superior resilience during market crashes. For example, Optimistic EG achieved a Calmar Ratio of 0.38 compared to 0.20 for the Markowitz benchmark, effectively nearly doubling the return per unit of tail risk.
2. Efficiency: The Meta-Algorithm combining all experts achieved the highest efficiency with an Annualized Sharpe Ratio of 1.10, successfully smoothing out the volatility of individual experts.
3. Sensitivity Analysis: The project highlights a critical trade-off in hyperparameter tuning. Optimistic strategies are highly sensitive to the learning rate. We observed that certain configurations lead to a "barcode effect" (binary switching between assets), which creates high turnover and practical implementation challenges .
 

## Requirements

To run the code, the following Python libraries are required:

* numpy
* pandas
* matplotlib
* scipy 

## References

[1] Thomas M Cover. Universal portfolios. Mathematical finance, 1991.
[2] David P Helmbold, Robert E Schapire, Yoram Singer, and Manfred K Warmuth. On-line portfolio
selection using multiplicative updates. Mathematical Finance, 1998.
[3] Amit Agarwal, Elad Hazan, Satyen Kale, and Robert E Schapire. Algorithms for portfolio management.
In Proceedings of the 23rd International Conference on Machine Learning, 2006.
[4] Bin Li and Steven CH Hoi. Online portfolio selection: A survey. ACM Computing Surveys (CSUR),
2014.
[5] Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends in
Machine Learning, 4(2):107–194, 2011.
[6] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends in Optimization,
2(3-4):157–325, 2016.
[7] Harry Markowitz. Portfolio selection. The Journal of Finance, pages 77–91, 1952.
[8] Alexander Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences.
In Advances in Neural Information Processing Systems, volume 26, 2013.
[9] Francesco Orabona. A Modern Introduction to Online Learning. 2023.
[10] Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized
learning in games. In Advances in Neural Information Processing Systems, volume 28, 2015.
[11] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge University Press,
2006.

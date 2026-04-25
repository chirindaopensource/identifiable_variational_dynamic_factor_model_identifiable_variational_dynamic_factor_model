# **`README.md`**

# Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.22886-b31b1b.svg)](https://arxiv.org/abs/2603.22886)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2603.22886)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)
[![Discipline](https://img.shields.io/badge/Discipline-Structural%20Econometrics%20%7C%20Deep%20Probabilistic%20Programming-00529B)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)
[![Data Sources](https://img.shields.io/badge/Data-ETT%20%7C%20Weather%20%7C%20Synthetic%20SCMs-lightgrey)](https://github.com/thuml/Time-Series-Library)
[![Core Method](https://img.shields.io/badge/Method-iVDFM%20%7C%20iVAE%20Conditioning-orange)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)
[![Analysis](https://img.shields.io/badge/Analysis-Variational%20Inference%20%7C%20AR(p)%20Dynamics-red)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)
[![Validation](https://img.shields.io/badge/Validation-MCC%20%7C%20IRF--MSE%20%7C%20CRPS-green)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)
[![Robustness](https://img.shields.io/badge/Robustness-Laplace%20Prior%20%7C%20Diagonal%20Restriction-yellow)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Optuna](https://img.shields.io/badge/Optuna-%2325292E.svg?style=flat)](https://optuna.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model)

**Repository:** `https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics"** by:

*   **Minkey Chang**
*   **Jae-Young Kim**

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, highly optimized pipeline that executes the entire research workflow: from the rigorous ingestion of high-dimensional multivariate time series and the deterministic construction of regime embeddings, to the execution of amortized variational inference and the unrolling of linear diagonal dynamics. The pipeline culminates in the evaluation of factor recovery, causal intervenability via Impulse Response Functions (IRFs), and probabilistic forecasting against state-of-the-art baselines.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_full_ivdfm_research_suite`](#key-callable-execute_full_ivdfm_research_suite)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Chang & Kim (2026). The core of this repository is the iPython Notebook `identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model_draft.ipynb`, which contains a comprehensive suite of orchestrated tasks to replicate the paper's findings.

The pipeline addresses a foundational vulnerability in classical financial econometrics and machine learning: the **rotational indeterminacy** of latent temporal representations. Classical Dynamic Factor Models (DFMs) are identifiable only up to orthogonal rotations, rendering the recovered factors semantically and causally ambiguous. 

The codebase operationalizes the proposed solution—the **Identifiable Variational Dynamic Factor Model (iVDFM)**:
-   **Anchors** identifiability at the level of the stochastic innovation process ($\eta_t$) by applying iVAE-style conditioning on auxiliary variables ($u_t$) and deterministic regime embeddings ($e_t$).
-   **Enforces** a non-Gaussian (Laplace) exponential-family prior, mathematically circumventing the degeneracy inherent to Gaussian families.
-   **Propagates** the identified innovations through linear, strictly diagonal dynamics (including $AR(p)$ companion forms) to prevent cross-factor mixing and preserve the equivalence class (permutation and component-wise affine maps).
-   **Evaluates** the representation's utility through rigorous factor recovery metrics (MCC), causal do-interventions (IRF-MSE), and probabilistic forecasting (CRPS).

## Theoretical Background

The implemented methods combine techniques from Information Geometry, Structural Econometrics, and Deep Probabilistic Generative Modeling.

**1. The Degeneracy of Gaussian Innovations:**
In the Gaussian case, any invertible linear map preserves the quadratic form of the log-density. Thus, even with auxiliary conditioning, Gaussian models remain identifiable only up to an invertible linear transformation. iVDFM overcomes this by utilizing a Laplace prior:
$$ p(\eta_t | u_t, e_t) = \prod_{i=1}^r h_i(\eta_{i,t}) \exp \left( T_i(\eta_{i,t})^\top \lambda_i(u_t, e_t) - A_i(\lambda_i(u_t, e_t)) \right) $$

**2. Linear Diagonal Dynamics:**
To prevent the reintroduction of rotational ambiguity during temporal evolution, the transition matrices must be strictly diagonal. This ensures each factor evolves solely from its own lag and unique innovation component:
$$ f_{t+1} = \bar{A}_t f_t + \bar{B}_t \eta_t $$

**3. Amortized Variational Inference (ELBO):**
The model is trained by maximizing the Evidence Lower Bound, balancing the expected reconstruction log-likelihood against the KL divergence between the Gaussian posterior and the non-Gaussian prior:
$$ \mathcal{L} = \mathbb{E}_{q_\phi} \left[ \sum_{t=1}^T \log p_\theta(y_t | f_t) \right] - \sum_{t=1}^T \text{KL}(q_\phi(\eta_t | \cdot) \parallel p_\theta(\eta_t | u_t, e_t)) $$

**4. Causal Intervention via the Do-Operator:**
The intervenability of the learned representation is tested by applying a shock $c$ to a specific innovation component $k$ at time $t_0$, and measuring the resulting Impulse Response Function (IRF):
$$ \text{do}(\eta_{t_0}^{(k)} = \eta_{\text{baseline}, t_0}^{(k)} + c) $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model/blob/main/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model_ipo_main.png" alt="iVDFM Architecture" width="100%">
</div>

## Features

The provided iPython Notebook implements the full research pipeline, including:

-   **Strict Identifiability Guards:** The pipeline utilizes a rigorous configuration validation gate that mathematically enforces Theorem A.2 assumptions, including the $r \le N$ topological constraint, the fixed observation noise variance ($\sigma^2$), and the non-Gaussian prior requirement.
-   **Deterministic Regime Embeddings:** Implements a `RegimeNet` that maps auxiliary variables to a probability simplex, computing a convex combination of learnable regime vectors to provide non-stationary context without stochastic sampling.
-   **Companion Form AR(p) Dynamics:** Supports higher-order temporal dependencies by augmenting the state vector while maintaining strict block-diagonal integrity to prevent cross-factor mixing.
-   **Causal Intervention Engine:** Features a dedicated module for executing do-calculus interventions at the innovation level, computing model-implied IRFs, and evaluating Sign Accuracy against ground-truth Structural Causal Models (SCMs).
-   **Comprehensive Ablation Framework:** Automates the execution of sensitivity checks and identifiability ablations (e.g., Laplace vs. Gaussian prior, Varying vs. Constant context) to empirically validate the theoretical proofs.
-   **Configuration-Driven Design:** All study parameters, architectural topologies, and optimization constraints are managed in an external `config.yaml` file, ensuring strict methodological reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preprocessing (Tasks 1-5):** Ingests raw multivariate time series. Enforces strict temporal monotonicity, standardizes observations to prevent look-ahead bias, and defines the observation vector $\mathbf{y}_t$.
2.  **Contextual Synthesis (Tasks 6-7):** Deterministically maps the temporal index to geometric features ($u_t$) and passes them through `RegimeNet` to establish the context embedding ($e_t$).
3.  **Amortized Inference & Dynamics (Tasks 8-11):** The Variational Encoder samples innovations $\eta_t$ using the reparameterization trick. These are propagated through the `CompanionARpDynamics` using strictly diagonal matrices.
4.  **Optimization (Task 12):** Trains the model via Algorithm 1, maximizing the ELBO using a Monte Carlo estimate for the KL divergence and applying gradient clipping for recurrent stability.
5.  **Evaluation (Tasks 13-16):** Aligns recovered factors using max-weight bipartite matching (MCC), computes causal IRFs, and generates Monte Carlo samples for probabilistic forecasting (CRPS) against SOTA baselines (iTransformer, TimeMixer).

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 18+ major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `execute_full_ivdfm_research_suite`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_full_ivdfm_research_suite`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data validation, amortized inference, structural dynamic unrolling, causal intervention, probabilistic forecasting, and the final robustness ablations.

## Prerequisites

-   Python 3.10+
-   Core Python dependencies: `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `optuna`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model.git
    cd identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install torch numpy pandas scipy scikit-learn optuna pyyaml
    ```

## Input Data Structure

The pipeline requires two primary data structures, strictly validated at runtime:

1.  **`df_ett` (Dict[str, pd.DataFrame]):** A mapping of the four ETT benchmark datasets (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`). Each DataFrame must contain a strictly monotonic `date` column (`datetime64[ns]`) and continuous floating-point observation columns (e.g., `HUFL`, `OT`).
2.  **`df_weather` (pd.DataFrame):** A singular DataFrame for the Weather benchmark, adhering to the same strict schema requirements.

*Note: The pipeline includes a high-fidelity synthetic data generator that simulates stable VAR(1) processes with Laplace innovations for testing purposes.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic benchmark data, and use the top-level orchestrator to execute the pipeline:

```python
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any

# 1. Load the master configuration from the YAML file.
# (Assumes config.yaml is in the working directory)
def load_ivdfm_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath)}.")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CRITICAL ERROR: {filepath} not found.")
    try:
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded iVDFM configuration from {filepath}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        raise

study_config = load_ivdfm_configuration("config.yaml")

# 2. Load raw datasets (Example using synthetic generator provided in the notebook)
# In production, load from CSV/Parquet: pd.read_csv("data/ETTh1.csv")
df_ett_mock, df_weather_mock = generate_synthetic_benchmark_data(n_steps=4000)

# 3. Execute the entire replication study.
if __name__ == "__main__":
    if df_ett_mock and not df_weather_mock.empty and study_config:
        print("\nInitiating Identifiable Variational Dynamic Factor Model (iVDFM) Suite...")
        
        suite_artifacts = execute_full_ivdfm_research_suite(
            df_ett=df_ett_mock,
            df_weather=df_weather_mock,
            config=study_config,
            output_dir="./ivdfm_output",
            dry_run=False
        )
        
        # 4. Access results
        if suite_artifacts["global_status"] == "success":
            print("\n" + "="*80)
            print("STUDY EXECUTION COMPLETE: ARTIFACTS SECURED")
            print("="*80)
            
            main_results = suite_artifacts["main_pipeline_results"]["artifacts"]
            print(f"\nFactor Recovery (Table 1) saved to: {main_results.get('table1')}")
            print(f"Causal Intervention (Table 2) saved to: {main_results.get('table2')}")
            print(f"Probabilistic Forecasting (Table 3) saved to: {main_results.get('table3')}")
            
            robustness = suite_artifacts["robustness_analysis_results"]
            if "Ablations" in robustness:
                print("\nIdentifiability Ablations (Laplace vs. Gaussian):")
                print(robustness["Ablations"])
```

## Output Structure

The pipeline returns a master dictionary and serializes artifacts to an `output/{run_id}/` directory containing:
-   **`config/`**: The frozen configuration and the cryptographic provenance manifest.
-   **`data/`**: Cleansed DataFrames, split registries, scaler parameters, and generated SCMs.
-   **`models/`**: Trained PyTorch checkpoints for the iVDFM and all baselines.
-   **`predictions/`**: Stored forecasts, recovered factors, and model-implied IRFs.
-   **`metrics/`**: The final CSV reproductions of Table 1, Table 2, Table 3, and Table 5.
-   **`robustness/`**: JSON and CSV files detailing the stress tests, identifiability ablations, and sensitivity checks.

## Project Structure

```
identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model/
│
├── identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model_draft.ipynb     	# Main implementation notebook
├── config.yaml																									# Master configuration file
├── requirements.txt																								# Python package dependencies
│
├── LICENSE																										# MIT Project License File
└── README.md																										# This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Identifiability Guards:** Toggle the `innovation_prior_family` (Laplace vs. Gaussian) to test degeneracy.
-   **Structural Dynamics:** Adjust the `latent_dim_r` or the AR lag depth `order_p`.
-   **Neural Architecture:** Modify the `hidden_units`, `layers`, or `dropout` for the Encoder, Decoder, and PriorNet.
-   **Evaluation Protocols:** Change the `window_size`, `training_stride`, or the `horizons_H` for probabilistic forecasting.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required.

## Recommended Extensions

Future extensions, as suggested by the authors, could include:
-   **Learned Context Embeddings:** Exploring task-specific learned embeddings for context rather than deterministic time-based features.
-   **Relaxing Diagonal Constraints:** Investigating methods to relax the strict diagonal dynamics constraint while preserving identifiability (e.g., block-lower triangular structures).
-   **Higher-Dimensional Modalities:** Extending the framework to handle video or language sequences where the observation space is vastly larger and more complex.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{chang2026conditionally,
  title={Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics},
  author={Chang, Minkey and Kim, Jae-Young},
  journal={arXiv preprint arXiv:2603.22886},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics.
GitHub repository: https://github.com/chirindaopensource/identifiable_variational_dynamic_factor_model_identifiable_variational_dynamic_factor_model
```

## Acknowledgments

-   Credit to **Minkey Chang and Jae-Young Kim** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **PyTorch**, **NumPy**, **Pandas**, and **Optuna** contributors.

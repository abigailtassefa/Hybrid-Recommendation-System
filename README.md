# Hybrid Recommendation System Benchmark

A Python-based machine learning pipeline that implements and compares collaborative filtering, content-based, and hybrid recommendation system architectures using synthetic user interaction data.

---

## Overview

This project provides an end-to-end environment for developing, evaluation-testing, and benchmarking multiple recommendation approaches. It features data ingestion, synthetic user profile generation, NLP-based text preprocessing, and side-by-side evaluation metrics comparing collaborative, content-based, and combined hybrid recommendation techniques.

---

## Key Features

* **Synthetic Data Processing:** Includes scripts to download baseline dataset structures, preprocess text attributes, and generate synthetic user interaction matrices.
* **Collaborative Filtering Model:** Implements matrix factorization/similarity-based user-item interaction modeling.
* **Content-Based Model:** Uses textual feature preprocessing and item embeddings to make item-to-item recommendations based on metadata.
* **Hybrid Architecture:** Combines content similarity and collaborative ratings into a unified recommendation score.
* **Comparative Evaluation:** Includes a dedicated benchmark script (`compare_models.py`) to systematically compare accuracy metrics (e.g., RMSE, precision@k) across models.

---

## Repository Structure

```text
4th_training/
├── download_data.py                    # Downloads baseline dataset/resources
├── generate_synthetic_users.py         # Generates synthetic user profiles & interactions
├── preprocess_text.py                  # Text cleaning and embedding pipeline
├── create_train_test_split.py          # Splits interaction data into training and test sets
├── collaborative_filtering_model (2).py# Collaborative filtering implementation
├── content_based_model (1).py          # Content-based filtering implementation
├── hybrid_model.py                     # Hybrid recommendation architecture combining both
├── compare_models.py                   # Performance evaluation and cross-model comparison
└── synthetic_user_interactions.csv    # Sample dataset of synthetic user-item interactions

```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with standard data science libraries:

```bash
pip install pandas numpy scikit-learn

```

### Quickstart

1. **Clone the repository:**
```bash
git clone https://github.com/abigailtassefa/4th_training.git
cd 4th_training

```


2. **Generate synthetic data & split dataset:**
```bash
python generate_synthetic_users.py
python create_train_test_split.py

```


3. **Run model evaluation and comparison:**
```bash
python compare_models.py

```

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------
# 1. LOAD PREDICTIONS
# ----------------------------
print("Loading predictions from all three models...")

try:
    cb_df = pd.read_csv("predictions_content_based.csv")
    cf_df = pd.read_csv("predictions_collaborative.csv")
    hy_df = pd.read_csv("predictions_hybrid.csv")
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Missing prediction file: {e.filename}. "
        "Please ensure you run content_based_model.py, "
        "collaborative_filtering_model.py, and hybrid_model.py first!"
    )

# ----------------------------
# 2. DEFINE EVALUATION METRICS
# ----------------------------
def calculate_metrics(df, pred_col="predicted_score", true_col="true_rating"):
    """Calculate RMSE and MAE."""
    rmse = np.sqrt(mean_squared_error(df[true_col], df[pred_col]))
    mae = mean_absolute_error(df[true_col], df[pred_col])
    return rmse, mae

def calculate_precision_recall_at_k(df, k=10, threshold=7.0, pred_col="predicted_score", true_col="true_rating"):
    """Calculate Precision@K and Recall@K for rating predictions."""
    precisions = []
    recalls = []
    
    # Evaluate performance for each user independently
    for user_id, group in df.groupby("synthetic_user_id"):
        # Sort their predicted ratings from highest to lowest
        group_sorted = group.sort_values(pred_col, ascending=False)
        top_k = group_sorted.head(k)
        
        # Count target relevant recommendations
        n_rel = (group[true_col] >= threshold).sum()
        n_rec_k = (top_k[pred_col] >= threshold).sum()
        n_rel_and_rec_k = ((top_k[true_col] >= threshold) & (top_k[pred_col] >= threshold)).sum()
        
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)
        
    return np.mean(precisions), np.mean(recalls)

# ----------------------------
# 3. RUN METRIC EVALUATIONS
# ----------------------------
results = []

for model_name, df in [
    ("Content-Based (MiniLM)", cb_df),
    ("Collaborative (SVD)", cf_df),
    ("Hybrid Recommender", hy_df)
]:
    # Determine the correct prediction column name for each file
    pred_col = "predicted_score"
    
    rmse, mae = calculate_metrics(df, pred_col=pred_col)
    precision, recall = calculate_precision_recall_at_k(df, k=10, threshold=7.0, pred_col=pred_col)
    
    results.append({
        "Model Name": model_name,
        "RMSE (Lower=Better)": rmse,
        "MAE (Lower=Better)": mae,
        "Precision@10": precision,
        "Recall@10": recall
    })

# ----------------------------
# 4. PRINT RESULTS TABLE
# ----------------------------
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print(f"{'RECOMMENDER PERFORMANCE COMPARISON':^80}")
print("="*80)
print(results_df.to_string(index=False, formatters={
    "RMSE (Lower=Better)": "{:.4f}".format,
    "MAE (Lower=Better)": "{:.4f}".format,
    "Precision@10": "{:.4f}".format,
    "Recall@10": "{:.4f}".format
}))
print("="*80)
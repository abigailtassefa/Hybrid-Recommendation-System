import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------
# 1. LOAD MODEL PREDICTIONS
# ----------------------------
print("Loading individual model predictions...")

try:
    cb_df = pd.read_csv("predictions_content_based.csv")
except FileNotFoundError:
    raise FileNotFoundError("Could not find predictions_content_based.csv. Please run content_based_model.py first!")

try:
    cf_df = pd.read_csv("predictions_collaborative.csv")
except FileNotFoundError:
    raise FileNotFoundError("Could not find predictions_collaborative.csv. Please run your collaborative filtering code first!")

# ----------------------------
# 1b. HELPER: PRECISION@K / RECALL@K (needed to compute each model's own precision)
# ----------------------------
def precision_recall_at_k(df, k=10, threshold=7.0, pred_col="predicted_score", true_col="true_rating"):
    precisions, recalls = [], []
    for user_id, group in df.groupby("synthetic_user_id"):
        group_sorted = group.sort_values(pred_col, ascending=False)
        top_k = group_sorted.head(k)

        n_rel = (group[true_col] >= threshold).sum()
        n_rec_k = (top_k[pred_col] >= threshold).sum()
        n_rel_and_rec_k = ((top_k[true_col] >= threshold) & (top_k[pred_col] >= threshold)).sum()

        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

    return np.mean(precisions), np.mean(recalls)

# ----------------------------
# 1c. COMPUTE EACH MODEL'S OWN RMSE AND PRECISION@10 (evidence for blended weighting)
# ----------------------------
cb_rmse = np.sqrt(mean_squared_error(cb_df["true_rating"], cb_df["predicted_score"]))
cf_rmse = np.sqrt(mean_squared_error(cf_df["true_rating"], cf_df["predicted_score"]))

cb_precision, _ = precision_recall_at_k(cb_df, k=10, threshold=7.0)
cf_precision, _ = precision_recall_at_k(cf_df, k=10, threshold=7.0)

print(f"Content-Based  -> RMSE: {cb_rmse:.4f}, Precision@10: {cb_precision:.4f}")
print(f"Collaborative  -> RMSE: {cf_rmse:.4f}, Precision@10: {cf_precision:.4f}")

# ----------------------------
# 2. MERGE PREDICTIONS BY USER AND DRUG
# ----------------------------
merged_df = pd.merge(
    cb_df, 
    cf_df, 
    on=["synthetic_user_id", "drugName", "true_rating"], 
    suffixes=("_cb", "_cf")
)

print(f"Successfully matched {len(merged_df)} test predictions for hybrid blending.")

# ----------------------------
# 3. COMBINE PREDICTIONS (HYBRID BLENDING) — blended accuracy + relevance weights
# ----------------------------
# Weight #1: accuracy-based (lower RMSE = more trusted)
rmse_cb_weight = (1 / cb_rmse) / ((1 / cb_rmse) + (1 / cf_rmse))
rmse_cf_weight = 1 - rmse_cb_weight

# Weight #2: relevance-based (higher Precision@10 = more trusted)
precision_cb_weight = cb_precision / (cb_precision + cf_precision)
precision_cf_weight = 1 - precision_cb_weight

# Final weight = average of the two perspectives, so the hybrid balances
# both rating accuracy AND recommendation relevance, instead of picking one
CB_WEIGHT = (rmse_cb_weight + precision_cb_weight) / 2
CF_WEIGHT = 1 - CB_WEIGHT

print(f"RMSE-based split:       {rmse_cb_weight*100:.1f}% CB / {rmse_cf_weight*100:.1f}% CF")
print(f"Precision-based split:  {precision_cb_weight*100:.1f}% CB / {precision_cf_weight*100:.1f}% CF")
print(f"Final blended split:    {CB_WEIGHT*100:.1f}% CB / {CF_WEIGHT*100:.1f}% CF")

# Calculate the final blended hybrid prediction
merged_df["predicted_score"] = (CB_WEIGHT * merged_df["predicted_score_cb"]) + (CF_WEIGHT * merged_df["predicted_score_cf"])

# Ensure predictions stay strictly within the valid clinical rating boundaries of 1.0 to 10.0
merged_df["predicted_score"] = np.clip(merged_df["predicted_score"], 1.0, 10.0)

# Save hybrid predictions to a CSV for final reporting
merged_df[["synthetic_user_id", "drugName", "true_rating", "predicted_score"]].to_csv("predictions_hybrid.csv", index=False)
print("Saved hybrid predictions to predictions_hybrid.csv")

# ----------------------------
# 4. EVALUATE THE HYBRID MODEL
# ----------------------------
rmse = np.sqrt(mean_squared_error(merged_df["true_rating"], merged_df["predicted_score"]))
mae = mean_absolute_error(merged_df["true_rating"], merged_df["predicted_score"])
precision, recall = precision_recall_at_k(merged_df, k=10, threshold=7.0)

print("\n================ HYBRID MODEL PERFORMANCE ================")
print(f"Hybrid Model RMSE:        {rmse:.4f}")
print(f"Hybrid Model MAE:         {mae:.4f}")
print(f"Hybrid Model Precision@10:{precision:.4f}")
print(f"Hybrid Model Recall@10:   {recall:.4f}")
print("==========================================================")
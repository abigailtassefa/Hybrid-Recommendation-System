
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy

# ----------------------------
# 1. LOAD SHARED TRAIN/TEST SPLIT
# ----------------------------
train_df = pd.read_csv("train_interactions.csv")
test_df = pd.read_csv("test_interactions.csv")

# keep the full df around too, for the top-N recommendation demo later
df = pd.concat([train_df, test_df], ignore_index=True)

print("Train set:", train_df.shape)
print("Test set:", test_df.shape)

# ----------------------------
# 2. PREP DATA FOR SURPRISE
# ----------------------------
# Surprise expects exactly 3 columns: user, item, rating
# Rating scale in this dataset is 1-10
reader = Reader(rating_scale=(1, 10))

trainset = Dataset.load_from_df(
    train_df[["synthetic_user_id", "drugName", "rating"]], reader
).build_full_trainset()

# testset needs to be a list of (user, item, true_rating) tuples
testset = list(
    test_df[["synthetic_user_id", "drugName", "rating"]].itertuples(index=False, name=None)
)

# ----------------------------
# 4. TRAIN SVD MODEL
# ----------------------------
# n_factors: number of latent dimensions learned per user/item
# n_epochs: training iterations
model = SVD(n_factors=50, n_epochs=20, random_state=42)
model.fit(trainset)

# ----------------------------
# 5. EVALUATE
# ----------------------------
predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# ----------------------------
# 6. GENERATE TOP-N RECOMMENDATIONS FOR A SAMPLE USER
# ----------------------------
def get_top_n_recommendations(model, df, user_id, n=5):
    """Recommend drugs this user hasn't rated yet, ranked by predicted rating."""
    all_drugs = df["drugName"].unique()
    rated_drugs = df[df["synthetic_user_id"] == user_id]["drugName"].unique()
    unrated_drugs = [d for d in all_drugs if d not in rated_drugs]

    predictions = [
        (drug, model.predict(user_id, drug).est)
        for drug in unrated_drugs
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

sample_user = df["synthetic_user_id"].iloc[0]
top_recs = get_top_n_recommendations(model, df, sample_user, n=5)

print(f"\nTop 5 recommendations for {sample_user}:")
for drug, predicted_rating in top_recs:
    print(f"  {drug}: predicted rating {predicted_rating:.2f}")

# ----------------------------
# 7. PRECISION@K / RECALL@K (for later comparison across the 3 approaches)
# ----------------------------
def precision_recall_at_k(predictions, k=10, threshold=7.0):
    """
    Standard precision@k / recall@k for rating-based recommenders.
    threshold: rating value considered a "relevant" / "good" recommendation
    """
    from collections import defaultdict

    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum(true_r >= threshold for (_, true_r) in user_ratings)
        n_rec_k = sum(est >= threshold for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold) and (est >= threshold)
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    return avg_precision, avg_recall

precision, recall = precision_recall_at_k(predictions, k=10, threshold=7.0)
print(f"\nPrecision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")

# ----------------------------
# 8. SAVE PREDICTIONS FOR CROSS-MODEL COMPARISON
# ----------------------------
# Shared format every model writes to, so compare_models.py can load
# all three side by side: user, item, true_rating, predicted_score
preds_df = pd.DataFrame([
    {
        "synthetic_user_id": p.uid,
        "drugName": p.iid,
        "true_rating": p.r_ui,
        "predicted_score": p.est,
    }
    for p in predictions
])
preds_df.to_csv("predictions_collaborative.csv", index=False)
print("\nSaved predictions to predictions_collaborative.csv")

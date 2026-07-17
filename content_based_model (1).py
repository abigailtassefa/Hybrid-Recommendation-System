import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# CONFIG

ORIGINAL_DATA_PATH = "drugsComTrain_raw.csv"  
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good general-purpose sentence transformer model 
REVIEWS_PER_DRUG_FOR_PROFILE = 5  # how many real reviews to sample per drug to build its text profile

# ----------------------------
# 1. LOAD DATA
# ----------------------------
raw_df = pd.read_csv(ORIGINAL_DATA_PATH)
raw_df = raw_df.dropna(subset=["drugName", "condition", "review"])

train_df = pd.read_csv("train_interactions.csv")
test_df = pd.read_csv("test_interactions.csv")

# ----------------------------
# 2. BUILD ONE TEXT PROFILE PER DRUG
# ----------------------------
# For each drug, combine its condition + a handful of real review snippets
# into one text blob to embed. This captures both "what it treats" and
# "what patients say about it."
print("Building text profiles per drug...")

drug_text_profiles = {}
for drug_name, group in raw_df.groupby("drugName"):
    condition = group["condition"].mode()[0]  # most common condition listed for this drug
    sample_reviews = group["review"].sample(
        n=min(REVIEWS_PER_DRUG_FOR_PROFILE, len(group)), random_state=42
    ).tolist()
    combined_text = f"Used for {condition}. " + " ".join(sample_reviews)
    drug_text_profiles[drug_name] = combined_text

print(f"Built profiles for {len(drug_text_profiles)} unique drugs")

# ----------------------------
# 3. EMBED ALL DRUG PROFILES
# ----------------------------
print("Loading sentence transformer model (first run downloads it, ~90MB)...")
model = SentenceTransformer(MODEL_NAME)

drug_names = list(drug_text_profiles.keys())
drug_texts = list(drug_text_profiles.values())

print("Encoding drug embeddings...")
drug_embeddings = model.encode(drug_texts, show_progress_bar=True, batch_size=64)
drug_embedding_lookup = dict(zip(drug_names, drug_embeddings))


# 4. BUILD USER PROFILES FROM TRAIN SET
# A user's profile = weighted average of the embeddings of drugs
# they rated in the train set, weighted by their rating
# (so a 10/10 rated drug contributes more than a 3/10 rated drug)

print("Building user profile embeddings from train set...")

user_profiles = {}
for user_id, group in train_df.groupby("synthetic_user_id"):
    embeddings = []
    weights = []
    for _, row in group.iterrows():
        drug = row["drugName"]
        if drug in drug_embedding_lookup:
            embeddings.append(drug_embedding_lookup[drug])
            weights.append(row["rating"])

    if len(embeddings) == 0:
        continue

    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weighted_profile = np.average(embeddings, axis=0, weights=weights)
    user_profiles[user_id] = weighted_profile

print(f"Built profiles for {len(user_profiles)} users")

# ----------------------------
# 5. PREDICT SCORES FOR TEST SET
# ----------------------------
print("Predicting scores for test interactions...")

predictions = []
skipped = 0

for _, row in test_df.iterrows():
    user_id = row["synthetic_user_id"]
    drug = row["drugName"]
    true_rating = row["rating"]

    if user_id not in user_profiles or drug not in drug_embedding_lookup:
        skipped += 1
        continue  # cold-start case: no profile or no drug embedding available

    user_vec = user_profiles[user_id].reshape(1, -1)
    drug_vec = drug_embedding_lookup[drug].reshape(1, -1)

    similarity = cosine_similarity(user_vec, drug_vec)[0][0]  # ranges -1 to 1

    # Rescale similarity (-1 to 1) into rating scale (1 to 10) for comparability
    predicted_score = 1 + (similarity + 1) * (9 / 2)

    predictions.append({
        "synthetic_user_id": user_id,
        "drugName": drug,
        "true_rating": true_rating,
        "predicted_score": predicted_score,
    })

print(f"Predicted {len(predictions)} test interactions, skipped {skipped} (cold-start)")

preds_df = pd.DataFrame(predictions)
preds_df.to_csv("predictions_content_based.csv", index=False)
print("Saved predictions to predictions_content_based.csv")

# ----------------------------
# 6. EVALUATE (same metrics as collaborative filtering, for fair comparison)
# ----------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(preds_df["true_rating"], preds_df["predicted_score"]))
mae = mean_absolute_error(preds_df["true_rating"], preds_df["predicted_score"])

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

def precision_recall_at_k(df, k=10, threshold=7.0):
    precisions, recalls = [], []
    for user_id, group in df.groupby("synthetic_user_id"):
        group_sorted = group.sort_values("predicted_score", ascending=False)
        top_k = group_sorted.head(k)

        n_rel = (group["true_rating"] >= threshold).sum()
        n_rec_k = (top_k["predicted_score"] >= threshold).sum()
        n_rel_and_rec_k = ((top_k["true_rating"] >= threshold) & (top_k["predicted_score"] >= threshold)).sum()

        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

    return np.mean(precisions), np.mean(recalls)

precision, recall = precision_recall_at_k(preds_df, k=10, threshold=7.0)
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")

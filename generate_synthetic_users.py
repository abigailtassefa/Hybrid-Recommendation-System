"""
Synthetic User Interaction Generator
-------------------------------------
Goal: Turn anonymous drug reviews (no user_id) into a realistic
user-item interaction matrix that a collaborative filtering
algorithm can use.

Input columns expected: drugName, condition, rating, review, usefulCount, date
Output columns: synthetic_user_id, drugName, condition, rating
"""

import pandas as pd
import numpy as np

# ----------------------------
# 0. CONFIG - tune these
# ----------------------------
NUM_SYNTHETIC_USERS = 800       # how many pseudo-patients to create
MIN_CONDITIONS_PER_USER = 1     # each synthetic user "has" 1-2 conditions
MAX_CONDITIONS_PER_USER = 2
MIN_REVIEWS_PER_USER = 3        # each user rates between 3-8 drugs
MAX_REVIEWS_PER_USER = 8
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

import pandas as pd


# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv(r"C:\Users\abiga\.cache\kagglehub\datasets\jessicali9530\kuc-hackathon-winter-2018\versions\2\drugsComTrain_raw.csv")

# Basic cleanup
df = df.dropna(subset=["drugName", "condition", "rating"])
df["condition"] = df["condition"].str.strip()
df["drugName"] = df["drugName"].str.strip()

# ----------------------------
# 2. BUILD CONDITION -> AVAILABLE REVIEWS LOOKUP
# ----------------------------
# For each condition, we need a pool of (drugName, rating) pairs to sample from
condition_pool = df.groupby("condition")

all_conditions = df["condition"].unique()

# ----------------------------
# 3. GENERATE SYNTHETIC USERS
# ----------------------------
synthetic_rows = []

for user_id in range(1, NUM_SYNTHETIC_USERS + 1):
    # Step A: assign this user 1-2 "conditions" they have
    n_conditions = np.random.randint(MIN_CONDITIONS_PER_USER, MAX_CONDITIONS_PER_USER + 1)
    user_conditions = np.random.choice(all_conditions, size=n_conditions, replace=False)

    # Step B: decide how many drugs this user has tried/rated overall
    n_reviews = np.random.randint(MIN_REVIEWS_PER_USER, MAX_REVIEWS_PER_USER + 1)

    # Step C: pull real reviews matching this user's condition(s)
    candidate_rows = df[df["condition"].isin(user_conditions)]

    if len(candidate_rows) == 0:
        continue  # skip if no data for this condition (rare edge case)

    # sample without replacement where possible, with replacement if pool is small
    replace_flag = len(candidate_rows) < n_reviews
    sampled = candidate_rows.sample(n=n_reviews, replace=replace_flag, random_state=user_id)

    # Step D: add noise to ratings so synthetic users aren't just carbon copies
    # of a single real review, and clip to valid 1-10 range
    for _, row in sampled.iterrows():
        
        synthetic_rows.append({
            "synthetic_user_id": f"user_{user_id}",
            "drugName": row["drugName"],
            "condition": row["condition"],
            "rating": row["rating"],
        })

# ----------------------------
# 4. BUILD FINAL INTERACTION TABLE
# ----------------------------
interactions_df = pd.DataFrame(synthetic_rows)

# Optional: drop exact duplicate (user, drug) pairs, keep first occurrence
interactions_df = interactions_df.drop_duplicates(subset=["synthetic_user_id", "drugName"])

print("Synthetic interaction matrix shape:", interactions_df.shape)
print(interactions_df.head(10))

# ----------------------------
# 5. SAVE FOR USE IN CF MODEL
# ----------------------------
interactions_df.to_csv("synthetic_user_interactions.csv", index=False)
print("Saved to synthetic_user_interactions.csv")

# ----------------------------
# 6. SANITY CHECKS (run these before trusting the data)
# ----------------------------
print("\n--- Sanity checks ---")
print("Unique synthetic users:", interactions_df["synthetic_user_id"].nunique())
print("Unique drugs rated:", interactions_df["drugName"].nunique())
print("Avg ratings per user:", interactions_df.groupby("synthetic_user_id").size().mean())
print("Avg ratings per drug:", interactions_df.groupby("drugName").size().mean())

# Density = filled cells / total possible cells in the user-item matrix
n_users = interactions_df["synthetic_user_id"].nunique()
n_items = interactions_df["drugName"].nunique()
density = len(interactions_df) / (n_users * n_items)
print(f"Matrix density: {density:.4%}  (typical real-world CF datasets: 0.1%-5%)")

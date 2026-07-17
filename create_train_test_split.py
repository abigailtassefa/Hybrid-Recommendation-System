"""
Shared Train/Test Split
-------------------------
Run this ONCE. It creates train_interactions.csv and test_interactions.csv
from synthetic_user_interactions.csv, so collaborative filtering,
content-based, and hybrid models are all evaluated on the exact same split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("synthetic_user_interactions.csv")

# Stratifying by user isn't straightforward with sklearn's train_test_split
# directly, so we do a simple random split at the interaction level.
# This means: for each user, some of their ratings go to train, some to test -
# which is exactly what we want (predict a rating for a drug the user
# "hasn't tried" using their OTHER ratings as history).
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_df.to_csv("train_interactions.csv", index=False)
test_df.to_csv("test_interactions.csv", index=False)

print("Train set:", train_df.shape)
print("Test set:", test_df.shape)

# Sanity check: make sure test set users/drugs also appear in train set
# (otherwise the model has never seen them and can't predict anything useful)
train_users = set(train_df["synthetic_user_id"])
test_users = set(test_df["synthetic_user_id"])
train_drugs = set(train_df["drugName"])
test_drugs = set(test_df["drugName"])

unseen_users = test_users - train_users
unseen_drugs = test_drugs - train_drugs

print(f"\nTest users not seen in train: {len(unseen_users)} / {len(test_users)}")
print(f"Test drugs not seen in train: {len(unseen_drugs)} / {len(test_drugs)}")
print("(Some unseen entries are expected/OK given how sparse this data is -")
print(" this is the realistic 'cold start' problem, worth mentioning in your report)")

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# --------------------------------------------------
# Configuration
# --------------------------------------------------
TRAIN_CSV = "features_csv/train_features.csv"
TEST_CSV = "features_csv/test_features.csv"
N_NEIGHBORS = 7
METRIC = "manhattan"
WEIGHTS = "distance"
N_COMPONENTS = 150  # Number of PCA components

# --------------------------------------------------
# Load CSV files
# --------------------------------------------------
print("Loading feature files...")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# --------------------------------------------------
# Split labels and features
# --------------------------------------------------
y_train = train_df["label"].values
X_train = train_df.drop(columns=["label"]).values

y_test = test_df["label"].values
X_test = test_df.drop(columns=["label"]).values

# --------------------------------------------------
# Feature normalization (critical for kNN)
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# PCA for dimensionality reduction
# --------------------------------------------------
pca = PCA(n_components=N_COMPONENTS)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# --------------------------------------------------
# k-NN classifier
# --------------------------------------------------
knn = KNeighborsClassifier(
    n_neighbors=N_NEIGHBORS,
    metric=METRIC,
    weights=WEIGHTS
)

print("Training k-NN...")
knn.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = knn.predict(X_test)

print("\nClassification Report (k-NN):")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*50)
print("k-NN Configuration")
print("="*50)
print(f"  Train CSV:      {TRAIN_CSV}")
print(f"  Test CSV:       {TEST_CSV}")
print(f"  k (neighbors):  {N_NEIGHBORS}")
print(f"  Metric:         {METRIC}")
print(f"  Scaler:         StandardScaler")
print(f"  Train shape:    {X_train.shape}")
print(f"  Test shape:     {X_test.shape}")
print(f"  Num features:   {X_train.shape[1]}")
print(f"  Train samples:  {X_train.shape[0]}")
print(f"  Test samples:   {X_test.shape[0]}")
print("="*50 + "\n")

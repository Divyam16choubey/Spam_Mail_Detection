# =============================================================================
# Naive Bayes Email Spam Classifier — Built from Scratch
# =============================================================================
# Author : Divyam Kumar Choubey
# Course : Computer Science and Engineering
#
# Description:
#   This script implements a Naive Bayes classifier entirely from scratch
#   to classify emails as Spam (1) or Not Spam (0). It uses word-frequency
#   features, Laplace smoothing, and log probabilities. No ML library
#   (e.g., sklearn) is used for classification or evaluation.
#
# Allowed Libraries: pandas, numpy, math, random (standard library)
# =============================================================================

import pandas as pd
import numpy as np
import math
import random

# ----------------------------- 1. Load Dataset -------------------------------

# Read the dataset from the CSV file
data = pd.read_csv("emails/emails.csv")

# Display the shape of the dataset to verify it loaded correctly
print(f"Dataset Shape: {data.shape}")

# ----------------------------- 2. Prepare Data -------------------------------

# Drop the first column ("Email No.") — it is an identifier, not a feature
data = data.drop(columns=["Email No."])

# Separate features (X) and labels (y)
# Features: all columns except the last one ("Prediction")
# Labels:   the last column ("Prediction")
X = data.iloc[:, :-1].values   # shape: (5172, 3000) — numpy array
y = data.iloc[:, -1].values    # shape: (5172,)       — numpy array

# Total number of samples and features
n_samples, n_features = X.shape

# ------------------- 3. Random Train / Test Split ----------------------------

# Set a random seed for reproducibility
random.seed(42)

# Create a list of all indices and shuffle them randomly
indices = list(range(n_samples))
random.shuffle(indices)

# Split: first 4500 for training, remaining 672 for testing
train_size = 4500
test_size  = n_samples - train_size   # 672

train_indices = indices[:train_size]
test_indices  = indices[train_size:]

# Create training and testing sets
X_train = X[train_indices]
y_train = y[train_indices]
X_test  = X[test_indices]
y_test  = y[test_indices]

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# ------------------- 4. Calculate Class Prior Probabilities ------------------

# Count the number of spam and not-spam emails in the training set
n_spam     = int(np.sum(y_train == 1))   # Spam emails
n_not_spam = int(np.sum(y_train == 0))   # Not Spam emails

print(f"\nSpam Training Emails: {n_spam}")
print(f"Not Spam Training Emails: {n_not_spam}")

# Prior probability: P(class) = count(class) / total training samples
prior_spam     = n_spam / train_size
prior_not_spam = n_not_spam / train_size

# ------------------- 5. Calculate Feature Likelihoods ------------------------
# For each word feature, we compute the likelihood for each class using
# the Multinomial Naive Bayes approach:
#
#   P(word_i | class) = (sum of word_i counts in class + alpha)
#                       / (total word count in class + alpha * n_features)
#
# where alpha = 1 (Laplace smoothing parameter).
# -------------------------------------------------------------------------

alpha = 1   # Laplace smoothing parameter

# Separate training data by class
X_spam     = X_train[y_train == 1]   # All spam emails
X_not_spam = X_train[y_train == 0]   # All not-spam emails

# Sum of each word's count across all emails in each class
# shape: (3000,) for each
word_count_spam     = np.sum(X_spam, axis=0)      # Total count of each word in spam
word_count_not_spam = np.sum(X_not_spam, axis=0)   # Total count of each word in not-spam

# Total word count across all features in each class
total_words_spam     = np.sum(word_count_spam)
total_words_not_spam = np.sum(word_count_not_spam)

# Apply Laplace smoothing and compute log-likelihoods for each feature
# log P(word_i | spam) and log P(word_i | not_spam)
log_likelihood_spam = np.log(
    (word_count_spam + alpha) / (total_words_spam + alpha * n_features)
)
log_likelihood_not_spam = np.log(
    (word_count_not_spam + alpha) / (total_words_not_spam + alpha * n_features)
)

# Compute log of prior probabilities
log_prior_spam     = math.log(prior_spam)
log_prior_not_spam = math.log(prior_not_spam)

# ------------------- 6. Prediction on Test Set -------------------------------
# For each test email, calculate the log-posterior for both classes:
#
#   log P(spam | email)     = log P(spam) + Σ [ x_i * log P(word_i | spam) ]
#   log P(not_spam | email) = log P(not_spam) + Σ [ x_i * log P(word_i | not_spam) ]
#
# Predict the class with the higher log-posterior score.
# -------------------------------------------------------------------------

y_pred = np.zeros(test_size, dtype=int)

for i in range(test_size):
    # Current test email's feature vector
    email = X_test[i]

    # Calculate log-posterior for each class
    # The dot product sums x_i * log P(word_i | class) over all features
    log_post_spam     = log_prior_spam     + np.dot(email, log_likelihood_spam)
    log_post_not_spam = log_prior_not_spam + np.dot(email, log_likelihood_not_spam)

    # Assign the class with the higher log-posterior
    if log_post_spam > log_post_not_spam:
        y_pred[i] = 1   # Spam
    else:
        y_pred[i] = 0   # Not Spam

# ------------------- 7. Evaluation Metrics (Manual) -------------------------
# Confusion matrix components (Spam = Positive, Not Spam = Negative):
#   TP = predicted spam & actually spam
#   TN = predicted not spam & actually not spam
#   FP = predicted spam & actually not spam
#   FN = predicted not spam & actually spam
# -------------------------------------------------------------------------

TP = 0   # True Positives
TN = 0   # True Negatives
FP = 0   # False Positives
FN = 0   # False Negatives

for i in range(test_size):
    if y_pred[i] == 1 and y_test[i] == 1:
        TP += 1
    elif y_pred[i] == 0 and y_test[i] == 0:
        TN += 1
    elif y_pred[i] == 1 and y_test[i] == 0:
        FP += 1
    elif y_pred[i] == 0 and y_test[i] == 1:
        FN += 1

# Calculate metrics manually
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# ------------------- 8. Display Results --------------------------------------

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

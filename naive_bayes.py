# Acadmic Project :- Naive Bayes Email Spam Classifier

import pandas as pd
import numpy as np
import math
import random

# 1. Load Dataset 
data = pd.read_csv("emails/emails.csv")
print(f"Dataset Shape: {data.shape}")

# 2. Seperate & Split Data 
data = data.drop(columns=["Email No."])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

n_samples, n_features = X.shape

# 3. Random selects the 4500 emails and fixed the seed to 42.
random.seed(42)

indices = list(range(n_samples))
random.shuffle(indices)

train_size = 4500
test_size = n_samples - train_size

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# 4. Calculate Class Prior Probabilities
n_spam = int(np.sum(y_train == 1))
n_not_spam = int(np.sum(y_train == 0))

print(f"\nSpam Training Emails: {n_spam}")
print(f"Not Spam Training Emails: {n_not_spam}")

prior_spam = n_spam / train_size
prior_not_spam = n_not_spam / train_size

print(f"Prior Probability P(Spam): {prior_spam:.4f} ({prior_spam*100:.2f}%)")
print(f"Prior Probability P(Not Spam): {prior_not_spam:.4f} ({prior_not_spam*100:.2f}%)")

# 5. Calculate Feature Likelihoods (Word Statistics)
alpha = 1

X_spam = X_train[y_train == 1]
X_not_spam = X_train[y_train == 0]

word_count_spam = np.sum(X_spam, axis=0)
word_count_not_spam = np.sum(X_not_spam, axis=0)

total_words_spam = np.sum(word_count_spam)
total_words_not_spam = np.sum(word_count_not_spam)

log_likelihood_spam = np.log((word_count_spam + alpha) / (total_words_spam + alpha * n_features))

log_likelihood_not_spam = np.log((word_count_not_spam + alpha) / (total_words_not_spam + alpha * n_features))

log_prior_spam = math.log(prior_spam)
log_prior_not_spam = math.log(prior_not_spam)

# 6. Prediction on Test Set
y_pred = np.zeros(test_size, dtype=int)

for i in range(test_size):
    email = X_test[i]
    log_post_spam = log_prior_spam + np.dot(email, log_likelihood_spam)
    log_post_not_spam = log_prior_not_spam + np.dot(email, log_likelihood_not_spam)

    if log_post_spam > log_post_not_spam:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# 7. Evaluation Metrics (Confusion Matrix) 
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(test_size):
    if y_pred[i] == 1 and y_test[i] == 1:
        TP += 1
    elif y_pred[i] == 0 and y_test[i] == 0:
        TN += 1
    elif y_pred[i] == 1 and y_test[i] == 0:
        FP += 1
    elif y_pred[i] == 0 and y_test[i] == 1:
        FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# 8. Showing the Results 
print(f"\nTrue Positives: {TP}")
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score: {f1_score:.4f} ({f1_score*100:.2f}%)")

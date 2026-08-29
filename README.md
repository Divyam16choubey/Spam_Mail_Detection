# Naive Bayes Email Spam Classifier

## Project Overview

This project implements a **Naive Bayes classifier from scratch in
Python** for classifying emails into two categories:

-   **0 --- Not Spam**
-   **1 --- Spam**

The implementation is based on the `emails.csv` dataset provided for the
assignment. The main goal is to understand and implement the Naive Bayes
classification process manually without using a machine-learning library
that provides the Naive Bayes algorithm.

## Dataset

The dataset is provided in the file:

``` text
emails.csv
```

According to the assignment:

-   Total emails: **5172**
-   Total columns: **3002**
-   First column: **Email No.**
-   Last column: **Prediction**
-   Feature columns: **3000**
-   `Prediction = 0`: Not Spam
-   `Prediction = 1`: Spam

The 3000 feature columns represent the most common words in the emails.
Each value represents the count of the corresponding word in that email.

### Dataset Representation

``` text
Email No. | word_1 | word_2 | ... | word_3000 | Prediction
```

For example, if a word has a value of `5` for an email, that word occurs
five times in that email.

## Objective

The objectives of this project are:

1.  Implement Naive Bayes classification from scratch.
2.  Randomly select **4500 emails for training**.
3.  Use the remaining **672 emails for testing**.
4.  Calculate class prior probabilities manually.
5.  Calculate feature likelihood probabilities manually.
6.  Apply **Laplace smoothing** to handle zero probabilities.
7.  Use **log probabilities** to improve numerical stability.
8.  Predict whether test emails are spam or not spam.
9.  Calculate:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1 Score

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Python `math` module

Pandas and NumPy are used only for basic data processing and numerical
operations.

## Restrictions

The Naive Bayes algorithm must be implemented manually.

The project does **not** use ready-made Naive Bayes implementations such
as:

``` python
from sklearn.naive_bayes import MultinomialNB
```

The following are also not used for calculating the final evaluation
metrics:

``` python
accuracy_score()
precision_score()
recall_score()
f1_score()
```

The classifier and evaluation calculations are implemented manually.

## Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes'
theorem.

For an email represented by its word features, the classifier calculates
a score for each class:

``` text
Spam
Not Spam
```

The class with the higher probability score is selected as the
prediction.

The implementation uses two main types of probabilities.

### Class Prior Probability

The prior probability represents the probability of a class before
considering the word features.

For example:

``` text
P(Spam)     = Number of Spam training emails / Total training emails

P(Not Spam) = Number of Not Spam training emails / Total training emails
```

### Feature Likelihood

The likelihood represents how strongly each word feature is associated
with a particular class.

For every word feature, probabilities are calculated separately for:

``` text
Spam
Not Spam
```

These probabilities are then used during prediction.

## Laplace Smoothing

A word may have zero occurrences in one of the classes in the training
data.

Without smoothing, this could produce a probability of zero. A zero
probability can cause the complete probability calculation to become
zero.

To avoid this problem, Laplace smoothing is applied.

The general idea is:

``` text
smoothed count = count + alpha
```

where `alpha` is the smoothing parameter.

This prevents zero probabilities.

## Log Probabilities

The dataset contains 3000 word features. Directly multiplying many small
probabilities can result in extremely small numbers and numerical
underflow.

Instead of calculating:

``` text
P1 × P2 × P3 × ... × Pn
```

the implementation uses logarithms:

``` text
log(P1) + log(P2) + log(P3) + ... + log(Pn)
```

This is mathematically equivalent for comparing the probabilities while
being numerically more stable.

The final prediction is based on the larger log-probability score.

## Train-Test Split

The dataset contains:

``` text
5172 emails
```

The required split is:

``` text
Training: 4500 emails
Testing:   672 emails
```

The 4500 training emails are selected randomly, while the remaining 672
emails are reserved for testing.

The test data is not used while calculating the model parameters.

## Project Workflow

``` text
                    emails.csv
                        |
                        v
                  Load Dataset
                        |
                        v
             Separate Features/Labels
                        |
                        v
              Random Train/Test Split
                   /            \
                  /              \
                 v                v
        4500 Training        672 Testing
                |
                v
        Calculate Class Priors
                |
                v
      Calculate Feature Statistics
                |
                v
         Apply Laplace Smoothing
                |
                v
       Calculate Log Probabilities
                |
                v
          Predict Test Emails
                |
                v
       Compare Actual/Predicted
                |
                v
       Calculate Evaluation Metrics
                |
                v
     Accuracy / Precision / Recall / F1
```

## Evaluation Metrics

The classifier is evaluated using four metrics.

### Accuracy

Accuracy measures the percentage of correctly classified emails.

``` text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

Precision measures how many emails predicted as spam are actually spam.

``` text
Precision = TP / (TP + FP)
```

### Recall

Recall measures how many actual spam emails are correctly identified.

``` text
Recall = TP / (TP + FN)
```

### F1 Score

F1 Score is the harmonic mean of precision and recall.

``` text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:

``` text
TP = True Positive
TN = True Negative
FP = False Positive
FN = False Negative
```

## Project Structure

``` text
Naive-Bayes-Email-Classifier/
│
├── emails.csv
├── naive_bayes.py
└── README.md
```

### `emails.csv`

Contains the email dataset used for training and testing.

### `naive_bayes.py`

Contains the complete Naive Bayes implementation from scratch.

### `README.md`

Contains information about the project, dataset, methodology,
implementation approach, and evaluation.

## Implementation Plan

The implementation will be completed in the following stages:

1.  Load `emails.csv`.
2.  Verify the dataset dimensions and columns.
3.  Separate the 3000 word-count features from the labels.
4.  Randomly select 4500 emails for training.
5.  Use the remaining 672 emails for testing.
6.  Calculate the prior probability of each class.
7.  Calculate word-frequency statistics for each class.
8.  Apply Laplace smoothing.
9.  Convert probabilities into log probabilities.
10. Implement the prediction logic.
11. Predict the labels of the 672 test emails.
12. Calculate the confusion matrix values.
13. Calculate accuracy.
14. Calculate precision.
15. Calculate recall.
16. Calculate F1 score.
17. Verify the results.

## Expected Output

After the implementation is completed, the program will display results
similar to:

``` text
Dataset Shape: (5172, 3002)

Training Samples: 4500
Testing Samples: 672

Spam Training Emails: ...
Not Spam Training Emails: ...

Accuracy: ...
Precision: ...
Recall: ...
F1 Score: ...
```

The actual values will be added after the model has been implemented and
tested.

## Learning Outcomes

This project is intended to provide practical understanding of:

-   Bayes' theorem
-   Naive Bayes classification
-   Class prior probability
-   Feature likelihood
-   Word-frequency based classification
-   Laplace smoothing
-   Log probabilities
-   Numerical underflow
-   Binary classification
-   Confusion matrix
-   Accuracy
-   Precision
-   Recall
-   F1 Score

## Author

**Divyam Kumar Choubey**

Computer Science and Engineering

# Naive Bayes Email Spam Classifier

An academic machine learning project implementing a **Multinomial Naive Bayes Classifier** in Python to classify emails as **Spam (`1`)** or **Not Spam (`0`)**.

---

## Project Overview

Spam email filtering is a fundamental natural language processing and binary classification task. This project explores the inner workings of probabilistic classification by building a Naive Bayes classifier without relying on high-level machine learning libraries such as `scikit-learn`.

### Key Highlights
- **From Scratch:** Zero machine learning libraries used for both model building and performance evaluation.
- **Multinomial Feature Likelihoods:** Word frequency-based likelihood calculation tailored for bag-of-words text representations.
- **Laplace Smoothing ($\alpha = 1$):** Handles unseen or zero-frequency vocabulary items without mathematical divergence.
- **Log Probability Computation:** Prevents arithmetic underflow caused by multiplying thousands of small conditional probabilities.
- **Manual Evaluation:** Custom implementations of Confusion Matrix (TP, TN, FP, FN), Accuracy, Precision, Recall, and F1 Score.

---

## Dataset Description

The dataset is stored in `emails/emails.csv` and contains word-frequency statistics extracted from a collection of emails.

| Property | Value |
| :--- | :--- |
| **Total Emails (Samples)** | 5,172 |
| **Total Columns** | 3,002 |
| **Identifier Column** | `Email No.` (First column, excluded from features) |
| **Feature Columns** | 3,000 common vocabulary word count attributes |
| **Target Column** | `Prediction` (Last column: `0` = Not Spam, `1` = Spam) |

### Dataset Representation

```text
+-----------+--------+--------+-----+-----------+------------+
| Email No. | word_1 | word_2 | ... | word_3000 | Prediction |
+-----------+--------+--------+-----+-----------+------------+
|  Email 1  |   0    |   2    | ... |     1     |     0      |
|  Email 2  |   8    |  13    | ... |     0     |     1      |
+-----------+--------+--------+-----+-----------+------------+
```

---

## Mathematical Formulation

### 1. Bayes' Theorem for Classification

Given an email vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ where $x_i$ is the count of word $i$, the posterior probability for class $C_k \in \{\text{Spam}, \text{Not Spam}\}$ is:

$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) \cdot P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}$$

Under the Naive Bayes conditional independence assumption:

$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^{n} P(w_i \mid C_k)^{x_i}$$

---

### 2. Class Prior Probabilities

Calculated directly from the distribution of classes in the training partition:

$$P(\text{Spam}) = \frac{N_{\text{Spam}}}{N_{\text{Total}}}, \quad P(\text{Not Spam}) = \frac{N_{\text{Not Spam}}}{N_{\text{Total}}}$$

---

### 3. Feature Likelihood with Laplace Smoothing

To eliminate zero probabilities for words that do not appear in a specific class within the training split, **Add-1 (Laplace) Smoothing** is applied:

$$P(w_i \mid C_k) = \frac{\sum_{j \in C_k} x_{ji} + \alpha}{\sum_{i=1}^{n} \sum_{j \in C_k} x_{ji} + \alpha \cdot |V|}$$

Where:
- $\sum_{j \in C_k} x_{ji}$ = Total count of word $i$ in class $C_k$.
- $\alpha = 1$ = Laplace smoothing parameter.
- $|V| = 3000$ = Vocabulary size (number of feature words).

---

### 4. Log-Probability Formulation

Multiplying 3,000 fractional probabilities causes floating-point underflow. Taking the natural logarithm converts the product into a stable summation:

$$\ln P(C_k \mid \mathbf{x}) \propto \ln P(C_k) + \sum_{i=1}^{n} x_i \cdot \ln P(w_i \mid C_k)$$

The predicted class $\hat{y}$ is selected using the `argmax`:

$$\hat{y} = \arg\max_{C_k} \left( \ln P(C_k) + \sum_{i=1}^{n} x_i \cdot \ln P(w_i \mid C_k) \right)$$

---

## Project Workflow

```text
                      [ emails.csv ] (5172, 3002)
                              │
                              ▼
                     [ Data Preprocessing ]
               (Drop 'Email No.', Split X and y)
                              │
                              ▼
                   [ Random Train/Test Split ]
                  (Random Seed = 42 for parity)
                     ┌────────┴────────┐
                     ▼                 ▼
             Training Set (4500)   Test Set (672)
                     │                  │
                     ▼                  │
          [ Parameter Estimation ]      │
          - Class Priors P(C)           │
          - Smoothed Likelihoods        │
          - Precompute Log Probabilities|
                     │                  │
                     └────────┬─────────┘
                              ▼
                    [ Vectorized Inference ]
                    (Compute Log Posteriors)
                              │
                              ▼
                 [ Confusion Matrix & Metrics ]
                  - True Positives / Negatives
                  - False Positives / Negatives
                  - Accuracy, Precision, Recall, F1
```

---

## Evaluation Metrics

The classifier is evaluated on the 672 unseen test samples using manual metric calculations:

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall percentage of correct email classifications |
| **Precision** | $\frac{TP}{TP + FP}$ | Proportion of predicted spam emails that are truly spam |
| **Recall** | $\frac{TP}{TP + FN}$ | Proportion of actual spam emails that were caught |
| **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall |

---

## Project Structure

```text
Spam_Mail_Detection/
│
├── emails/
│   └── emails.csv          # Dataset file containing 5,172 email word-count vectors
│
├── naive_bayes.py          # Complete Naive Bayes algorithm & evaluation from scratch
│
└── README.md               # Detailed academic project documentation
```

---

## Requirements & Execution

### Prerequisites
- Python 3.8+
- `pandas` (for CSV reading and column manipulation)
- `numpy` (for matrix and vector operations)

### 1. Installation

Install the required dependencies:

```bash
pip install pandas numpy
```

### 2. Running the Classifier

Run the script from the workspace root directory:

```bash
python naive_bayes.py
```

---

## Experimental Results

Execution of `naive_bayes.py` with a 4500 / 672 random split (Seed = 42) produces the following output:

```text
Dataset Shape: (5172, 3002)

Training Samples: 4500
Testing Samples: 672

Spam Training Emails: 1301
Not Spam Training Emails: 3199
Prior Probability P(Spam): 0.2891 (28.91%)
Prior Probability P(Not Spam): 0.7109 (71.09%)

True Positives: 186
True Negatives: 441
False Positives: 32
False Negatives: 13

Accuracy: 0.9330 (93.30%)
Precision: 0.8532 (85.32%)
Recall: 0.9347 (93.47%)
F1 Score: 0.8921 (89.21%)
```

### Confusion Matrix Breakdown

```text
                          Actual Spam (1)    Actual Not Spam (0)
  Predicted Spam (1)         TP = 186             FP = 32
  Predicted Not Spam (0)     FN = 13              TN = 441
```

---

## Discussion & Analysis

1. **Handling Zero Probabilities:** Laplace smoothing prevents zero probability assignments for words not observed in a given class during training.
2. **Numerical Stability:** Applying the log transform prevents arithmetic underflow during joint probability calculations across 3,000 features.
3. **Classification Performance:** The model demonstrates a **93.47% Recall** on spam emails and an overall **Accuracy of 93.30%**, showing the effectiveness of the Multinomial Naive Bayes model on bag-of-words text data.

---

## Author

**Divyam Kumar Choubey**  

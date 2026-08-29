# Naive Bayes Email Spam Classifier

An academic machine learning project implementing a **Multinomial Naive Bayes Classifier** in Python to classify emails as **Spam (`1`)** or **Not Spam (`0`)**.

---

## Quick Start (Run from Terminal)

Ensure dependencies are installed and run the script directly from your terminal:

```bash
# 1. Install dependencies
pip install pandas numpy

# 2. Run the classifier
python naive_bayes.py
```

---

## Project Overview & References

This project fulfills the requirements defined in the assignment specification:
- **Assignment Specification:** [Naive Bayes Email Classification.pdf](Naive%20Bayes%20Email%20Classification.pdf)
- **Dataset File:** [emails/emails.csv](emails/emails.csv)

The objective is to understand and manually implement the complete Naive Bayes mathematical classification workflow without using black-box machine learning libraries.

---

## Assignment Requirements & Conditions

The project adheres strictly to the rules and constraints outlined in [Naive Bayes Email Classification.pdf](Naive%20Bayes%20Email%20Classification.pdf):

| Condition / Rule | Assignment Requirement | Implementation in `naive_bayes.py` |
| :--- | :--- | :--- |
| **No ML Libraries** | Strict prohibition of `scikit-learn` or pre-built classifiers/metrics (zero-mark penalty). | Uses only `pandas`, `numpy`, `math`, and `random`. |
| **Feature Column Selection** | Exclude `Email No.` (first column) from feature matrix. | Dropped via `data.drop(columns=["Email No."])`. |
| **Dataset Partitioning** | Randomly split into exactly 4,500 training and 672 testing emails. | Randomly shuffled indices (Seed = 42) into 4,500 train and 672 test samples. |
| **Prior Probabilities** | Manually compute class priors from training data. | $P(\text{Spam}) = \frac{N_{\text{Spam}}}{4500}$ and $P(\text{Not Spam}) = \frac{N_{\text{Not Spam}}}{4500}$. |
| **Feature Likelihoods** | Calculate word likelihoods per class with Laplace smoothing. | Add-1 smoothing: $( \text{count} + 1 ) / ( \text{total words in class} + 3000 )$. |
| **Numerical Stability** | Prevent floating-point underflow. | Logarithm transformation applied: $\ln P(C) + \sum x_i \ln P(w_i \mid C)$. |
| **Manual Evaluation** | Calculate Confusion Matrix and evaluation metrics without library functions. | Manual calculation of TP, TN, FP, FN, Accuracy, Precision, Recall, and F1 Score. |

---

## Dataset Details

The dataset [emails.csv](emails/emails.csv) represents emails pre-extracted into a word frequency (Bag of Words) format:

| Property | Value |
| :--- | :--- |
| **File Location** | `emails/emails.csv` |
| **Total Samples (Rows)** | 5,172 emails |
| **Total Columns** | 3,002 columns |
| **Column 1 (`Email No.`)** | Unique string identifier (e.g., `Email 1`, `Email 2`) — excluded from modeling |
| **Columns 2 to 3001** | Word occurrence counts for the 3,000 most common vocabulary terms |
| **Column 3002 (`Prediction`)** | Target ground-truth label: `0` for Not Spam, `1` for Spam |

### Data Representation Example

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

Given an email feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ where $x_i$ represents the count of word $i$, the posterior probability for class $C_k \in \{\text{Spam}, \text{Not Spam}\}$ is:

$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) \cdot P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}$$

Under the Naive Bayes conditional independence assumption:

$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^{n} P(w_i \mid C_k)^{x_i}$$

---

### 2. Class Prior Probabilities

Calculated directly from the distribution of classes in the training set:

$$P(\text{Spam}) = \frac{N_{\text{Spam}}}{N_{\text{Total}}}, \quad P(\text{Not Spam}) = \frac{N_{\text{Not Spam}}}{N_{\text{Total}}}$$

---

### 3. Feature Likelihood with Laplace Smoothing

To eliminate zero probabilities for words that do not appear in a specific class within the training split, **Add-1 (Laplace) Smoothing** is applied:

$$P(w_i \mid C_k) = \frac{\sum_{j \in C_k} x_{ji} + \alpha}{\sum_{i=1}^{n} \sum_{j \in C_k} x_{ji} + \alpha \cdot |V|}$$

Where:
- $\sum_{j \in C_k} x_{ji}$ = Total occurrences of word $i$ in class $C_k$.
- $\alpha = 1$ = Laplace smoothing parameter.
- $|V| = 3000$ = Total number of vocabulary features.

---

### 4. Log-Probability Formulation

Multiplying 3,000 fractional probabilities causes arithmetic underflow. Converting the product into log space yields a stable summation:

$$\ln P(C_k \mid \mathbf{x}) \propto \ln P(C_k) + \sum_{i=1}^{n} x_i \cdot \ln P(w_i \mid C_k)$$

The predicted class $\hat{y}$ is selected via `argmax`:

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
          - Precompute Log Probabilities│
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

The classifier is evaluated on the 672 test samples using custom metric calculations:

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall percentage of correct email classifications |
| **Precision** | $\frac{TP}{TP + FP}$ | Proportion of predicted spam emails that are truly spam |
| **Recall** | $\frac{TP}{TP + FN}$ | Proportion of actual spam emails correctly identified |
| **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall |

---

## Project Structure

```text
Spam_Mail_Detection/
│
├── emails/
│   └── emails.csv                      # Email dataset (5,172 samples x 3,002 columns)
│
├── Naive Bayes Email Classification.pdf # Assignment specification document
│
├── naive_bayes.py                      # Complete Naive Bayes implementation from scratch
│
└── README.md                           # Project documentation and results
```

---

## Experimental Results

Running `python naive_bayes.py` in the terminal outputs:

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

1. **Handling Zero Probabilities:** Laplace smoothing ($\alpha = 1$) prevents unseen words from causing a zero product probability.
2. **Numerical Stability:** Using log probabilities completely mitigates floating-point underflow across 3,000 features.
3. **Classification Performance:** The model attains an **Accuracy of 93.30%** and a **Recall of 93.47%**, demonstrating that the from-scratch Multinomial Naive Bayes classifier is highly effective at identifying spam emails.

---

## Author

**Divyam Kumar Choubey**

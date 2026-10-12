# Project Report — News Classification Model

**Course:** TIDA — Assignment 2  |  **Student:** Kabeir  |  **GitHub:** `<repo URL>`  |  **Live app:** `<Streamlit URL>`

## 1. Objective
Build, evaluate and deploy a supervised machine-learning pipeline that automatically assigns a news article to a topic category (raw text → preprocessing → feature representation → model → evaluation → deployment).

## 2. Dataset
- Source used: *India News Headlines* CSV (3,876,557 headlines; columns `publish_date`, `headline_category`, `headline_text`).
- The raw file has 1,024 fine-grained categories (e.g. `city.mumbai`, `sports.tennis`). The top-level prefix was mapped to 7 topic classes: **Business, Entertainment, Lifestyle, National, Sports, Technology, World**. City-wise, "unknown" and other non-topic sections were excluded because they describe location or are unlabeled rather than topic.
- Duplicates were removed and classes balanced to 15,000 headlines each (105,000 total) so no class dominates the metrics.
- *Note:* the assignment names the Akashvani English News Bulletin (Aug 2025) dataset. The pipeline is dataset-agnostic: only the loading and label-mapping cells change if that data is used.

## 3. Preprocessing
Lowercasing, URL removal, removal of non-letter characters, whitespace normalization. English stop-words are removed inside the TF-IDF vectorizer.

## 4. Feature representation
TF-IDF with unigrams + bigrams, `min_df=2`, sublinear term-frequency scaling.

## 5. Models
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Multinomial Naive Bayes | 79.0% | 0.790 |
| Logistic Regression | 79.3% | 0.793 |
| **Linear SVM (calibrated)** | **80.2%** | **0.802** |

Split: 80% train (84,000) / 20% test (21,000), stratified, `random_state=42`. Linear SVM was selected on macro-F1 and wrapped in `CalibratedClassifierCV` so the app can display class probabilities.

## 6. Evaluation (Linear SVM)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Sports | 0.900 | 0.904 | 0.902 |
| Technology | 0.830 | 0.827 | 0.828 |
| Entertainment | 0.802 | 0.817 | 0.809 |
| World | 0.778 | 0.801 | 0.789 |
| Lifestyle | 0.766 | 0.796 | 0.781 |
| Business | 0.783 | 0.750 | 0.766 |
| National | 0.757 | 0.720 | 0.738 |

**Observations.** Sports is easiest (distinct vocabulary: *ipl, bcci, cup*). National is hardest — it overlaps with Business (policy/tax), World (diplomacy) and Entertainment. Lifestyle vs Entertainment is also confused (celebrity fitness/fashion stories). Headlines are short (~8 words), which limits available signal; longer articles should be easier. Insert the confusion-matrix figure from the notebook here.

## 7. Deployment
The trained pipeline is saved with `joblib` and served by a Streamlit app (`app.py`) hosted on Streamlit Community Cloud from the GitHub repository. The user pastes news text, clicks **Classify**, and sees the predicted category, confidence and a probability bar chart.

## 8. Limitations and future work
- Trained on headlines from one publisher, so performance on full-length articles or other outlets may be lower.
- Category labels come from website sections and are noisy.
- Future work: transformer embeddings (e.g. DistilBERT), multi-label output, retraining on the Akashvani bulletin text.

## 9. Conclusion
A TF-IDF + calibrated Linear SVM pipeline reaches ~80% accuracy on a balanced 7-class task and is deployed as a public web app.

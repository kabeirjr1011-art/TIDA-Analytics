# News Classification Model (TF-IDF + Linear SVM)

Classifies a news headline/article into one of 7 categories: Business, Entertainment, Lifestyle, National, Sports, Technology, World.

**Live app:** `<paste your Streamlit URL here>`

## Files
| File | Purpose |
|---|---|
| `News_Classification.ipynb` | Full pipeline: EDA → preprocessing → TF-IDF → 3 models → evaluation → save model |
| `train.py` | Same pipeline as a script (`python train.py india-news-headlines.csv`) |
| `app.py` | Streamlit web app |
| `model.joblib`, `metrics.json` | Trained model + metrics loaded by the app |
| `requirements.txt` | Dependencies for Streamlit Cloud |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Results (20% hold-out, 21,000 headlines)
Naive Bayes 79.0% · Logistic Regression 79.3% · **Linear SVM 80.2%** (macro-F1 0.802)

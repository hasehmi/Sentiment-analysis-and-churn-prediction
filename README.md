# Sentiment Analysis & Churn Prediction for E-Commerce — Hybrid NLP Architecture


Predicts customer churn risk directly from the text of a product review — no purchase history or demographics required — using a stacking ensemble of three models, deployed as a live web dashboard.

## Why this project

Most sentiment-analysis portfolio projects stop at "positive vs. negative." This one frames the problem the way a business actually cares about it — **which customers are at risk of leaving** — and builds a full pipeline around that:

- **Stacking ensemble, justified with evidence, not assumed.** Three models with different failure modes (Linear SVM, XGBoost, Naive Bayes) are combined via a logistic regression meta-learner. The notebook includes a direct comparison proving the ensemble beats every individual model — it's not complexity for its own sake.
- **Properly validated.** 5-fold cross-validation confirms the result (88.6% accuracy, std dev 0.0035) is stable, not a lucky train/test split.
- **Deployed, not just a notebook.** A Flask app serves both single-review predictions and batch CSV processing, with a live dashboard (churn breakdown, sentiment polarity, top keywords) and CSV export.
- **Honest framing.** No inflated accuracy claims — 88.6% on genuinely noisy, subjective review text is a strong, credible result.

## Project structure

```
├── sentiment_churn_prediction.ipynb   # Full analysis: EDA → TF-IDF → stacking ensemble → evaluation
├── app.py                             # Flask app: single & batch prediction dashboard
├── templates/index.html               # Dashboard UI
├── hybrid_model.pkl                   # Trained stacking ensemble
├── tfidf_vectorizer.pkl               # Fitted TF-IDF vectorizer
├── sample_reviews.csv                 # 300-row sample for quick testing (full dataset below)
├── requirements.txt
└── .gitignore
```

## Dataset

[Amazon Product Reviews — Baby category](http://jmcauley.ucsd.edu/data/amazon/) (McAuley et al., UCSD). The full file (~88MB) is not committed to this repo — download it from the link above and place it as `reviews_Baby_5.csv` in the project root before re-running the notebook. A 300-row sample (`sample_reviews.csv`) is included for quickly testing the app's batch-upload feature without downloading the full dataset.

## Approach

1. **Labeling** — 1–2 star reviews are labeled "churn," 4–5 star "stay," 3-star (ambiguous) reviews are dropped. Classes are balanced 1:1 by downsampling.
2. **Feature engineering** — TF-IDF with domain-specific stopwords (baby, product, amazon, etc.) and up to trigrams, to capture short sentiment-bearing phrases rather than only single words.
3. **Modeling** — three base learners (Linear SVM, XGBoost, Naive Bayes) combined via a stacking ensemble with a logistic regression meta-learner.
4. **Validation** — 5-fold cross-validation, held-out test set evaluation, and a direct comparison against each individual base model to justify the ensemble.
5. **Deployment** — Flask app with a dashboard for single-review analysis and batch CSV processing (with Chart.js visualizations and CSV export).

## Results

| Model | Accuracy |
|---|---|
| Naive Bayes | 85.4% |
| Linear SVC | 88.0% |
| XGBoost | 79.7% |
| **Stacking ensemble** | **88.6%** |

Cross-validated (5-fold): mean accuracy 87.9%, std dev 0.0035 — a stable result, not a lucky split.

## Running it locally

```bash
pip install -r requirements.txt

# App uses the pre-trained model — runs immediately:
python app.py
# then open http://127.0.0.1:5000

# To re-run the full training notebook, first download the full dataset
# (see "Dataset" above) into the project root, then:
jupyter notebook sentiment_churn_prediction.ipynb
```

## What I'd improve with more time

- Add a transformer-based embedding (e.g. DistilBERT) as a fourth base learner or standalone comparison — TF-IDF can't capture word order or context the way embeddings can.
- Add SHAP/LIME on top of the TF-IDF features so the app can explain *which words* drove a specific prediction, not just show the probability.
- Test generalization on a second product category, since the model's vocabulary is currently tuned to baby-product reviews.
- Move the Flask app to a production WSGI server (e.g. gunicorn) instead of the development server for a real deployment.

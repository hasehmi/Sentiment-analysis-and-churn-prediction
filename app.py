from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from textblob import TextBlob
from collections import Counter
import re

app = Flask(__name__)

# ✅ YAHAN SIRF YEH LINE CHANGE HUI HAI
BASE_PATH = r'D:\Users\ans\Desktop\CS-604 FYP\Final Year Project'

model = joblib.load(os.path.join(BASE_PATH, 'hybrid_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_PATH, 'tfidf_vectorizer.pkl'))

def get_top_words(text_list, n=15):
    all_text = " ".join(text_list).lower()
    words = re.findall(r'\w+', all_text)
    stop_words = {'the', 'and', 'is', 'it', 'to', 'for', 'of', 'in', 'this', 'product', 'item', 'bought', 'amazon'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    return Counter(filtered_words).most_common(n)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    category = request.form['category']
    
    vect = vectorizer.transform([review])
    ml_prob = model.predict_proba(vect)[0][1]
    ml_pred = int(ml_prob > 0.5)
    
    analysis = TextBlob(review)
    lex_score = analysis.sentiment.polarity
    
    if ml_pred == 1 or lex_score < -0.1:
        sentiment, status, color = "Negative", "At Risk (Churn)", "#ef233c"
    elif lex_score > 0.1:
        sentiment, status, color = "Positive", "Loyal (Stay)", "#2ecc71"
    else:
        sentiment, status, color = "Neutral", "Undecided", "#f39c12"

    return render_template('index.html', prediction_text=sentiment, status_text=status, 
                           res_color=color, lex_val=round(lex_score, 2), 
                           ml_val=round(ml_prob * 100, 1), original_review=review, 
                           selected_cat=category)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    selected_category = request.form.get('category')
    
    if file:
        try:
            df = pd.read_csv(file)
            col_name = next((c for c in df.columns if c.lower() in ['reviewtext', 'review']), None)
            
            if col_name:
                X = vectorizer.transform(df[col_name].astype(str))
                df['Churn_Prob'] = model.predict_proba(X)[:, 1]
                df['Status'] = df['Churn_Prob'].apply(lambda x: 'Churn' if x > 0.5 else 'Stay')
                
                df['Polarity'] = df[col_name].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                pos = len(df[df['Polarity'] > 0.1])
                neg = len(df[df['Polarity'] <= -0.1])
                neu = len(df) - (pos + neg)
                
                top_words = get_top_words(df[col_name].astype(str).tolist())
                word_labels = [w[0] for w in top_words]
                word_values = [w[1] for w in top_words]
                
                stats = df['Status'].value_counts().to_dict()
                
                result_path = os.path.join(os.getcwd(), 'batch_results.csv')
                df.to_csv(result_path, index=False)
                
                return render_template('index.html', batch_done=True, total=len(df),
                                       churns=stats.get('Churn', 0), stays=stats.get('Stay', 0),
                                       pos=pos, neg=neg, neu=neu, cat=selected_category,
                                       word_labels=word_labels, word_values=word_values)
            else:
                return "Error: Please ensure your CSV has a column named 'reviewText' or 'review'."
        except Exception as e:
            return f"Error processing file: {str(e)}"
    return "Invalid File"

@app.route('/download_result')
def download_result():
    result_file = os.path.join(os.getcwd(), 'batch_results.csv')
    if not os.path.exists(result_file):
        return "No batch results found. Please upload a CSV file first.", 404
    return send_file(result_file, as_attachment=True, download_name='batch_results.csv')

if __name__ == "__main__":
    app.run(debug=True)
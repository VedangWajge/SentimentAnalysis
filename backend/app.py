from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize Hugging Face sentiment pipeline (lightweight)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/")
def home():
    return "Sentiment Analysis Backend is live!"

# Helper to truncate text to a certain number of words
def truncate_text(text, max_words=150):
    words = text.split()
    return ' '.join(words[:max_words])

# Perform sentiment analysis using both VADER and Hugging Face
def analyze_text(text):
    truncated = truncate_text(text)
    vader_scores = vader_analyzer.polarity_scores(truncated)

    try:
        hf_result = sentiment_pipeline(truncated[:512])[0]  # avoid long inputs
        hf_sentiment = hf_result['label']
        hf_score = hf_result['score']
    except Exception as e:
        hf_sentiment = "Error"
        hf_score = str(e)

    return {
        "input": truncated,
        "vader": vader_scores,
        "huggingface": {
            "sentiment": hf_sentiment,
            "confidence": hf_score
        }
    }

# Route to compare results
@app.route('/analyze-comparison', methods=['POST'])
def analyze_comparison():
    data = request.get_json()
    text = data.get('text', '')
    file_content = data.get('fileContent', '')

    combined_texts = []
    if text:
        combined_texts.append(text)
    if file_content:
        combined_texts.append(file_content)

    if not combined_texts:
        return jsonify({'error': 'No valid input provided'}), 400

    results = [analyze_text(txt) for txt in combined_texts]
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

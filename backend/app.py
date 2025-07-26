from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob
import os

app = Flask(__name__)
CORS(app)

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Load Hugging Face model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("Abirate/gpt_3_finetuned_multi_x_science")
model = AutoModelForCausalLM.from_pretrained("Abirate/gpt_3_finetuned_multi_x_science")

# Helper to truncate text to a certain number of words
def truncate_text(text, max_words=150):
    words = text.split()
    return ' '.join(words[:max_words])

# Perform sentiment analysis using both VADER and Hugging Face
def analyze_text(text):
    vader_scores = vader_analyzer.polarity_scores(text)

    short_text = truncate_text(text)

    input_ids = tokenizer.encode(short_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    polarity = TextBlob(response).sentiment.polarity
    if polarity > 0.1:
        hf_sentiment = "Positive"
    elif polarity < -0.1:
        hf_sentiment = "Negative"
    else:
        hf_sentiment = "Neutral"

    return {
        "vader": vader_scores,
        "huggingface": {
            "response": response,
            "sentiment": hf_sentiment,
            "polarity": polarity
        }
    }

# Flask route for comparison
@app.route('/analyze-comparison', methods=['POST'])
def analyze_comparison():
    data = request.get_json()
    text = data.get('text', '')
    file_content = data.get('fileContent', '')

    combined_texts = []

    if text:
        combined_texts.append(truncate_text(text))
    if file_content:
        combined_texts.append(truncate_text(file_content))

    if not combined_texts:
        return jsonify({'error': 'No valid input provided'}), 400

    all_results = []
    for txt in combined_texts:
        result = analyze_text(txt)
        result['input'] = txt
        all_results.append(result)

    return jsonify(all_results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides this automatically
    app.run(debug=False, host="0.0.0.0", port=port)
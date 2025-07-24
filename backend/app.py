from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

vader_analyzer = SentimentIntensityAnalyzer()

tokenizer = AutoTokenizer.from_pretrained("Abirate/gpt_3_finetuned_multi_x_science")
model = AutoModelForCausalLM.from_pretrained("Abirate/gpt_3_finetuned_multi_x_science")

def truncate_text(text, max_words=150):
    words = text.split()
    return ' '.join(words[:max_words])

def analyze_text(text):
    vader_scores = vader_analyzer.polarity_scores(text)

    # Truncate for Hugging Face
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

def extract_reviews_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    reviews = []
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        if "amazon" in url:
            review_blocks = soup.find_all("span", {"data-hook": "review-body"})
        elif "flipkart" in url:
            review_blocks = soup.find_all("div", class_="t-ZTKy")
        else:
            review_blocks = []

        for rb in review_blocks:
            text = rb.get_text(strip=True)
            if text:
                reviews.append(text)
    except Exception as e:
        print("Failed to fetch reviews:", e)

    return reviews

@app.route('/analyze-comparison', methods=['POST'])
def analyze_comparison():
    data = request.get_json()
    text = data.get('text', '')
    file_content = data.get('fileContent', '')
    url = data.get('url', '')

    combined_texts = []

    if text:
        combined_texts.append(text)
    if file_content:
        combined_texts.append(truncate_text(file_content))
    if url:
        reviews = extract_reviews_from_url(url)
        short_reviews = [truncate_text(r) for r in reviews[:5]]
        combined_texts.extend(short_reviews)

    if not combined_texts:
        return jsonify({'error': 'No valid input provided'}), 400

    all_results = []
    for txt in combined_texts:
        result = analyze_text(txt)
        result['input'] = txt
        all_results.append(result)

    return jsonify(all_results)

if __name__ == "__main__":
    app.run(debug=True)

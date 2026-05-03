from fastapi import FastAPI
import pickle
import pandas as pd
import tldextract
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("backend/model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Columns
columns = [
    'url_length', 'valid_url', 'at_symbol', 'sensitive_words_count',
    'path_length', 'isHttps', 'nb_dots', 'nb_hyphens',
    'nb_and', 'nb_or', 'nb_www', 'nb_com', 'nb_underscore'
]

# Storage
history = deque(maxlen=10)

# Rules
TRUSTED_DOMAINS = ["google.com", "github.com", "wikipedia.org", "amazon.com"]
SUSPICIOUS_WORDS = ["login", "verify", "secure", "update", "bank"]

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        url = data.get("url", "")

        ext = tldextract.extract(url)
        domain = ext.domain + "." + ext.suffix

        # Trusted domain
        if domain in TRUSTED_DOMAINS:
            output = {
                "url": url,
                "prediction": "Safe (Trusted Domain)",
                "confidence": 100
            }
            history.append(output)
            return output

        # Suspicious words
        if any(word in url.lower() for word in SUSPICIOUS_WORDS):
            output = {
                "url": url,
                "prediction": "Phishing (Suspicious URL pattern)",
                "confidence": 90
            }
            history.append(output)
            return output

        # ML
        df = pd.DataFrame([data])
        df = df[columns]

        proba = model.predict_proba(df)[0][1]
        prediction = 1 if proba > 0.5 else 0

        result = "Phishing" if prediction == 1 else "Safe"

        output = {
            "url": url,
            "prediction": result,
            "confidence": round(proba * 100, 2)
        }

        history.append(output)
        return output

    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history():
    return list(history)
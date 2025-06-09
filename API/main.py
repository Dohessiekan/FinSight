from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Spam Detector API")

# Load artifacts at startup
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)

model = load_model('spam_classifier.h5')

# Request & Response schemas
class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    confidence: float
    probabilities: dict

@app.post("/predict", response_model=PredictionOut)
def predict_spam(payload: TextIn):
    # 1. Text → sequence → padded
    seq = tokenizer.texts_to_sequences([payload.text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')

    # 2. Model inference
    probs = model.predict(pad).flatten()

    # 3. Build probability dict
    if probs.shape[0] == 1:
        # Binary sigmoid
        prob_dict = {
            label_encoder.classes_[0]: float(1 - probs[0]),
            label_encoder.classes_[1]: float(probs[0])
        }
    else:
        # (rare) multiclass softmax
        prob_dict = {
            label: float(p)
            for label, p in zip(label_encoder.classes_, probs)
        }

    # 4. Pick top
    idx = int(np.argmax(list(prob_dict.values())))
    predicted_label = label_encoder.classes_[idx]
    confidence = list(prob_dict.values())[idx]

    return PredictionOut(
        label=predicted_label,
        confidence=confidence,
        probabilities=prob_dict
    )

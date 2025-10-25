# FastAPI server for Twitter Sentiment Analysis using DistilBERT
# This version includes a threading lock to prevent resource contention during model inference.

import os
import warnings
import threading # New: Import threading for the lock
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, TFDistilBertModel

# -----------------------------
# Config / Environment
# -----------------------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

MODEL_NAME = "distilbert-base-uncased"
WEIGHTS_FILE = "final_high_accuracy_weights.h5"
MAX_LEN = 128
NUM_LABELS = 2
DROPOUT_RATE = 0.4

# -----------------------------
# Global Placeholders for Resources & Lock
# -----------------------------
global_tokenizer = None
global_model = None 
# CRITICAL FIX: Lock to prevent resource contention during model inference
PREDICTION_LOCK = threading.Lock() 

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Twitter Sentiment Analysis API")

class TextPayload(BaseModel):
    text: str

# -----------------------------
# Worker Startup Hook
# -----------------------------
@app.on_event("startup")
def load_resources():
    global global_tokenizer
    global global_model
    
    # 1. Load Tokenizer
    print("Initializing Tokenizer...")
    global_tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # 2. Build and Load Model
    print("Building model architecture...")
    base_encoder = TFDistilBertModel.from_pretrained(MODEL_NAME)

    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

    outputs = base_encoder(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.last_hidden_state[:, 0, :]

    x = tf.keras.layers.Dropout(DROPOUT_RATE)(pooled_output)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    logits = tf.keras.layers.Dense(NUM_LABELS, name="classification_output")(x)

    global_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

    # Load trained weights
    print(f"Loading custom trained weights from {WEIGHTS_FILE}...")
    try:
        global_model.load_weights(WEIGHTS_FILE)
        print("✅ Weights loaded successfully in worker process!")
    except Exception as e:
        print(f"FATAL ERROR: Could not load weights. Check file path: {e}")
        global_model = None 
        
# -----------------------------
# Prediction function (Now wrapped in a lock)
# -----------------------------
def predict(text: str):
    # Ensure model is ready
    if global_model is None or global_tokenizer is None:
        raise RuntimeError("Model resources not initialized.")

    encoded = global_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    
    # --- LOCKED SECTION ---
    # Acquire the lock. Only one thread per worker can execute the model prediction at a time.
    with PREDICTION_LOCK:
        # Use global_model for prediction
        logits = global_model([encoded["input_ids"], encoded["attention_mask"]], training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        
        # Convert tensors to Python primitives only once at the end
        pred_class_tensor = tf.argmax(probs, axis=-1)
        confidence_tensor = tf.reduce_max(probs)
        
        pred_class = int(pred_class_tensor.numpy()[0])
        confidence = float(confidence_tensor.numpy())
    # --- LOCK RELEASED ---
    
    sentiment = "Positive" if pred_class == 1 else "Negative"
    return sentiment, confidence

# -----------------------------
# FastAPI endpoint
# -----------------------------
@app.post("/predict")
async def predict_sentiment(payload: TextPayload):
    try:
        sentiment, confidence = predict(payload.text)
        return {
            "sentiment": sentiment,
            "confidence_score": round(confidence, 4)
        }
    except RuntimeError as e:
        return {"error": str(e), "message": "API is not ready or failed to load model."}, 503
    except Exception as e:
        # Catch any other unhandled prediction error
        print(f"Prediction Error: {e}")
        return {"error": "Internal prediction error.", "details": str(e)}, 500

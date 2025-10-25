# Locustfile to simulate realistic traffic to the sentiment analysis API

from locust import HttpUser, task, between
import random
import json

# A list of diverse samples to simulate realistic API traffic
SENTIMENT_SAMPLES = [
    "I absolutely loved deploying this model! It was challenging but fun and ran perfectly.", # Positive
    "This is completely wrong. The latency is terrible and the predictions are inaccurate.", # Negative
    "I am extremely disappointed with the outcome and the failures.", # Negative
    "The customer support was surprisingly helpful and fast, 10/10 service!", # Positive
    "Performance is bottlenecking, which is a major concern for production deployment.", # Negative
    "Fantastic day for development and testing, everything compiled on the first try.", # Positive
]

class SentimentUser(HttpUser):
    # Time (in seconds) that a simulated user will wait between executing tasks
    wait_time = between(1, 3) 

    @task
    def predict_sentiment(self):
        # Select a random text sample for realistic traffic
        sample_text = random.choice(SENTIMENT_SAMPLES)
        
        payload = {
            "text": sample_text
        }
        
        # Use the 'request' context manager to properly handle failures and assert content
        with self.client.request("POST", "/predict", json=payload, catch_response=True) as response:
            
            # Check for HTTP Status Code (API is reachable)
            if response.status_code != 200:
                response.failure(f"API returned non-200 status code: {response.status_code}")
                return

            try:
                # Attempt to parse the JSON response
                result = response.json()
                
                # Check for required keys in the response
                if 'sentiment' not in result or 'confidence_score' not in result:
                     response.failure("Response missing required keys (sentiment or confidence_score)")
                     
            except json.JSONDecodeError:
                response.failure("Response was not valid JSON")
            except Exception as e:
                response.failure(f"Validation failed: {e}")
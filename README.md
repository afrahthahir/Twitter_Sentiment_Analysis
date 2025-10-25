# 🤖 AI/ML Technical Assessment Report

This repository contains the solution for the two-part technical exercise:
1.  **Deep Learning:** Fine-tuning and deploying a sentiment analysis model optimized for high concurrency.
---

## 1. Technical Exercise 1: High-Concurrency Sentiment Analysis API

### 1.1 Model Development and Iterations

The project's goal was to fine-tune a Transformer Encoder model for sentiment analysis on the Twitter Entity Sentiment Analysis Dataset and deploy it as a highly scalable API. The iteration process focused on optimizing for accuracy, training time, and deployment feasibility.

| Attempt | Classification Scope | Model & Strategy | Outcome & Challenge |
| :---: | :---: | :--- | :--- |
| **1** | 4 Classes (Positive, Negative, Neutral, Irrelevant) | TFBertForSequenceClassification + Class Weights | High complexity due to large parameter count; infeasible for rapid iteration. Dropped due to BERT model size. |
| **2** | 4 Classes (Positive, Negative, Neutral, Irrelevant) | TFDistilBertForSequenceClassification + Full Fine-tuning + Class Weights | Training was excessively slow due to full fine-tuning. |
| **3** | 2 Classes (Positive, Negative) | TFDistilBertForSequenceClassification + Full Fine-tuning | **Final Selection.** Achieved high performance with reduced complexity and smaller model size for optimal deployment. |

### 1.2 Final Model Performance Metrics

The final selected model configuration provided the following performance on the validation set:

* **Model:** **TFDistilBertForSequenceClassification**
* **Task:** 2-Class Sentiment Classification (Positive, Negative)
* **Final Accuracy:** **93.3%**
* **Final Validation Loss:** **0.17**

### 1.3 Model Deployment and Scalability

The model was containerized and deployed as a **FastAPI** service, chosen for its native asynchronous capabilities, which are essential for handling high concurrent load.

* **Framework:** FastAPI
* **Containerization:** Docker
* **Inference Server:** Uvicorn running under Gunicorn (configured for asynchronous workers).
* **Image Name:** `afrahthahir/sentiment-api:latest`

#### Concurrency Validation (500 Concurrent Users)

The API was subjected to a stress test using **Locust** to validate the requirement for handling at least 500 concurrent users. The deployment utilized the asynchronous nature of Uvicorn workers to prevent request blocking.

(./images/image1.png)
(./images/image2.png)

| Metric | Result |
| :--- | :--- |
| **Concurrent Users** | **500** |
| **Request Per Second (RPS)** | **15.5** |
| **Total Failures** | **0%** |

This confirms the API is scaled to handle 500 concurrent users with zero failed requests, maintaining high stability under extreme load.

---

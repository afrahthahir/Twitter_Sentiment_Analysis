# Stage 1: Build the environment on a slim Python image									
FROM python:3.11-slim									
									
# Set the working directory inside the container									
WORKDIR /app									
									
# 1. Copy the Python dependencies file									
COPY requirements.txt .									
									
# 2. Install dependencies (now strictly controlled by frozen versions)									
RUN pip install --no-cache-dir -r requirements.txt									
									
# --- Application and Model Files ---									
									
# 3. Copy the corrected API code									
COPY server.py .									
									
# 4. Copy the entire model weights directory									
# NOTE: This ensures the 'best_frozen_model_weights_stable' folder is available inside /app/									
COPY final_high_accuracy_weights.h5 /app/									
									
# --- Gunicorn/Uvicorn Setup ---									
									
# 5. Expose the port where the application will be listening (8000)
EXPOSE 8000

# 6. Command to run the application (Production Startup)
CMD ["gunicorn", "server:app", "--workers", "7", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "180"]

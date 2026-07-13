# Use the official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the backend requirements first
COPY requirements.txt .

# Install dependencies (CPU versions of PyTorch to save space if GPU isn't available)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the backend files, including the saved_model folder
COPY . .

# Expose port 7860 which Hugging Face Spaces uses by default
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

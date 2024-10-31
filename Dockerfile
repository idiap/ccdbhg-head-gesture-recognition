# Use Python slim as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install curl and any additional dependencies (like build-essential for compiling packages)
RUN apt-get update && apt-get install -y curl


# Install system dependencies for OpenCV
RUN apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install -r requirements.txt --root-user-action=ignore

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 7342

# Run the Streamlit application
CMD ["streamlit", "run", "inventory_forecast.py", "--server.port", "7342", "--server.address", "0.0.0.0"]

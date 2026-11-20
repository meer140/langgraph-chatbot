
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["sh", "-c", "streamlit run frontend.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]

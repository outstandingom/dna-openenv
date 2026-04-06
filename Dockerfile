FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the FastAPI server for HF Space (default)
CMD ["uvicorn", "knowledge_graph_env:app", "--host", "0.0.0.0", "--port", "7860"]
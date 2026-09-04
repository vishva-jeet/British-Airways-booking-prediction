FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh","-c", "streamlit run ui.py --server.address 0.0.0.0 --server.port $PORT"]
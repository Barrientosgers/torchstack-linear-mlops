FROM python:3.11-slim

WORKDIR /app

COPY requirements-train.txt /app/
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src /app/src

CMD ["python", "src/training/train.py"]
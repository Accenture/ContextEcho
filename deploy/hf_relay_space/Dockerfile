FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-relay.txt .
RUN pip install --no-cache-dir -r requirements-relay.txt

COPY donate ./donate

EXPOSE 7860
CMD ["uvicorn", "donate.relay_server:app", "--host", "0.0.0.0", "--port", "7860"]

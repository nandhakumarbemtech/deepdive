# Minimal production image: non-root user, no dev tooling, built-in health check.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m app && mkdir -p /data && chown -R app /data /app
USER app
ENV DEEPDIVE_DB=/data/deepdive.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
# One worker on purpose: runs execute in background threads and share one SQLite file.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

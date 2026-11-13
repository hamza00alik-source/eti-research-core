FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ETI_DATABASE_PATH=/app/data/eti.sqlite3

WORKDIR /app

RUN groupadd --system eti && useradd --system --gid eti --home-dir /app eti \
    && mkdir -p /app/data && chown -R eti:eti /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=eti:eti eti ./eti
COPY --chown=eti:eti web ./web
COPY --chown=eti:eti README.md ./

USER eti
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "eti.api:app", "--host", "0.0.0.0", "--port", "8000"]

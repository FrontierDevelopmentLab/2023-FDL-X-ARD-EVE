## -----------  builder stage  -----------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY ui/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

## -----------  runtime stage  -----------
FROM python:3.12-slim

WORKDIR /app

# `streamlit run ui/main.py` puts ui/ on sys.path, not the workdir, so the
# shared `core` package wouldn't be importable without this.
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY core/ /app/core/
COPY ui/ /app/ui/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

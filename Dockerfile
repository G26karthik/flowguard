# Server deployment of the same app the static site runs in the browser. No torch: the network runs from
# exported NumPy weights (flowguard/npfno.py). models.json decides which weights and evidence files ship.
FROM python:3.11-slim AS bundle
WORKDIR /src
COPY . .
RUN python scripts/served_files.py /bundle

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir numpy==2.4.6 scipy==1.17.1 fastapi==0.162.0 uvicorn==0.54.0
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --from=bundle --chown=app /bundle /app
USER app
# Railway and most hosts set PORT
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn flowguard.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]

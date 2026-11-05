# Server deployment of the same app the static site runs in the browser. No torch: the network
# runs from exported NumPy weights (flowguard/npfno.py).
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.4.6 scipy fastapi uvicorn
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --chown=app flowguard flowguard
COPY --chown=app web web
COPY --chown=app runs runs
COPY --chown=app results results
USER app
EXPOSE 7860
CMD ["uvicorn", "flowguard.api:app", "--host", "0.0.0.0", "--port", "7860"]

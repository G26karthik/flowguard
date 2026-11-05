FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy scipy fastapi uvicorn
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --chown=app flowguard flowguard
COPY --chown=app web web
COPY --chown=app runs runs
COPY --chown=app results results
USER app
ENV OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "flowguard.api:app", "--host", "0.0.0.0", "--port", "7860"]

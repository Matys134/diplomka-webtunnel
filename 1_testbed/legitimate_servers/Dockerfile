FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi hypercorn websockets
COPY server.py /app/server.py
EXPOSE 8000
CMD ["python", "server.py"]

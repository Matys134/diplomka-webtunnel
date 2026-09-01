FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi hypercorn websockets
COPY server.py /app/server.py
# FIX B-4: plain HTTP only. TLS is terminated by the nginx front end, using the same build and
# the same shared parameter file as the WebTunnel bridge.
EXPOSE 8000
CMD ["python", "server.py"]

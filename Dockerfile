FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY app.py utils_app.py columns.txt TunedLightGBM.pkl requirements_app.txt ./
COPY dicts ./dicts
RUN pip install --no-cache-dir -r requirements_app.txt
EXPOSE 8080
CMD ["python", "app.py"]
FROM python:3.9-slim
WORKDIR /main
COPY . /main

# Update and install dependencies, including libgl1-mesa-glx and libgthread-2.0-0
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
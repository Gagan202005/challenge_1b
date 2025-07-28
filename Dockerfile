FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y curl ca-certificates gnupg && rm -rf /var/lib/apt/lists/*

# Ollama binary
RUN curl -L https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64 -o /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama

# Pull model during build (optional, slows down build)
RUN ollama serve & sleep 5 && ollama pull qwen2.5:0.5b-instruct

# Workdir and files
WORKDIR /app
COPY . .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy and allow execution of start.sh
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose Ollama API port
EXPOSE 11434

# Start Ollama and your script
ENTRYPOINT ["/start.sh"]
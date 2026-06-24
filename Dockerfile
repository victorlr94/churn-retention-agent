FROM python:3.12-slim

WORKDIR /app

# Instala uv
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Instala deps de runtime + demo (sin devtools)
RUN uv sync --frozen --no-dev --extra demo

COPY data/sample/ ./data/sample/
COPY models/demo/ ./models/demo/
COPY app.py ./

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

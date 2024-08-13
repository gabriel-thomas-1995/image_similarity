FROM python:3.11.9-slim-bookworm

WORKDIR /app

COPY requirements.txt .
COPY setup_env.sh .

RUN bash setup_env.sh

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
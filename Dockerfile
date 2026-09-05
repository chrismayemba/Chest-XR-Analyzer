FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

RUN useradd --create-home --uid 1000 user

WORKDIR /home/user/app

COPY --chown=user:user requirements-space.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-space.txt

COPY --chown=user:user app/ app/
COPY --chown=user:user static/ static/
COPY --chown=user:user model/ model/

USER user

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

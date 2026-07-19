# HF Docker Space — TaxMate LK
FROM python:3.11-slim

# HF Spaces runs as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy source code
COPY --chown=user . .

# Expose the port Gradio listens on
EXPOSE 7860

# HF Spaces needs PORT=7860
ENV PORT=7860

CMD ["python", "app.py"]

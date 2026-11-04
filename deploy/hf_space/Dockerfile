# Hugging Face Space (Docker, free CPU: 2 vCPU, 16 GB RAM; sizes off: one estimate takes ~8 min on 2 CPU threads). Built by tools/build_hf_space.py.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/hf \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user prefetch.py .
# model weights go into the image, so a restart does not download ~3 GB again
RUN python prefetch.py
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "-u", "src/app.py", "--host", "0.0.0.0", "--port", "7860", "--no-save", "--no-dims"]

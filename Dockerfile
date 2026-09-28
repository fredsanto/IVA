# IVA — Intelligent Variant Analysis: single-container GPU deployment.
# Built locally from this repository (no prebuilt image is published):
#   docker build -t iva .
#   docker run --gpus all -p 8002:8002 -v iva-models:/models iva
# Model weights (~18 GB) are downloaded into /models on first start, so keep
# that volume between runs. Same environment as the native install
# (environment.yml).
FROM condaforge/miniforge3:24.9.2-0

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY environment.yml /opt/environment.yml
RUN conda env create -f /opt/environment.yml -n iva && conda clean -afy

ENV PATH=/opt/conda/envs/iva/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/models

WORKDIR /opt/iva
COPY . /opt/iva
RUN chmod +x start_iva.sh

EXPOSE 8002
CMD ["./start_iva.sh"]

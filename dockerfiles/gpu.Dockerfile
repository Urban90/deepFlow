FROM tensorflow/tensorflow:latest-gpu
RUN python3 -m pip --no-cache-dir install -U \
    pydicom opencv-python easygui python-math matplotlib \
    nibabel numpy pandas regex scipy sklearn \
    tqdm keras scipy scikit-image
RUN rm /etc/apt/sources.list.d/cuda.list; exit 0
RUN rm /etc/apt/sources.list.d/nvidia-ml.list; exit 0
RUN apt-get update && \
    apt-get install -y pigz libgl1-mesa-glx dcm2niix && \
    rm -rf /var/lib/apt/lists/*
COPY scripts/deepFlow.py /deepflow.py
COPY scripts/concat.sh /concat.sh
ENTRYPOINT [ "python3", "/deepflow.py" ]
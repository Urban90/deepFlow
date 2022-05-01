FROM tensorflow/tensorflow:latest
RUN python3 -m pip --no-cache-dir install -U \
    pydicom opencv-python easygui python-math matplotlib \
    nibabel numpy pandas regex scipy sklearn \
    tqdm keras scipy scikit-image
RUN apt-get update && \
    apt-get install -y pigz dcm2niix libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
COPY scripts/deepFlow.py /deepflow.py
COPY scripts/concat.sh /concat.sh
ENTRYPOINT [ "python3", "/deepflow.py" ]
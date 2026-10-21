# Triton with the extra Python packages that our Python backend models need.
#
# Build (from the lab07 directory):
#   docker build -t tritonserver-lab07 triton/
FROM nvcr.io/nvidia/tritonserver:26.09-py3

RUN pip install --no-cache-dir --break-system-packages pillow==12.3.0

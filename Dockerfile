# Image for the "Advanced Ray Train on Anyscale" tutorial.
# The base image provides Ray 2.60.0, Python 3.11, pandas, NumPy, and Matplotlib.
FROM anyscale/ray:2.60.0-py311-cu128

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

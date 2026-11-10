# Use minimal Python base (amd64 for Linux systems)
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv

# Make sure venv's Python and pip are used
ENV PATH="/opt/venv/bin:$PATH"
# Install Python deps
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements2.txt .

RUN pip install torch==2.2.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

RUN pip install --no-cache-dir -r requirements2.txt

# make sure numpy is less than 2.0
RUN pip install "numpy<2.0" --no-cache-dir

RUN pip install nibabel==5.3.2

RUN pip install dipy==1.11.0

# Default command (adjust if running a script instead)
CMD ["python"]


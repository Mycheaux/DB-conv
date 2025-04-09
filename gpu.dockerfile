FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Create and activate the Conda environment
RUN conda env create -f environment.yml

# Install pip requirements
RUN conda run -n dl_gpu_2 pip install -r gpu_requirements.txt  

# Ensure Conda environment is activated by default
SHELL ["conda", "run", "--no-capture-output", "-n", "dl_gpu_2", "/bin/bash", "-c"]

# Define the entry point for interactive execution
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dl_gpu_2", "python", "main.py"]

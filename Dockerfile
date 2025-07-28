# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

## No need to download the model; it will use the local bge_model directory present in the repo

# Set the default command to run task1b.py
CMD ["python", "task1b.py"]

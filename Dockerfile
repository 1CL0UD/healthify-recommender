# FROM tensorflow/tensorflow:latest-gpu
# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the FastAPI app code into the container
COPY ./ /app/

# Install dependencies
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Expose the port your FastAPI app runs on

ENV HOST 0.0.0.0

EXPOSE 8090

CMD ["python", "main.py"]




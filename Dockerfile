## Moxie needs MQTT and services running on MQTT
# Use an official Python base image
FROM python:3.10-slim

# PIP for installing python dep
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1

# Set the working directory in the container
WORKDIR /app

# First add requirements
COPY requirements.txt /app

# Then install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy remaining directory contents into the container at /app
COPY . /app

# Create a volume for persistent data
VOLUME /app/site/work

# Expose the Mosquitto MQTT port and Django development server port
EXPOSE 8000

# Run Django development server
# - Does data migrations and ensure stock data available, then runs the service
CMD ["bash", "-c", "python3 site/manage.py makemigrations && python3 site/manage.py migrate && python3 site/manage.py init_data && python3 site/manage.py runserver --noreload 0.0.0.0:8000"]

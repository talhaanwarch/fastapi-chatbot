# Use Python 3.11 slim image as the base
FROM python:3.11-slim

# Disable pip version check
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt to the working directory (/app)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Create a non-root user and switch to it
RUN useradd -m myuser && chown -R myuser:myuser /app
USER myuser

# Copy the rest of the application app to the working directory (/app)
COPY --chown=myuser:myuser . .

# Make the start script executable
RUN chmod +x start.sh

# Run the start script (relative to the working directory)
CMD ["./start.sh"]

#Ensure the host directory has the correct permissions for the non-root user:
#chown -R 1000:1000 .

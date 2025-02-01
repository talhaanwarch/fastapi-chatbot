FROM python:3.10

# Disable pip version check
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory to /code
WORKDIR /code

# Copy requirements.txt to the working directory (/code)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the working directory (/code)
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Run the start script (relative to the working directory)
CMD ["./start.sh"]
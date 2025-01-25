FROM python:3.10

# Disable pip version check
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /code

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /code/

# Install dependencies
RUN pip install -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

WORKDIR /code/app
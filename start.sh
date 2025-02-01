#!/bin/sh

# Run the application using Gunicorn with Uvicorn workers
# gunicorn app.main:app \  # Entry point of the application (module:app)
#   --workers 1
#   --worker-class uvicorn.workers.UvicornWorker \  # Use Uvicorn workers for ASGI support
#   --bind 0.0.0.0:8001 \  # Bind to all network interfaces on port 8001
#   --timeout 120 \  # Worker timeout in seconds
#   --keep-alive 5 \  # Keep-alive timeout for connections
#   --log-level warning \  # Log level (warning, error, info, etc.)
#   --max-requests 1000 \  # Maximum number of requests a worker will handle before restarting
#   --max-requests-jitter 50 \  # Jitter to add to max_requests to prevent all workers restarting at once
#   --access-logfile - \  # Log access logs to stdout
#   --error-logfile -  # Log error logs to stdout



uvicorn app.main:app --host 0.0.0.0 --port 8001
# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Update the package list and install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Set the working directory
WORKDIR /app

# Copy your Python package wheel here (if you have it locally)
# COPY your-package.whl /app/

# Optional: Install additional dependencies if needed
# RUN pip3 install <your-dependency>

# Example: Install your package
# RUN pip3 install your-package.whl

# Default command to run when starting the container
CMD ["bash"]
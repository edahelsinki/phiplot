FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System dependencies for RDKit + image rendering
RUN apt-get update && apt-get install -y \
    libxrender1 \
    git curl ca-certificates && \
    apt-get clean

# Use a clean working directory
WORKDIR /phiplot

# Install dependencies ONLY
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-cache --no-install-project

# Copy the application source code and README file
COPY ./phiplot ./phiplot
COPY README.md ./

# Add permissions
RUN chgrp -R 0 /phiplot && chmod -R g+rwx /phiplot

# Bypass runtime project metadata checks
ENV UV_NO_PROJECT=1

# Create a writable directory for Matplotlib config to avoid permission errors inside the container
RUN mkdir -p /tmp/matplotlib_config \
    && chgrp -R 0 /tmp/matplotlib_config \
    && chmod -R g+rwx /tmp/matplotlib_config
ENV MPLCONFIGDIR=/tmp/matplotlib_config

# Expose the port for the Panel app
EXPOSE 5006

# Launch the application
CMD ["uv", "run", "python", "-m", "phiplot.main", "--port=5006", "--address=0.0.0.0"]

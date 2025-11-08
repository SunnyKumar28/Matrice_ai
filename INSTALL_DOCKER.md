# Docker Installation Guide

## Linux (Ubuntu/Debian)

### Install Docker Engine

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
sudo docker run hello-world
```

### Add user to docker group (to run without sudo)

```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, or run:
newgrp docker

# Verify (should work without sudo)
docker run hello-world
```

## Linux (Quick Install Script)

```bash
# Download and run Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

## macOS

### Using Homebrew (Recommended)

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop from Applications
```

### Manual Installation

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install the `.dmg` file
3. Open Docker Desktop from Applications

## Windows

### Using Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Run the installer
3. Follow the installation wizard
4. Restart your computer if prompted
5. Launch Docker Desktop

### Requirements
- Windows 10 64-bit: Pro, Enterprise, or Education (Build 15063 or later)
- OR Windows 11 64-bit
- WSL 2 feature enabled

## Install Docker Compose

### Linux (Ubuntu/Debian)

Docker Compose is included with Docker Engine installation above. If you need to install separately:

```bash
# Download latest version
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker-compose --version
```

### macOS/Windows

Docker Compose is included with Docker Desktop.

## Verify Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Test Docker
docker run hello-world

# Check Docker daemon status
sudo systemctl status docker  # Linux
```

## Troubleshooting

### Docker daemon not running (Linux)

```bash
# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Check status
sudo systemctl status docker
```

### Permission denied (Linux)

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker

# Verify
docker ps
```

### Cannot connect to Docker daemon

```bash
# Restart Docker service
sudo systemctl restart docker

# Check if Docker is running
sudo systemctl status docker
```

## Uninstall Docker (if needed)

### Linux (Ubuntu/Debian)

```bash
# Stop Docker
sudo systemctl stop docker

# Uninstall Docker packages
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Remove images, containers, volumes
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
```

## Quick Reference

```bash
# Check if Docker is installed
docker --version

# Check if Docker is running
docker ps

# Start Docker service (Linux)
sudo systemctl start docker

# Stop Docker service (Linux)
sudo systemctl stop docker

# Restart Docker service (Linux)
sudo systemctl restart docker
```

## Next Steps

After installing Docker:

1. Verify installation: `docker run hello-world`
2. Build your image: `docker build -t yolov8-inference-server .`
3. Run your container: `docker-compose up -d`

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for usage instructions.


#!/bin/bash
# Install SOVREN AI Python dependencies

echo "=== Installing SOVREN AI Dependencies ==="

# Install system dependencies first
echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    libssl-dev \
    libcurl4-openssl-dev \
    libopenblas-dev \
    libomp-dev \
    pkg-config \
    libffi-dev

# Create requirements file
echo "Creating requirements file..."
cat > /tmp/sovren_requirements.txt << 'EOF'
# Core ML/AI
numpy>=1.24.0
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0

# Web frameworks
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
aiohttp>=3.8.0

# Database
asyncpg>=0.28.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0

# Audio processing
soundfile>=0.12.0
librosa>=0.10.0
scipy>=1.10.0

# Utilities
pydantic>=2.0.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0
pytz>=2023.3
httpx>=0.24.0

# Monitoring
prometheus-client>=0.17.0
psutil>=5.9.0

# Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
EOF

# Install as sovren user
echo "Installing Python packages..."
sudo -u sovren pip3 install --user -r /tmp/sovren_requirements.txt

# For now, let's install only the essential packages to avoid dependency issues
echo "Installing essential packages..."
sudo -u sovren pip3 install --user \
    numpy \
    fastapi \
    uvicorn \
    asyncpg \
    psutil \
    websockets \
    aiohttp

# Create a simple test to verify installation
echo "Testing installation..."
sudo -u sovren python3 -c "
import sys
print('Python:', sys.version)
try:
    import numpy
    print('✓ NumPy installed')
except: 
    print('✗ NumPy not found')
try:
    import fastapi
    print('✓ FastAPI installed')
except: 
    print('✗ FastAPI not found')
"

echo ""
echo "=== Dependency Installation Complete ==="
echo ""
echo "Note: PyTorch installation requires CUDA compatibility."
echo "You may need to install it separately with:"
echo "  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "For production, you should build PyTorch from source as mentioned in the deployment scripts."
#!/bin/bash
# Install SOVREN AI Python dependencies directly (no venv)

echo "=== Installing SOVREN AI Dependencies Directly ==="

# Install essential packages with --break-system-packages flag
echo "Installing essential Python packages..."

# First, ensure pip is up to date
sudo python3 -m pip install --upgrade pip --break-system-packages

# Install core dependencies
echo "Installing NumPy..."
sudo pip3 install numpy --break-system-packages

echo "Installing FastAPI and web frameworks..."
sudo pip3 install fastapi uvicorn websockets aiohttp --break-system-packages

echo "Installing database drivers..."
sudo pip3 install asyncpg psycopg2-binary --break-system-packages

echo "Installing utilities..."
sudo pip3 install psutil pydantic python-multipart --break-system-packages

# Install PyTorch for CUDA 12.1 (compatible with your system)
echo "Installing PyTorch for CUDA 12.1..."
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Create a simplified main.py that doesn't require all dependencies
echo "Creating simplified main.py for testing..."
sudo tee /data/sovren/api/main_simple.py > /dev/null << 'EOF'
#!/usr/bin/env python3
"""
SOVREN AI Main API Server - Simplified for Testing
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SOVREN-MAIN')

class SovrenAI:
    def __init__(self):
        logger.info("Initializing SOVREN AI System (Simplified)...")
        logger.info("Hardware: 8x B200 PCIe GPUs (183GB each)")
        logger.info("Total GPU Memory: 1,464GB")
        logger.info("System RAM: 2.3TB")
        logger.info("CPU Cores: 288")
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs detected")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f}GB")
            else:
                logger.warning("CUDA not available")
        except ImportError:
            logger.warning("PyTorch not installed")
        
        logger.info("SOVREN AI System initialized (simplified mode)")
        
    async def start(self):
        """Start simplified SOVREN services"""
        logger.info("Starting SOVREN AI services...")
        
        # Start a simple HTTP server for testing
        try:
            from fastapi import FastAPI
            import uvicorn
            
            app = FastAPI(title="SOVREN AI")
            
            @app.get("/")
            async def root():
                return {"message": "SOVREN AI is running", "status": "active"}
            
            @app.get("/health")
            async def health():
                return {"status": "healthy", "service": "sovren-ai"}
            
            # Run the server
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
            
        except ImportError:
            logger.error("FastAPI/Uvicorn not installed, running in heartbeat mode")
            while True:
                await asyncio.sleep(60)
                logger.info("SOVREN AI heartbeat - System active")

async def main():
    sovren = SovrenAI()
    await sovren.start()

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Fix permissions
sudo chown sovren:sovren /data/sovren/api/main_simple.py
sudo chmod +x /data/sovren/api/main_simple.py

# Test the installation
echo ""
echo "Testing Python imports..."
python3 -c "
import sys
print('Python:', sys.version)
print()

packages = ['numpy', 'fastapi', 'uvicorn', 'asyncpg', 'psutil', 'torch']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} installed')
    except ImportError:
        print(f'✗ {pkg} not found')
"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To test SOVREN AI:"
echo "  sudo -u sovren python3 /data/sovren/api/main_simple.py"
echo ""
echo "Or start the service:"
echo "  sudo systemctl start sovren-main"
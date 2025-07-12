#!/bin/bash
# Start SOVREN services with proper configuration

echo "=== Starting SOVREN AI Services ==="

# First, ensure Python is available
if [ ! -f /data/sovren/bin/python ]; then
    echo "Setting up Python environment..."
    sudo mkdir -p /data/sovren/bin
    sudo ln -sf /usr/bin/python3 /data/sovren/bin/python
    sudo ln -sf /usr/bin/python3 /data/sovren/bin/python3
fi

# Check if models exist
echo "Checking for required models..."
MISSING_MODELS=0

if [ ! -f /data/sovren/models/ggml-large-v3.bin ]; then
    echo "WARNING: Whisper model missing at /data/sovren/models/ggml-large-v3.bin"
    MISSING_MODELS=1
fi

if [ ! -f /data/sovren/models/mixtral-8x7b-q4.gguf ]; then
    echo "WARNING: Mixtral model missing at /data/sovren/models/mixtral-8x7b-q4.gguf"
    MISSING_MODELS=1
fi

if [ ! -d /data/sovren/models/styletts2 ] && [ ! -f /data/sovren/models/styletts2_model.pth ]; then
    echo "WARNING: StyleTTS2 models missing"
    MISSING_MODELS=1
fi

if [ $MISSING_MODELS -eq 1 ]; then
    echo ""
    echo "⚠️  Models are missing. You need to download them:"
    echo "   - Whisper Large v3: https://huggingface.co/ggerganov/whisper.cpp"
    echo "   - Mixtral 8x7B: https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF"
    echo "   - StyleTTS2: https://huggingface.co/yl4579/StyleTTS2-LibriTTS"
    echo ""
fi

# Create mock services for testing
echo "Creating test versions of services..."

# Create a simple test API server
sudo tee /data/sovren/api/test_server.py > /dev/null << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import json

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'service': 'sovren-api'}).encode())
        else:
            self.send_response(404)
            self.end_headers()

print("SOVREN AI Test API Server starting on port 8000...")
with socketserver.TCPServer(("", 8000), TestHandler) as httpd:
    httpd.serve_forever()
EOF

# Create consciousness engine test
sudo tee /data/sovren/consciousness/test_consciousness.py > /dev/null << 'EOF'
#!/usr/bin/env python3
import time
import json

print("SOVREN AI Consciousness Engine (Test Mode)")
print("Hardware: 8x B200 PCIe GPUs (183GB each)")
print("Total GPU Memory: 1,464GB")
print("Status: Active (Test Mode)")

# Keep running
while True:
    time.sleep(60)
    print(f"Consciousness heartbeat: {time.time()}")
EOF

# Fix permissions
sudo chown -R sovren:sovren /data/sovren
sudo chmod +x /data/sovren/api/test_server.py
sudo chmod +x /data/sovren/consciousness/test_consciousness.py

# Start test services
echo ""
echo "Starting test services..."

# Kill any existing Python processes on these ports
sudo pkill -f "python.*8000" 2>/dev/null || true
sudo pkill -f "test_server.py" 2>/dev/null || true
sudo pkill -f "test_consciousness.py" 2>/dev/null || true

# Start in background
sudo -u sovren nohup python3 /data/sovren/api/test_server.py > /data/sovren/logs/test_api.log 2>&1 &
echo "✓ Test API server started on port 8000"

sudo -u sovren nohup python3 /data/sovren/consciousness/test_consciousness.py > /data/sovren/logs/test_consciousness.log 2>&1 &
echo "✓ Test consciousness engine started"

echo ""
echo "=== Test Services Running ==="
echo ""
echo "Check status:"
echo "  API Health: curl http://localhost:8000/health"
echo "  Logs: tail -f /data/sovren/logs/*.log"
echo ""
echo "Note: These are test services. The actual SOVREN AI implementation"
echo "      requires the AI models and proper Python environment setup."
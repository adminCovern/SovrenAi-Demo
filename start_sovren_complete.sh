#!/bin/bash

echo "=========================================="
echo "    STARTING SOVREN AI COMPLETE SYSTEM"
echo "=========================================="

# Kill any existing processes
echo "Stopping existing services..."
pkill -f "python.*main.py" 2>/dev/null
pkill -f "node.*next" 2>/dev/null
pkill -f "npm.*dev" 2>/dev/null

# Wait for processes to stop
sleep 3

# Start nginx if not running
echo "Starting nginx..."
sudo systemctl start nginx
sudo systemctl enable nginx

# Start API server
echo "Starting SOVREN API server..."
cd /data/sovren
export PYTHONPATH=/data/sovren
nohup python3 api/main.py > /tmp/sovren-api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Wait for API to initialize
echo "Waiting for API to start..."
sleep 8

# Check if API is responding
API_CHECK=$(curl -s http://localhost:8000/status 2>/dev/null | grep -o "operational" || echo "failed")
if [ "$API_CHECK" = "operational" ]; then
    echo "✓ API server is running on port 8000"
else
    echo "✗ API server failed to start. Check logs: tail -f /tmp/sovren-api.log"
fi

# Start frontend
echo "Starting frontend..."
cd /data/sovren/frontend
nohup npm run dev > /tmp/sovren-frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Wait for frontend to start
echo "Waiting for frontend to start..."
sleep 15

# Check if frontend is responding
FRONTEND_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null)
if [ "$FRONTEND_CHECK" = "200" ]; then
    echo "✓ Frontend is running on port 3000"
else
    echo "✗ Frontend failed to start. Check logs: tail -f /tmp/sovren-frontend.log"
fi

# Final status check
echo ""
echo "=========================================="
echo "SOVREN AI STATUS:"
echo "=========================================="
echo "API Server (port 8000): $API_CHECK"
echo "Frontend (port 3000): $([ "$FRONTEND_CHECK" = "200" ] && echo "running" || echo "failed")"
echo "Domain: https://sovrenai.app"
echo ""
echo "Logs:"
echo "  API: tail -f /tmp/sovren-api.log"
echo "  Frontend: tail -f /tmp/sovren-frontend.log"
echo "  Nginx: sudo tail -f /var/log/nginx/error.log"
echo ""
echo "Test URLs:"
echo "  Local API: curl http://localhost:8000/status"
echo "  Local Frontend: curl http://localhost:3000"
echo "  Public: https://sovrenai.app"
echo "=========================================="
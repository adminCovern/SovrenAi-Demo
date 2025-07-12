#!/bin/bash
# Start all SOVREN AI services

echo "🚀 Starting SOVREN AI Services..."

# 1. Database
echo "Starting PostgreSQL..."
sudo systemctl start postgresql
sleep 2

# 2. Main API
echo "Starting SOVREN Main Service..."
sudo systemctl start sovren-main
sleep 3

# 3. Frontend
echo "Starting Frontend..."
cd /data/sovren/frontend
sudo -u sovren npm start > /dev/null 2>&1 &
echo "Frontend started on port 3000"

# 4. MCP Server (optional)
echo "Starting MCP Server..."
sudo systemctl start sovren-mcp 2>/dev/null || echo "MCP service not configured"

# 5. Check status
echo ""
echo "✅ Services Status:"
sudo systemctl is-active sovren-main && echo "  - API: Running" || echo "  - API: Not running"
curl -s http://localhost:8000/health > /dev/null && echo "  - Health Check: OK" || echo "  - Health Check: Failed"
curl -s http://localhost:3000 > /dev/null && echo "  - Frontend: Running" || echo "  - Frontend: Starting..."

echo ""
echo "🌐 Access SOVREN AI at:"
echo "  - Local: http://localhost:3000"
echo "  - Domain: https://sovrenai.app (after DNS setup)"
echo ""
echo "📊 View logs:"
echo "  sudo journalctl -u sovren-main -f"
#!/bin/bash
# Final SOVREN AI Verification

echo "==================================================="
echo "          SOVREN AI FINAL VERIFICATION"
echo "==================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check function
check() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        echo "   Fix: $3"
    fi
}

# 1. Services
echo "1. SERVICES CHECK:"
systemctl is-active sovren-main >/dev/null 2>&1
check $? "Main API Service" "Run: sudo systemctl start sovren-main"

curl -s http://localhost:8000/health >/dev/null 2>&1
check $? "API Health Endpoint" "Check logs: sudo journalctl -u sovren-main -f"

curl -s http://localhost:3000 >/dev/null 2>&1
check $? "Frontend Running" "Run: cd /data/sovren/frontend && npm start"
echo ""

# 2. Features
echo "2. FEATURES CHECK:"
curl -s http://localhost:8000/status | grep -q "consciousness" 2>/dev/null
check $? "Consciousness Engine Active" "Restart service"

curl -s http://localhost:8000/status | grep -q "battalions" 2>/dev/null
check $? "Agent Battalions Ready" "Check API logs"

curl -s http://localhost:8000/status | grep -q "shadow_board" 2>/dev/null
check $? "Shadow Board Active" "Verify SMB tier"
echo ""

# 3. Domain
echo "3. DOMAIN CHECK:"
host sovrenai.app | grep -q "38.127.229.132" 2>/dev/null
check $? "DNS Points to Server" "Update DNS records at registrar"

curl -s https://sovrenai.app --connect-timeout 5 >/dev/null 2>&1
check $? "HTTPS Working" "Run: sudo certbot --nginx -d sovrenai.app -d www.sovrenai.app"
echo ""

# 4. Access Points
echo "4. ACCESS POINTS:"
echo -e "${YELLOW}Local Access:${NC}"
echo "  • API: http://38.127.229.132:8000"
echo "  • Frontend: http://38.127.229.132:3000"
echo ""
echo -e "${YELLOW}Public Access (after DNS/SSL):${NC}"
echo "  • Web: https://sovrenai.app"
echo "  • API: https://sovrenai.app/api"
echo "  • Mobile: https://sovrenai.app (responsive)"
echo ""

# 5. Quick Actions
echo "5. QUICK ACTIONS:"
echo "  • Start all: /home/ubuntu/start_sovren_all.sh"
echo "  • View logs: sudo journalctl -u sovren-main -f"
echo "  • Update code: /home/ubuntu/easy_github_update.sh"
echo "  • SSL setup: sudo certbot --nginx -d sovrenai.app -d www.sovrenai.app"
echo ""

echo "==================================================="
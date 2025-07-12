#!/bin/bash
# Fix deployment issues after initial run

echo "=== Fixing SOVREN Deployment Issues ==="

# 1. Stop old processes using /opt/sovren
echo "Stopping old processes..."
sudo pkill -f "/opt/sovren/bin/python3"
sleep 2

# 2. Update the consciousness engine memory settings
echo "Updating consciousness engine for 183GB B200s..."
sudo cp /home/ubuntu/consciousness_engine_pcie_b200.py /data/sovren/consciousness/consciousness_engine.py

# 3. Create symlink for backward compatibility (temporary)
if [ ! -e /opt/sovren ]; then
    echo "Creating temporary symlink for backward compatibility..."
    sudo ln -s /data/sovren /opt/sovren
fi

# 4. Update Python path in running services
echo "Updating service configurations..."
sudo systemctl stop sovren-* 2>/dev/null || true

# 5. Fix any remaining /opt/sovren references in production files
if [ -f /data/sovren/launch_sovren_production.py ]; then
    sudo sed -i 's|/opt/sovren|/data/sovren|g' /data/sovren/launch_sovren_production.py
fi

if [ -f /data/sovren/api/api_server.py ]; then
    sudo sed -i 's|/opt/sovren|/data/sovren|g' /data/sovren/api/api_server.py
fi

# 6. Ensure Python is available at /data/sovren/bin/python
if [ ! -f /data/sovren/bin/python ]; then
    echo "Setting up Python in /data/sovren..."
    sudo mkdir -p /data/sovren/bin
    if [ -f /opt/sovren/bin/python3 ]; then
        sudo cp -a /opt/sovren/bin/* /data/sovren/bin/
    elif command -v python3 &> /dev/null; then
        sudo ln -s $(which python3) /data/sovren/bin/python
        sudo ln -s $(which python3) /data/sovren/bin/python3
    fi
fi

# 7. Fix permissions
sudo chown -R sovren:sovren /data/sovren

echo "=== Fixes Applied ==="
echo ""
echo "Next steps:"
echo "1. Run the deployment again: sudo /home/ubuntu/DEPLOY_NOW_CORRECTED.sh"
echo "2. Check services: sudo systemctl status sovren-*"
echo "3. Monitor logs: sudo journalctl -f -u sovren-*"
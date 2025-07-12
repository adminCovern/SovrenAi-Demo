#!/bin/bash
# Verify SOVREN paths are correctly set to /data/sovren

echo "=== SOVREN Path Verification ==="
echo

# Check environment
echo "1. Environment variables:"
env | grep -i sovren | grep -E "(opt|data)" || echo "   No SOVREN env vars found"
echo

# Check running processes
echo "2. Running processes:"
ps aux | grep -i sovren | grep -v grep | grep -E "(opt|data)" || echo "   No SOVREN processes running"
echo

# Check systemd services
echo "3. Systemd services:"
systemctl list-units --all | grep sovren || echo "   No SOVREN services found"
echo

# Check file references
echo "4. File references in /home/ubuntu:"
echo "   Files with /opt/sovren (should be 0):"
grep -l "/opt/sovren" /home/ubuntu/*.sh /home/ubuntu/*.py /home/ubuntu/*.txt 2>/dev/null | grep -v verify-sovren-paths.sh | wc -l
echo
echo "   Files with /data/sovren (should be 11+):"
grep -l "/data/sovren" /home/ubuntu/*.sh /home/ubuntu/*.py /home/ubuntu/*.txt 2>/dev/null | wc -l
echo

# Check if symlink exists
echo "5. Symlink check:"
if [ -L /opt/sovren ]; then
    echo "   ✓ Symlink exists: /opt/sovren -> $(readlink /opt/sovren)"
else
    echo "   ✗ No symlink from /opt/sovren to /data/sovren"
fi
echo

# Check data directory
echo "6. Data directory check:"
if [ -d /data/sovren ]; then
    echo "   ✓ /data/sovren exists"
    echo "   Size: $(du -sh /data/sovren 2>/dev/null | cut -f1)"
else
    echo "   ✗ /data/sovren does not exist"
fi

echo
echo "=== Path migration verification complete ==="
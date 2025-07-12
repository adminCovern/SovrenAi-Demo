#!/bin/bash
# Quick check of current SOVREN path usage

echo "=== SOVREN Path Compliance Check ==="
echo

echo "Files containing /opt/sovren (non-compliant):"
grep -l "/opt/sovren" /home/ubuntu/*.sh /home/ubuntu/*.py /home/ubuntu/*.txt 2>/dev/null | grep -v "check-sovren-paths.sh" | grep -v "fix-sovren-paths.sh" || echo "None found"
echo

echo "Files containing /data/sovren (compliant):"
grep -l "/data/sovren" /home/ubuntu/*.sh /home/ubuntu/*.py /home/ubuntu/*.txt 2>/dev/null | grep -v "check-sovren-paths.sh" || echo "None found"
echo

echo "=== Specific violations found ==="
grep -n "/opt/sovren" /home/ubuntu/sovren-deployment-final.sh 2>/dev/null | head -10 || echo "No violations in sovren-deployment-final.sh"
echo

echo "To fix these violations, run:"
echo "sudo ./fix-sovren-paths.sh"
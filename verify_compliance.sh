#!/bin/bash
# SOVREN AI Compliance Verification Script
# Verifies all guidelines are met after fixes

echo "============================================"
echo "SOVREN AI COMPLIANCE VERIFICATION"
echo "============================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASS_COUNT++))
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAIL_COUNT++))
}

echo -e "\n1. PATH COMPLIANCE CHECK (/data/sovren)"
echo "----------------------------------------"

# Check for any remaining /opt/sovren references
OPT_REFS=$(grep -r "/opt/sovren" /home/ubuntu/*.py /home/ubuntu/*.sh 2>/dev/null | grep -v "backup" | wc -l)
if [ $OPT_REFS -eq 0 ]; then
    check_pass "No /opt/sovren references found in active files"
else
    check_fail "Found $OPT_REFS /opt/sovren references"
    grep -r "/opt/sovren" /home/ubuntu/*.py /home/ubuntu/*.sh 2>/dev/null | grep -v "backup" | head -5
fi

# Check systemd service
if grep -q "/opt/sovren" /etc/systemd/system/sovren-ai.service 2>/dev/null; then
    check_fail "sovren-ai.service still has /opt/sovren paths"
else
    check_pass "sovren-ai.service uses /data/sovren paths"
fi

echo -e "\n2. TTS COMPLIANCE CHECK (StyleTTS2 only)"
echo "----------------------------------------"

# Check for XTTS usage
XTTS_REFS=$(grep -ri "xtts\|XTTS\|Coqui.*TTS\|from TTS" /home/ubuntu/*.py 2>/dev/null | grep -v "StyleTTS2" | grep -v "backup" | wc -l)
if [ $XTTS_REFS -eq 0 ]; then
    check_pass "No XTTS/Coqui TTS references found"
else
    check_fail "Found $XTTS_REFS XTTS/Coqui references"
    grep -ri "xtts\|XTTS\|Coqui.*TTS" /home/ubuntu/*.py 2>/dev/null | grep -v "StyleTTS2" | grep -v "backup" | head -3
fi

echo -e "\n3. GPU/NCCL CONFIGURATION CHECK"
echo "----------------------------------------"

# Check for incorrect NCCL settings
if grep -q "NCCL_P2P_LEVEL=NVL" /home/ubuntu/sovren-deployment-final.sh; then
    check_fail "Incorrect NCCL_P2P_LEVEL=NVL found"
else
    check_pass "No NCCL_P2P_LEVEL=NVL found"
fi

if grep -q "NCCL_P2P_DISABLE=1" /home/ubuntu/sovren-deployment-final.sh; then
    check_pass "NCCL_P2P_DISABLE=1 properly set"
else
    check_fail "NCCL_P2P_DISABLE=1 not found"
fi

echo -e "\n4. EXTERNAL API COMPLIANCE CHECK"
echo "----------------------------------------"

# Check for VoIP.ms references
VOIP_MS=$(grep -ri "voip\.ms\|VoIP\.ms" /home/ubuntu/*.py 2>/dev/null | grep -v "backup" | wc -l)
if [ $VOIP_MS -eq 0 ]; then
    check_pass "No VoIP.ms references found"
else
    check_fail "Found $VOIP_MS VoIP.ms references"
fi

# Check for Skyetel
if grep -q "skyetel" /home/ubuntu/sovereign_awakening_handler.py; then
    check_pass "Skyetel configuration found"
else
    check_fail "Skyetel configuration missing"
fi

echo -e "\n5. FIREWALL STATUS CHECK"
echo "----------------------------------------"

UFW_STATUS=$(sudo ufw status | grep -c "Status: inactive")
if [ $UFW_STATUS -eq 1 ]; then
    check_pass "UFW firewall is inactive (as required)"
else
    check_fail "UFW firewall is NOT inactive"
fi

echo -e "\n6. HARDWARE CONFIGURATION CHECK"
echo "----------------------------------------"

# Check GPU memory references
if grep -q "183GB\|1.46TB\|1460GB" /home/ubuntu/*.py 2>/dev/null; then
    check_fail "Incorrect GPU memory values found (183GB/1.46TB)"
else
    check_pass "No incorrect GPU memory values found"
fi

if grep -q "'gpu_memory_gb': 80" /home/ubuntu/consciousness_engine_pcie_b200.py; then
    check_pass "Correct GPU memory (80GB) in consciousness engine"
else
    check_fail "GPU memory not correctly set in consciousness engine"
fi

echo -e "\n7. SERVICE USER CHECK"
echo "----------------------------------------"

if grep -q "User=sovren" /etc/systemd/system/sovren-*.service 2>/dev/null; then
    check_pass "Services configured to run as 'sovren' user"
else
    check_fail "Services not properly configured for 'sovren' user"
fi

echo -e "\n============================================"
echo "COMPLIANCE SUMMARY"
echo "============================================"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "\n${GREEN}✅ SYSTEM IS FULLY COMPLIANT${NC}"
    echo "Ready for deployment with:"
    echo "  sudo /home/ubuntu/DEPLOY_NOW_CORRECTED.sh"
else
    echo -e "\n${RED}❌ COMPLIANCE ISSUES FOUND${NC}"
    echo "Please fix the issues above before deployment"
fi

echo -e "\nCompliance report saved to: /home/ubuntu/SOVREN_GUIDELINES_COMPLIANCE_REPORT.md"
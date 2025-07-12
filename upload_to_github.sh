#!/bin/bash
# Simple GitHub Upload Script for SOVREN AI

echo "================================================"
echo "     SOVREN AI - GitHub Upload Assistant"
echo "================================================"
echo ""

# Colors for clarity
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Go to SOVREN directory
cd /data/sovren

# Step 2: Initialize git if needed
if [ ! -d .git ]; then
    echo -e "${YELLOW}Setting up Git for the first time...${NC}"
    git init
    echo "✓ Git initialized"
fi

# Step 3: Create .gitignore to exclude sensitive files
echo "Creating safety file to protect sensitive data..."
cat > .gitignore << 'EOF'
# Sensitive files - never upload these
*.pem
*.key
*.crt
.env
config/secrets.json
logs/*.log
*.db

# Large files we don't need in git
models/
*.bin
*.pth
*.gguf
node_modules/
.next/
build/
dist/
*.pyc
__pycache__/

# Personal data
data/users.db
data/api_auth.db
data/phone_numbers.db

# Temporary files
*.tmp
*.swp
.DS_Store
EOF

echo "✓ Safety file created"

# Step 4: Add all files
echo -e "${YELLOW}Preparing files for upload...${NC}"
git add .
echo "✓ Files prepared"

# Step 5: Create commit
echo -e "${YELLOW}Saving current version...${NC}"
COMMIT_MESSAGE="Complete SOVREN AI v3.1 deployment - $(date +'%Y-%m-%d %H:%M')"
git commit -m "$COMMIT_MESSAGE" || {
    echo "No changes to save, continuing..."
}

# Step 6: Add GitHub repository
echo -e "${YELLOW}Connecting to GitHub...${NC}"
git remote remove origin 2>/dev/null  # Remove if exists
git remote add origin https://github.com/adminCovern/SovrenAI-deployment.git
echo "✓ Connected to GitHub"

# Step 7: Instructions for the user
echo ""
echo "================================================"
echo -e "${GREEN}     READY TO UPLOAD!${NC}"
echo "================================================"
echo ""
echo "Now you need to do ONE simple step:"
echo ""
echo "1. Copy and paste this command:"
echo ""
echo -e "${YELLOW}cd /data/sovren && git push -u origin main${NC}"
echo ""
echo "2. When prompted:"
echo "   - Username: adminCovern"
echo "   - Password: (your GitHub personal access token)"
echo ""
echo "💡 Don't have a token? Here's how to get one:"
echo "   1. Go to: https://github.com/settings/tokens"
echo "   2. Click 'Generate new token (classic)'"
echo "   3. Give it a name like 'SOVREN Upload'"
echo "   4. Check the 'repo' checkbox"
echo "   5. Click 'Generate token' at the bottom"
echo "   6. Copy the token (it looks like: ghp_xxxxxxxxxxxxx)"
echo ""
echo "================================================"

# Create a backup script too
cat > /home/ubuntu/quick_github_push.sh << 'SCRIPT'
#!/bin/bash
cd /data/sovren
git add .
git commit -m "Update SOVREN AI - $(date +'%Y-%m-%d %H:%M')"
git push origin main
SCRIPT

chmod +x /home/ubuntu/quick_github_push.sh

echo ""
echo "✅ Also created a quick update script at:"
echo "   /home/ubuntu/quick_github_push.sh"
echo ""
echo "Use it anytime to quickly save changes to GitHub!"
#!/bin/bash
# Easy GitHub update script for SOVREN AI

cd /data/sovren

echo "📤 Uploading SOVREN AI updates to GitHub..."

# Add any new or changed files
git add .

# Create a commit with timestamp
git commit -m "Update SOVREN AI - $(date +'%Y-%m-%d %H:%M')"

# Push to GitHub
git push origin main

echo "✅ Updates uploaded successfully!"
echo "View at: https://github.com/adminCovern/SovrenAI-deployment"
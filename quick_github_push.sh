#!/bin/bash
cd /data/sovren
git add .
git commit -m "Update SOVREN AI - $(date +'%Y-%m-%d %H:%M')"
git push origin main

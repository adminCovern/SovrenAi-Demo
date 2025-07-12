#!/bin/bash
# SOVREN Path Migration Script
# Migrates all references from /opt/sovren to /data/sovren
# This ensures compliance with Project Guidelines.md

set -e
set -u

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    error "This script must be run as root: sudo $0"
fi

log "Starting SOVREN path migration from /opt/sovren to /data/sovren"

# Step 1: Create backup of current files
log "Creating backup of files to be modified..."
BACKUP_DIR="/home/ubuntu/sovren-path-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Find all files that contain /opt/sovren
FILES_TO_UPDATE=$(grep -rl "/opt/sovren" /home/ubuntu/ 2>/dev/null | grep -v ".git" | grep -v "fix-sovren-paths.sh" || true)

if [ -z "$FILES_TO_UPDATE" ]; then
    warn "No files found containing /opt/sovren in /home/ubuntu/"
else
    for file in $FILES_TO_UPDATE; do
        if [ -f "$file" ]; then
            cp "$file" "$BACKUP_DIR/" || warn "Could not backup $file"
            log "Backed up: $file"
        fi
    done
fi

# Step 2: Check if /data/sovren exists, create if not
log "Checking /data/sovren directory..."
if [ ! -d /data ]; then
    error "/data mount point does not exist! Please ensure the data volume is mounted."
fi

if [ ! -d /data/sovren ]; then
    log "Creating /data/sovren directory structure..."
    mkdir -p /data/sovren/{bin,lib,config,logs,data,models,src}
    mkdir -p /data/sovren/{consciousness,shadow_board,agent_battalion,time_machine}
    mkdir -p /data/sovren/{security,voice,api,frontend,billing,approval}
    mkdir -p /data/sovren/data/{users,api_auth,applications,phone_numbers}
    mkdir -p /data/sovren/models/{llms,whisper,tts}
    
    # Set ownership
    if id "sovren" &>/dev/null; then
        chown -R sovren:sovren /data/sovren
        log "Set ownership to sovren user"
    fi
else
    log "/data/sovren already exists"
fi

# Step 3: If /opt/sovren exists, offer to migrate data
if [ -d /opt/sovren ]; then
    log "Found existing /opt/sovren directory"
    echo -e "${YELLOW}Would you like to migrate existing data from /opt/sovren to /data/sovren?${NC}"
    echo "This will copy all data and preserve permissions."
    read -p "Migrate data? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Migrating data from /opt/sovren to /data/sovren..."
        
        # Use rsync for reliable copying with progress
        if command -v rsync &> /dev/null; then
            rsync -avP --owner --group --perms /opt/sovren/ /data/sovren/
        else
            cp -rpv /opt/sovren/* /data/sovren/ 2>/dev/null || true
            cp -rpv /opt/sovren/.* /data/sovren/ 2>/dev/null || true
        fi
        
        log "Data migration completed"
        
        # Verify critical files
        if [ -f /opt/sovren/bin/python ]; then
            if [ -f /data/sovren/bin/python ]; then
                log "Python binary successfully migrated"
            else
                error "Python binary migration failed!"
            fi
        fi
    fi
fi

# Step 4: Update all file references
log "Updating file references from /opt/sovren to /data/sovren..."

# Update files in /home/ubuntu
for file in $FILES_TO_UPDATE; do
    if [ -f "$file" ]; then
        sed -i 's|/opt/sovren|/data/sovren|g' "$file"
        log "Updated: $file"
    fi
done

# Step 5: Update systemd service files
log "Checking for systemd service files..."
SYSTEMD_FILES=$(find /etc/systemd/system/ -name "sovren*.service" 2>/dev/null || true)

if [ -n "$SYSTEMD_FILES" ]; then
    for service in $SYSTEMD_FILES; do
        if grep -q "/opt/sovren" "$service"; then
            cp "$service" "$BACKUP_DIR/"
            sed -i 's|/opt/sovren|/data/sovren|g' "$service"
            log "Updated systemd service: $service"
        fi
    done
    
    # Reload systemd
    systemctl daemon-reload
    log "Reloaded systemd configuration"
fi

# Step 6: Update environment variables in current session files
log "Checking shell configuration files..."
for config_file in /home/*/.bashrc /home/*/.profile /root/.bashrc /root/.profile; do
    if [ -f "$config_file" ] && grep -q "/opt/sovren" "$config_file"; then
        cp "$config_file" "$BACKUP_DIR/"
        sed -i 's|/opt/sovren|/data/sovren|g' "$config_file"
        log "Updated: $config_file"
    fi
done

# Step 7: Create symlink for compatibility (optional)
echo -e "${YELLOW}Would you like to create a symlink from /opt/sovren to /data/sovren for compatibility?${NC}"
echo "This can help during transition but should be removed later."
read -p "Create symlink? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d /opt/sovren ] && [ ! -L /opt/sovren ]; then
        mv /opt/sovren /opt/sovren.old
        log "Moved /opt/sovren to /opt/sovren.old"
    fi
    
    if [ ! -e /opt/sovren ]; then
        ln -s /data/sovren /opt/sovren
        log "Created symlink: /opt/sovren -> /data/sovren"
    fi
fi

# Step 8: Verify the changes
log "Verifying changes..."
REMAINING=$(grep -r "/opt/sovren" /home/ubuntu/ 2>/dev/null | grep -v ".git" | grep -v "sovren-path-backup" | grep -v "fix-sovren-paths.sh" | wc -l || true)

if [ "$REMAINING" -gt 0 ]; then
    warn "Found $REMAINING remaining references to /opt/sovren"
    echo "You can view them with:"
    echo "grep -r '/opt/sovren' /home/ubuntu/ | grep -v '.git' | grep -v 'sovren-path-backup'"
else
    log "All references successfully updated!"
fi

# Step 9: Create verification script
cat > /home/ubuntu/verify-sovren-paths.sh << 'EOF'
#!/bin/bash
# Verify SOVREN paths are correctly set to /data/sovren

echo "Checking SOVREN path configuration..."
echo

# Check environment
echo "Environment variables:"
env | grep -i sovren | grep -E "(opt|data)" || echo "No SOVREN env vars found"
echo

# Check running processes
echo "Running processes:"
ps aux | grep -i sovren | grep -v grep | grep -E "(opt|data)" || echo "No SOVREN processes running"
echo

# Check systemd services
echo "Systemd services:"
systemctl list-units --all | grep sovren || echo "No SOVREN services found"
echo

# Check file references
echo "File references in /home/ubuntu:"
grep -r "/opt/sovren" /home/ubuntu/ 2>/dev/null | grep -v ".git" | grep -v "backup" | head -5 || echo "No /opt/sovren references found"
echo
grep -r "/data/sovren" /home/ubuntu/ 2>/dev/null | grep -v ".git" | grep -v "backup" | head -5 || echo "No /data/sovren references found"

echo
echo "Path migration verification complete."
EOF

chmod +x /home/ubuntu/verify-sovren-paths.sh

# Summary
echo
log "======================================================"
log "SOVREN PATH MIGRATION COMPLETED"
log "======================================================"
log "Backup location: $BACKUP_DIR"
log "Files updated: $(echo "$FILES_TO_UPDATE" | wc -l)"
log "Remaining references: $REMAINING"
log ""
log "Next steps:"
log "1. Run: source ~/.bashrc (to update environment)"
log "2. Run: ./verify-sovren-paths.sh (to verify changes)"
log "3. Restart SOVREN services if running"
log "4. Test system functionality"
log ""
log "To restore from backup if needed:"
log "cp $BACKUP_DIR/* /home/ubuntu/"
log "======================================================"
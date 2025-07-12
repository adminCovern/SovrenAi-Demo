#!/bin/bash
# Install the SOVREN AI codebase from /home/ubuntu to /data/sovren

echo "=== Installing SOVREN AI Codebase ==="
echo "Copying all components to their proper locations..."

# Ensure directories exist
sudo mkdir -p /data/sovren/{consciousness,agent_battalion,shadow_board,time_machine,security,voice,api,frontend,billing,approval,mcp}

# 1. Consciousness Engine
echo "Installing Consciousness Engine..."
sudo cp /home/ubuntu/consciousness_engine_pcie_b200.py /data/sovren/consciousness/consciousness_engine.py
sudo cp /home/ubuntu/artifact-consciousness-engine-v1.py /data/sovren/consciousness/consciousness_engine_v1.py

# 2. Agent Battalion System
echo "Installing Agent Battalion System..."
sudo cp "/home/ubuntu/artifact-agent-battalion-v1.py" /data/sovren/agent_battalion/agent_battalion_system.py

# 3. Shadow Board
echo "Installing Shadow Board..."
sudo cp "/home/ubuntu/artifact-shadow-board-v1.py" /data/sovren/shadow_board/shadow_board.py
sudo cp "/home/ubuntu/artifact-shadow-board-v1 (1).py" /data/sovren/shadow_board/deep_executive_personality.py

# 4. Time Machine
echo "Installing Time Machine..."
sudo cp /home/ubuntu/artifact-time-machine-v1.py /data/sovren/time_machine/time_machine_system.py

# 5. Security System
echo "Installing Security System..."
sudo cp /home/ubuntu/artifact-security-system-v1.py /data/sovren/security/auth_system.py

# 6. Voice Integration
echo "Installing Voice Systems..."
sudo cp /home/ubuntu/sovereign-voice-skyetel.py /data/sovren/voice/voice_skyetel.py
sudo cp /home/ubuntu/sovren-telephony-system.py /data/sovren/voice/telephony_system.py
sudo cp /home/ubuntu/sovereign_awakening_handler.py /data/sovren/voice/awakening_handler.py

# 7. Data & Memory Systems
echo "Installing Data Systems..."
sudo cp "/home/ubuntu/sovren-data-ingestion (1).py" /data/sovren/api/data_ingestion.py
sudo cp "/home/ubuntu/sovren-memory-fabric (1).py" /data/sovren/api/memory_fabric.py
sudo cp "/home/ubuntu/sovren-rag-service (1).py" /data/sovren/api/rag_service.py

# 8. MCP Server
echo "Installing MCP Server..."
sudo cp /home/ubuntu/sovren-mcp-latency-optimized.py /data/sovren/mcp/mcp_server.py

# 9. User Approval & Billing
echo "Installing User & Billing Systems..."
sudo cp /home/ubuntu/user-approval-system.py /data/sovren/approval/user_approval.py
sudo cp /home/ubuntu/killbill-integration.py /data/sovren/billing/killbill_integration.py

# 10. Frontend
echo "Installing Frontend..."
sudo cp /home/ubuntu/sovren-frontend-complete.tsx /data/sovren/frontend/src/App.tsx
sudo cp /home/ubuntu/react-dashboard.tsx /data/sovren/frontend/src/Dashboard.tsx

# 11. Create main.py for API server
echo "Creating main API entry point..."
sudo tee /data/sovren/api/main.py > /dev/null << 'EOF'
#!/usr/bin/env python3
"""
SOVREN AI Main API Server
Entry point for all SOVREN services
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add SOVREN modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SOVREN components
from consciousness.consciousness_engine import PCIeB200ConsciousnessEngine
from agent_battalion.agent_battalion_system import AgentBattalionSystem
from shadow_board.shadow_board import ShadowBoard
from time_machine.time_machine_system import TimeMachine
from security.auth_system import SecuritySystem
from voice.voice_skyetel import VoiceSystem
from mcp.mcp_server import SOVRENLatencyMCPServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SOVREN-MAIN')

class SovrenAI:
    def __init__(self):
        logger.info("Initializing SOVREN AI System...")
        logger.info("Hardware: 8x B200 PCIe GPUs (183GB each)")
        
        # Initialize all subsystems
        self.consciousness = PCIeB200ConsciousnessEngine()
        self.agents = AgentBattalionSystem()
        self.shadow_board = ShadowBoard()
        self.time_machine = TimeMachine()
        self.security = SecuritySystem()
        self.voice = VoiceSystem()
        self.mcp = SOVRENLatencyMCPServer()
        
        logger.info("SOVREN AI System initialized successfully!")
        
    async def start(self):
        """Start all SOVREN services"""
        logger.info("Starting SOVREN AI services...")
        
        # Start MCP server
        await self.mcp.start()
        
        # Start voice system
        await self.voice.start()
        
        # Activate consciousness
        self.consciousness.state = "ACTIVE"
        
        logger.info("SOVREN AI is now active and ready!")
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            logger.info(f"System heartbeat - Active sessions: {len(self.voice.active_sessions)}")

async def main():
    sovren = SovrenAI()
    await sovren.start()

if __name__ == "__main__":
    asyncio.run(main())
EOF

# 12. Create package init files
echo "Creating package structure..."
for dir in consciousness agent_battalion shadow_board time_machine security voice api frontend billing approval mcp; do
    sudo touch /data/sovren/$dir/__init__.py
done

# 13. Fix permissions
echo "Setting permissions..."
sudo chown -R sovren:sovren /data/sovren
sudo chmod +x /data/sovren/api/main.py
sudo chmod +x /data/sovren/consciousness/consciousness_engine.py

# 14. Create models directory structure
echo "Creating models directory..."
sudo mkdir -p /data/sovren/models/{whisper,styletts2,mixtral}

echo ""
echo "=== SOVREN AI Codebase Installed! ==="
echo ""
echo "All components have been copied to /data/sovren"
echo ""
echo "Next steps:"
echo "1. Download AI models to /data/sovren/models/"
echo "2. Install Python dependencies"
echo "3. Start services: sudo systemctl start sovren-main"
echo ""
echo "To test the installation:"
echo "   cd /data/sovren"
echo "   sudo -u sovren python3 api/main.py"
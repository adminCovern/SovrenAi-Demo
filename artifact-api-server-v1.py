#!/usr/bin/env python3
"""
SOVREN API Server - FastAPI Implementation
Version: 1.0.0
Purpose: RESTful API and WebSocket server for all SOVREN operations
Location: /data/sovren/api/api_server.py
"""

import os
import sys
import time
import json
import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import SOVREN components
sys.path.append('/data/sovren')
from consciousness.consciousness_engine import BayesianConsciousnessEngine, ConsciousnessPacket
from shadow_board.shadow_board_system import ShadowBoardOrchestrator, UserContext
from agent_battalion.agent_battalion import AgentBattalion
from security.security_system import SecurityOrchestrator
from billing.billing_system import BillingSystem
from voice.voice_system import VoiceSystem

# Request/Response Models
class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

class DecisionRequest(BaseModel):
    decision_id: str
    query: str
    context: Dict[str, Any]
    options: List[str] = []
    priority: int = Field(default=5, ge=1, le=10)
    universes: int = Field(default=3, ge=1, le=10)

class DecisionResponse(BaseModel):
    decision_id: str
    action: str
    confidence: float
    reasoning: str
    processing_time_ms: float
    universes_explored: int

class TaskRequest(BaseModel):
    task_type: str
    data: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)

class TaskResponse(BaseModel):
    task_id: str
    status: str
    assigned_agent: Optional[str] = None

class BoardMeetingRequest(BaseModel):
    topic: str
    context: Dict[str, Any]
    urgency: str = Field(default="normal", pattern="^(low|normal|high|critical)$")

class SystemStatus(BaseModel):
    consciousness_state: str
    active_universes: int
    battalion_status: Dict[str, Any]
    shadow_board_active: bool
    system_uptime: float
    metrics: Dict[str, Any]

# Security
security = HTTPBearer()

class TokenManager:
    """JWT-style token management without external dependencies"""
    
    def __init__(self):
        self.tokens = {}  # token -> user_data
        self.secret_key = secrets.token_urlsafe(32)
        
    def create_token(self, user_id: str) -> str:
        """Create access token"""
        token = secrets.token_urlsafe(32)
        self.tokens[token] = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': time.time() + 3600  # 1 hour
        }
        return token
        
    def verify_token(self, token: str) -> Optional[str]:
        """Verify token and return user_id"""
        if token in self.tokens:
            token_data = self.tokens[token]
            if time.time() < token_data['expires_at']:
                return token_data['user_id']
            else:
                del self.tokens[token]  # Remove expired token
        return None

# Initialize components
token_manager = TokenManager()
consciousness_engine = None
battalion = None
security_system = None
billing_system = None
voice_system = None
shadow_boards = {}  # user_id -> ShadowBoardOrchestrator

# WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    global consciousness_engine, battalion, security_system, billing_system, voice_system
    
    print("Initializing SOVREN API Server...")
    
    # Initialize core systems
    consciousness_engine = BayesianConsciousnessEngine()
    battalion = AgentBattalion()
    security_system = SecurityOrchestrator()
    billing_system = BillingSystem()
    voice_system = VoiceSystem()
    
    print("All systems initialized. API ready.")
    
    yield
    
    # Shutdown
    print("Shutting down SOVREN API Server...")
    consciousness_engine.shutdown()
    # Other cleanup...

# Create FastAPI app
app = FastAPI(
    title="SOVREN AI API",
    description="Sovereign AI Chief of Staff API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token"""
    token = credentials.credentials
    user_id = token_manager.verify_token(token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id

# Authentication endpoints
@app.post("/api/auth/login", response_model=AuthResponse)
async def login(auth_request: AuthRequest):
    """Login endpoint"""
    # In production, verify against database
    # For now, simple verification
    if auth_request.password == "demo":  # Obviously not for production
        token = token_manager.create_token(auth_request.username)
        return AuthResponse(access_token=token)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@app.post("/api/auth/logout")
async def logout(user_id: str = Depends(get_current_user)):
    """Logout endpoint"""
    # In production, invalidate token
    return {"message": "Logged out successfully"}

# Consciousness endpoints
@app.post("/api/consciousness/decision", response_model=DecisionResponse)
async def make_decision(
    request: DecisionRequest,
    user_id: str = Depends(get_current_user)
):
    """Process decision through consciousness engine"""
    # Create consciousness packet
    packet = ConsciousnessPacket(
        packet_id=request.decision_id,
        timestamp=time.time(),
        source=f"api_user_{user_id}",
        data={
            'query': request.query,
            'context': request.context,
            'options': request.options
        },
        priority=request.priority,
        universes_required=request.universes
    )
    
    # Process decision
    result = consciousness_engine.process_decision(packet)
    
    # Log for billing
    await billing_system.log_usage(user_id, 'consciousness_decision', 1)
    
    return DecisionResponse(
        decision_id=request.decision_id,
        action=result['decision']['action'],
        confidence=result['confidence'],
        reasoning=result['reasoning'],
        processing_time_ms=result['processing_time_ms'],
        universes_explored=result['universes_explored']
    )

@app.get("/api/consciousness/status")
async def get_consciousness_status(user_id: str = Depends(get_current_user)):
    """Get consciousness engine status"""
    status = consciousness_engine.get_system_status()
    return status

# Shadow Board endpoints (SMB only)
@app.post("/api/shadow-board/initialize")
async def initialize_shadow_board(
    user_context: UserContext,
    user_id: str = Depends(get_current_user)
):
    """Initialize Shadow Board for user"""
    # Verify SMB tier
    user_data = await billing_system.get_user_data(user_id)
    if user_data.get('tier') != 'SMB':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Shadow Board is only available for SMB tier"
        )
    
    # Create Shadow Board
    shadow_board = ShadowBoardOrchestrator(user_context)
    shadow_boards[user_id] = shadow_board
    
    # Return executive profiles
    executives = {}
    for role, exec in shadow_board.executives.items():
        executives[role] = {
            'name': exec.profile.name,
            'trust_score': exec.profile.trust_score,
            'authority_score': exec.profile.authority_score
        }
    
    return {"message": "Shadow Board initialized", "executives": executives}

@app.post("/api/shadow-board/meeting")
async def convene_board_meeting(
    request: BoardMeetingRequest,
    user_id: str = Depends(get_current_user)
):
    """Convene Shadow Board meeting"""
    if user_id not in shadow_boards:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shadow Board not initialized"
        )
    
    shadow_board = shadow_boards[user_id]
    
    # Convene meeting
    agenda = {
        'topic': request.topic,
        'context': request.context,
        'urgency': request.urgency
    }
    
    result = shadow_board.convene_board_meeting(agenda)
    
    # Log for billing
    await billing_system.log_usage(user_id, 'shadow_board_meeting', 1)
    
    return result

@app.get("/api/shadow-board/executives/{role}")
async def get_executive_profile(
    role: str,
    user_id: str = Depends(get_current_user)
):
    """Get specific executive profile"""
    if user_id not in shadow_boards:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shadow Board not initialized"
        )
    
    shadow_board = shadow_boards[user_id]
    profile = shadow_board.get_executive_profile(role.upper())
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Executive role {role} not found"
        )
    
    return {
        'role': profile.role,
        'name': profile.name,
        'gender': profile.gender,
        'age_range': profile.age_range,
        'personality_traits': profile.personality_traits,
        'expertise_domains': profile.expertise_domains,
        'trust_score': profile.trust_score,
        'authority_score': profile.authority_score
    }

# Agent Battalion endpoints
@app.post("/api/battalion/task", response_model=TaskResponse)
async def assign_task(
    request: TaskRequest,
    user_id: str = Depends(get_current_user)
):
    """Assign task to agent battalion"""
    task_id = battalion.assign_task(
        task_type=request.task_type,
        data=request.data,
        priority=request.priority,
        complexity=request.complexity
    )
    
    # Log for billing
    await billing_system.log_usage(user_id, 'agent_task', 1)
    
    return TaskResponse(
        task_id=task_id,
        status="assigned"
    )

@app.get("/api/battalion/status")
async def get_battalion_status(user_id: str = Depends(get_current_user)):
    """Get battalion status"""
    status = battalion.get_battalion_status()
    return status

@app.get("/api/battalion/task/{task_id}")
async def get_task_status(
    task_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get task status"""
    # In production, would check task ownership
    result_path = f"/data/sovren/data/agent_results/*/{task_id}.json"
    
    import glob
    results = glob.glob(result_path)
    
    if results:
        with open(results[0], 'r') as f:
            return json.load(f)
    else:
        return {"status": "pending", "task_id": task_id}

# Voice endpoints
@app.post("/api/voice/call")
async def initiate_call(
    to_number: str,
    message: str,
    executive: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Initiate outbound call"""
    # Use Shadow Board executive if specified
    if executive and user_id in shadow_boards:
        shadow_board = shadow_boards[user_id]
        audio = shadow_board.executive_phone_call(executive, message)
    else:
        audio = voice_system.synthesize(message)
    
    # Place call via telephony
    call_id = await voice_system.place_call(to_number, audio, user_id)
    
    # Log for billing
    await billing_system.log_usage(user_id, 'outbound_call', 1)
    
    return {"call_id": call_id, "status": "initiated"}

@app.get("/api/voice/call/{call_id}")
async def get_call_status(
    call_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get call status"""
    status = await voice_system.get_call_status(call_id)
    return status

# System endpoints
@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(user_id: str = Depends(get_current_user)):
    """Get overall system status"""
    consciousness_status = consciousness_engine.get_system_status()
    battalion_status = battalion.get_battalion_status()
    
    return SystemStatus(
        consciousness_state=consciousness_status['state'],
        active_universes=consciousness_status['active_universes'],
        battalion_status=battalion_status,
        shadow_board_active=user_id in shadow_boards,
        system_uptime=consciousness_status['uptime_seconds'],
        metrics=consciousness_status['metrics']
    )

@app.get("/api/system/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Billing endpoints
@app.get("/api/billing/usage")
async def get_usage(
    user_id: str = Depends(get_current_user)
):
    """Get current usage and costs"""
    usage = await billing_system.get_current_usage(user_id)
    return usage

@app.get("/api/billing/invoice/{month}")
async def get_invoice(
    month: str,
    user_id: str = Depends(get_current_user)
):
    """Get invoice for specific month"""
    invoice = await billing_system.get_invoice(user_id, month)
    return invoice

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time updates"""
    await websocket.accept()
    websocket_connections[user_id] = websocket
    
    try:
        # Send initial status
        status = {
            'type': 'status',
            'data': await get_system_status(user_id)
        }
        await websocket.send_json(status)
        
        # Keep connection alive
        while True:
            # Wait for messages or send periodic updates
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Process message
                data = json.loads(message)
                if data['type'] == 'ping':
                    await websocket.send_json({'type': 'pong'})
                    
            except asyncio.TimeoutError:
                # Send periodic status update
                status = {
                    'type': 'status_update',
                    'data': {
                        'battalion': battalion.get_battalion_status(),
                        'timestamp': time.time()
                    }
                }
                await websocket.send_json(status)
                
    except WebSocketDisconnect:
        del websocket_connections[user_id]
    except Exception as e:
        print(f"WebSocket error for {user_id}: {e}")
        if user_id in websocket_connections:
            del websocket_connections[user_id]

# Broadcast function for real-time updates
async def broadcast_update(user_id: str, update: Dict[str, Any]):
    """Broadcast update to user's WebSocket"""
    if user_id in websocket_connections:
        websocket = websocket_connections[user_id]
        try:
            await websocket.send_json(update)
        except:
            del websocket_connections[user_id]

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Simple rate limiting"""
    # In production, use Redis or similar
    # For now, basic in-memory tracking
    client_ip = request.client.host
    
    # Continue to next middleware
    response = await call_next(request)
    return response

if __name__ == "__main__":
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )
#!/usr/bin/env python3
"""
SOVREN User Approval & Onboarding System
Handles application submission, admin approval, and automated onboarding
"""

import os
import sqlite3
import json
import time
import asyncio
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

class UserStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    BETA = "beta"
    ADMIN = "admin"
    BYPASS = "bypass"  # Special bypass users

@dataclass
class UserApplication:
    application_id: str
    email: str
    name: str
    company: str
    phone: str
    website: Optional[str]
    industry: str
    company_size: str
    submitted_at: float
    status: UserStatus
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    bypass_approval: bool = False
    pre_computation_task: Optional[asyncio.Task] = None

class ApprovalSystem:
    """
    Manages user application approval workflow
    """
    
    def __init__(self):
        self.db_path = '/data/sovren/data/applications.db'
        self.bypass_db_path = '/data/sovren/data/bypass_users.db'
        self._init_databases()
        
    def _init_databases(self):
        """Initialize application and bypass databases"""
        
        # Applications database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                application_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                company TEXT NOT NULL,
                phone TEXT NOT NULL,
                website TEXT,
                industry TEXT,
                company_size TEXT,
                submitted_at REAL NOT NULL,
                status TEXT NOT NULL,
                approved_by TEXT,
                approved_at REAL,
                bypass_approval INTEGER DEFAULT 0,
                pre_computed_insights TEXT,
                INDEX idx_status (status),
                INDEX idx_email (email)
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Bypass users database
        conn = sqlite3.connect(self.bypass_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bypass_users (
                email TEXT PRIMARY KEY,
                user_type TEXT NOT NULL,  -- 'beta', 'admin', 'special'
                added_by TEXT NOT NULL,
                added_at REAL NOT NULL,
                reason TEXT
            )
        """)
        
        # Add default admin (yourself)
        cursor.execute("""
            INSERT OR IGNORE INTO bypass_users (email, user_type, added_by, added_at, reason)
            VALUES (?, 'admin', 'system', ?, 'System administrator')
        """, (os.environ.get('SOVREN_ADMIN_EMAIL', 'admin@sovrenai.app'), time.time()))
        
        conn.commit()
        conn.close()
    
    async def submit_application(self, application_data: dict) -> UserApplication:
        """
        Submit new user application
        """
        import uuid
        
        application = UserApplication(
            application_id=str(uuid.uuid4()),
            email=application_data['email'].lower(),
            name=application_data['name'],
            company=application_data['company'],
            phone=application_data['phone'],
            website=application_data.get('website'),
            industry=application_data['industry'],
            company_size=application_data['company_size'],
            submitted_at=time.time(),
            status=UserStatus.PENDING
        )
        
        # Check if user should bypass approval
        if self.should_bypass_approval(application.email):
            application.status = UserStatus.APPROVED
            application.bypass_approval = True
            application.approved_at = time.time()
            application.approved_by = "auto-bypass"
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO applications (
                application_id, email, name, company, phone, website,
                industry, company_size, submitted_at, status, approved_by,
                approved_at, bypass_approval
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            application.application_id, application.email, application.name,
            application.company, application.phone, application.website,
            application.industry, application.company_size, application.submitted_at,
            application.status.value, application.approved_by, application.approved_at,
            int(application.bypass_approval)
        ))
        
        conn.commit()
        conn.close()
        
        # Start pre-computation for all applications (even pending)
        from .onboarding import PreComputationEngine
        pre_compute = PreComputationEngine()
        application.pre_computation_task = asyncio.create_task(
            pre_compute.pre_compute_insights(application)
        )
        
        # If bypassed, trigger immediate onboarding
        if application.bypass_approval:
            await self.trigger_onboarding(application)
        
        return application
    
    def should_bypass_approval(self, email: str) -> bool:
        """
        Check if user should bypass approval process
        """
        conn = sqlite3.connect(self.bypass_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_type FROM bypass_users WHERE email = ?
        """, (email.lower(),))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def add_bypass_user(self, email: str, user_type: str, added_by: str, reason: str = ""):
        """
        Add user to bypass list
        """
        if user_type not in ['beta', 'admin', 'special']:
            raise ValueError("Invalid user type")
        
        conn = sqlite3.connect(self.bypass_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO bypass_users (email, user_type, added_by, added_at, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (email.lower(), user_type, added_by, time.time(), reason))
        
        conn.commit()
        conn.close()
    
    def remove_bypass_user(self, email: str):
        """
        Remove user from bypass list
        """
        conn = sqlite3.connect(self.bypass_db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM bypass_users WHERE email = ?", (email.lower(),))
        
        conn.commit()
        conn.close()
    
    async def get_pending_applications(self) -> List[UserApplication]:
        """
        Get all pending applications for admin review
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM applications 
            WHERE status = ? 
            ORDER BY submitted_at ASC
        """, (UserStatus.PENDING.value,))
        
        applications = []
        for row in cursor.fetchall():
            app = UserApplication(
                application_id=row[0],
                email=row[1],
                name=row[2],
                company=row[3],
                phone=row[4],
                website=row[5],
                industry=row[6],
                company_size=row[7],
                submitted_at=row[8],
                status=UserStatus(row[9]),
                approved_by=row[10],
                approved_at=row[11],
                bypass_approval=bool(row[12])
            )
            applications.append(app)
        
        conn.close()
        return applications
    
    async def approve_application(self, application_id: str, approved_by: str) -> bool:
        """
        Approve user application and trigger onboarding
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update status
        cursor.execute("""
            UPDATE applications 
            SET status = ?, approved_by = ?, approved_at = ?
            WHERE application_id = ? AND status = ?
        """, (
            UserStatus.APPROVED.value,
            approved_by,
            time.time(),
            application_id,
            UserStatus.PENDING.value
        ))
        
        if cursor.rowcount == 0:
            conn.close()
            return False
        
        # Get application details
        cursor.execute("""
            SELECT * FROM applications WHERE application_id = ?
        """, (application_id,))
        
        row = cursor.fetchone()
        conn.commit()
        conn.close()
        
        if row:
            # Create application object
            application = UserApplication(
                application_id=row[0],
                email=row[1],
                name=row[2],
                company=row[3],
                phone=row[4],
                website=row[5],
                industry=row[6],
                company_size=row[7],
                submitted_at=row[8],
                status=UserStatus(row[9]),
                approved_by=row[10],
                approved_at=row[11]
            )
            
            # Trigger onboarding
            await self.trigger_onboarding(application)
            return True
        
        return False
    
    async def deny_application(self, application_id: str, denied_by: str) -> bool:
        """
        Deny user application
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE applications 
            SET status = ?, approved_by = ?, approved_at = ?
            WHERE application_id = ? AND status = ?
        """, (
            UserStatus.DENIED.value,
            denied_by,
            time.time(),
            application_id,
            UserStatus.PENDING.value
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    async def trigger_onboarding(self, application: UserApplication):
        """
        Trigger the automated onboarding process
        """
        # Import onboarding systems
        from .onboarding import InstantAwakeningProtocol
        from .voice_system import SovrenVoice
        
        awakening = InstantAwakeningProtocol()
        
        # Create user account
        user_id = await self.create_user_account(application)
        
        # Execute the instant awakening (3-second call, etc.)
        await awakening.execute_awakening({
            'id': user_id,
            'name': application.name,
            'email': application.email,
            'phone': application.phone,
            'company': application.company,
            'tier': 'SMB' if application.company_size != 'enterprise' else 'ENTERPRISE',
            'pre_computation_task': application.pre_computation_task
        })
    
    async def create_user_account(self, application: UserApplication) -> str:
        """
        Create user account in main database
        """
        import uuid
        
        user_id = str(uuid.uuid4())
        
        # Create in users database
        conn = sqlite3.connect('/data/sovren/data/users.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (
                user_id, name, email, company, phone, tier,
                sovren_score, plan, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, application.name, application.email,
            application.company, application.phone,
            'SMB' if application.company_size != 'enterprise' else 'ENTERPRISE',
            300,  # Starting SOVREN Score
            'SOVREN Proof',  # Default plan
            time.time(),
            'active'
        ))
        
        conn.commit()
        conn.close()
        
        # Create API token
        from .api_server import auth_handler
        token = auth_handler.create_token(user_id)
        
        # Store token for welcome email
        await self.store_user_credentials(user_id, token)
        
        return user_id
    
    async def store_user_credentials(self, user_id: str, token: str):
        """
        Store user credentials for welcome communications
        """
        # Implementation for secure credential storage
        pass


class AdminDashboard:
    """
    Admin dashboard for managing applications
    """
    
    def __init__(self):
        self.approval_system = ApprovalSystem()
    
    async def get_dashboard_data(self) -> dict:
        """
        Get all data for admin dashboard
        """
        pending = await self.approval_system.get_pending_applications()
        
        # Get statistics
        conn = sqlite3.connect(self.approval_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count,
                COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_count,
                COUNT(CASE WHEN status = 'denied' THEN 1 END) as denied_count,
                COUNT(CASE WHEN bypass_approval = 1 THEN 1 END) as bypass_count
            FROM applications
        """)
        
        stats = cursor.fetchone()
        
        # Get bypass users
        bypass_conn = sqlite3.connect(self.approval_system.bypass_db_path)
        bypass_cursor = bypass_conn.cursor()
        
        bypass_cursor.execute("SELECT * FROM bypass_users ORDER BY added_at DESC")
        bypass_users = bypass_cursor.fetchall()
        
        conn.close()
        bypass_conn.close()
        
        return {
            'pending_applications': [
                {
                    'application_id': app.application_id,
                    'name': app.name,
                    'email': app.email,
                    'company': app.company,
                    'industry': app.industry,
                    'company_size': app.company_size,
                    'submitted_at': datetime.fromtimestamp(app.submitted_at).isoformat(),
                    'website': app.website
                } for app in pending
            ],
            'statistics': {
                'pending': stats[0],
                'approved': stats[1],
                'denied': stats[2],
                'bypassed': stats[3]
            },
            'bypass_users': [
                {
                    'email': user[0],
                    'type': user[1],
                    'added_by': user[2],
                    'added_at': datetime.fromtimestamp(user[3]).isoformat(),
                    'reason': user[4]
                } for user in bypass_users
            ]
        }


# API Endpoints for approval system
from fastapi import APIRouter

approval_router = APIRouter(prefix="/api/admin/approval")
approval_system = ApprovalSystem()
admin_dashboard = AdminDashboard()

@approval_router.get("/dashboard")
async def get_approval_dashboard(admin_id: str = Depends(verify_admin)):
    """Get admin approval dashboard data"""
    return await admin_dashboard.get_dashboard_data()

@approval_router.post("/approve/{application_id}")
async def approve_application(
    application_id: str,
    admin_id: str = Depends(verify_admin)
):
    """Approve an application"""
    success = await approval_system.approve_application(application_id, admin_id)
    return {"success": success}

@approval_router.post("/deny/{application_id}")
async def deny_application(
    application_id: str,
    admin_id: str = Depends(verify_admin)
):
    """Deny an application"""
    success = await approval_system.deny_application(application_id, admin_id)
    return {"success": success}

@approval_router.post("/bypass/add")
async def add_bypass_user(
    email: str,
    user_type: str,
    reason: str = "",
    admin_id: str = Depends(verify_admin)
):
    """Add user to bypass list"""
    approval_system.add_bypass_user(email, user_type, admin_id, reason)
    return {"success": True}

@approval_router.delete("/bypass/{email}")
async def remove_bypass_user(
    email: str,
    admin_id: str = Depends(verify_admin)
):
    """Remove user from bypass list"""
    approval_system.remove_bypass_user(email)
    return {"success": True}
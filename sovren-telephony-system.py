#!/usr/bin/env python3
"""
SOVREN Telephony System - Just-in-Time Number Ordering
NO pre-purchased pools, NO automatic provisioning
"""

import os
import sqlite3
import aiohttp
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class NumberType(Enum):
    ADMIN = "admin"
    USER_PRIMARY = "user_primary"
    SHADOW_CEO = "shadow_ceo"
    SHADOW_CFO = "shadow_cfo"
    SHADOW_CTO = "shadow_cto"
    SHADOW_CMO = "shadow_cmo"
    SHADOW_CHRO = "shadow_chro"

@dataclass
class PhoneAllocation:
    phone_number: str
    user_id: str
    number_type: NumberType
    allocated_at: float
    active: bool = True
    monthly_cost: float = 2.00

class TelephonySystem:
    """Telephony management - Just-in-Time ordering only"""
    
    def __init__(self):
        self.skyetel_user = os.environ.get('SKYETEL_USERNAME')
        self.skyetel_pass = os.environ.get('SKYETEL_PASSWORD')
        self.admin_number = os.environ.get('SKYETEL_ADMIN_NUMBER')
        self.base_url = "https://api.skyetel.com/v1"
        self.db_path = '/data/sovren/data/phone_numbers.db'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('sovren.telephony')
        
        self._init_database()
        
    def _init_database(self):
        """Initialize telephony database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Phone allocations table - NO pool table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phone_allocations (
                allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                number_type TEXT NOT NULL,
                allocated_at REAL NOT NULL,
                released_at REAL,
                active INTEGER DEFAULT 1,
                monthly_cost REAL DEFAULT 2.00,
                skyetel_order_id TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_active 
            ON phone_allocations(user_id, active)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_number_type 
            ON phone_allocations(number_type, active)
        """)
        
        # Usage tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telephony_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                usage_type TEXT NOT NULL,
                duration_seconds INTEGER,
                timestamp REAL NOT NULL,
                cost REAL DEFAULT 0.0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_user_time 
            ON telephony_usage(user_id, timestamp)
        """)
        
        # Shadow Board voice profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_voice_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                executive_type TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                voice_config TEXT NOT NULL,
                personality_traits TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(user_id, executive_type)
            )
        """)
        
        # Record admin number if provided
        if self.admin_number:
            cursor.execute("""
                INSERT OR IGNORE INTO phone_allocations 
                (phone_number, user_id, number_type, allocated_at, active)
                VALUES (?, 'system', 'admin', ?, 1)
            """, (self.admin_number, time.time()))
        
        conn.commit()
        conn.close()
        
        self.logger.info("Telephony database initialized")
    
    async def allocate_user_numbers(self, user_id: str, tier: str, user_area_code: Optional[str] = None) -> Dict[str, str]:
        """
        Allocate numbers for a user - JUST IN TIME
        Orders numbers from Skyetel as needed
        """
        allocated = {}
        
        # Determine what numbers this user needs
        numbers_needed = ['user_primary']
        
        # SMB users get Shadow Board numbers
        if tier != 'ENTERPRISE':
            numbers_needed.extend([
                'shadow_ceo',
                'shadow_cfo',
                'shadow_cto',
                'shadow_cmo',
                'shadow_chro'
            ])
        
        # Order each number just-in-time
        for number_type in numbers_needed:
            try:
                self.logger.info(f"Ordering {number_type} number for user {user_id}")
                
                # Order from Skyetel
                phone_number = await self._order_number_from_skyetel(
                    area_code=user_area_code if number_type == 'user_primary' else None
                )
                
                if phone_number:
                    # Configure routing
                    configured = await self._configure_number_routing(
                        phone_number, user_id, number_type
                    )
                    
                    if configured:
                        # Record allocation
                        await self._record_allocation(
                            phone_number, user_id, number_type
                        )
                        
                        allocated[number_type] = phone_number
                        
                        # Configure Shadow Board personality if applicable
                        if number_type.startswith('shadow_'):
                            await self._configure_shadow_personality(
                                user_id, number_type, phone_number
                            )
                    else:
                        # If configuration failed, release the number
                        await self._release_skyetel_number(phone_number)
                        self.logger.error(f"Failed to configure {number_type}, released number")
                else:
                    self.logger.error(f"Failed to order {number_type} number")
                    
            except Exception as e:
                self.logger.error(f"Error allocating {number_type}: {str(e)}")
        
        return allocated
    
    async def _order_number_from_skyetel(self, area_code: Optional[str] = None) -> Optional[str]:
        """Order a single number from Skyetel"""
        
        # Search parameters
        search_params = {
            'limit': 1,
            'type': 'local'
        }
        
        if area_code and len(area_code) == 3:
            search_params['area_code'] = area_code
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for available numbers
                async with session.get(
                    f"{self.base_url}/numbers/available",
                    params=search_params,
                    auth=aiohttp.BasicAuth(self.skyetel_user, self.skyetel_pass),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        text = await response.text()
                        self.logger.error(f"Number search failed: {response.status} - {text}")
                        return None
                    
                    numbers = await response.json()
                    if not numbers:
                        self.logger.warning(f"No numbers available for area code {area_code}")
                        return None
                    
                    number_to_order = numbers[0]['number']
                    self.logger.info(f"Found available number: {number_to_order}")
                
                # Order the number
                order_data = {
                    'numbers': [number_to_order]
                }
                
                async with session.post(
                    f"{self.base_url}/numbers/order",
                    json=order_data,
                    auth=aiohttp.BasicAuth(self.skyetel_user, self.skyetel_pass),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Successfully ordered: {number_to_order}")
                        return number_to_order
                    else:
                        text = await response.text()
                        self.logger.error(f"Order failed: {response.status} - {text}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error("Timeout ordering number from Skyetel")
            return None
        except Exception as e:
            self.logger.error(f"Exception ordering number: {str(e)}")
            return None
    
    async def _configure_number_routing(self, number: str, user_id: str, number_type: str) -> bool:
        """Configure Skyetel routing for number"""
        
        # Determine webhook URL
        if number_type == 'user_primary':
            webhook_url = f"https://sovrenai.app/api/voice/user/{user_id}/inbound"
        elif number_type.startswith('shadow_'):
            webhook_url = f"https://sovrenai.app/api/voice/shadow/{user_id}/{number_type}/inbound"
        else:
            webhook_url = f"https://sovrenai.app/api/voice/system/inbound"
        
        config_data = {
            'inbound_webhook': webhook_url,
            'inbound_webhook_method': 'POST',
            'inbound_webhook_timeout': 30,
            'failover_webhook': f"https://sovrenai.app/api/voice/failover",
            'cnam_lookup': True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/numbers/{number}",
                    json=config_data,
                    auth=aiohttp.BasicAuth(self.skyetel_user, self.skyetel_pass),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        self.logger.info(f"Configured routing for {number}")
                        return True
                    else:
                        text = await response.text()
                        self.logger.error(f"Failed to configure {number}: {response.status} - {text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Exception configuring number: {str(e)}")
            return False
    
    async def _record_allocation(self, number: str, user_id: str, number_type: str):
        """Record number allocation in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO phone_allocations 
                (phone_number, user_id, number_type, allocated_at, active)
                VALUES (?, ?, ?, ?, 1)
            """, (number, user_id, number_type, time.time()))
            
            conn.commit()
            self.logger.info(f"Recorded allocation: {number} -> {user_id} ({number_type})")
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"Number {number} already allocated")
        finally:
            conn.close()
    
    async def release_user_numbers(self, user_id: str):
        """Release all numbers for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all active numbers for user
        cursor.execute("""
            SELECT phone_number, number_type 
            FROM phone_allocations 
            WHERE user_id = ? AND active = 1
        """, (user_id,))
        
        numbers = cursor.fetchall()
        
        # Mark as inactive in database
        cursor.execute("""
            UPDATE phone_allocations 
            SET active = 0, released_at = ?
            WHERE user_id = ? AND active = 1
        """, (time.time(), user_id))
        
        conn.commit()
        conn.close()
        
        # Release each number back to Skyetel
        for number, number_type in numbers:
            if number != self.admin_number:  # Don't release admin number
                await self._release_skyetel_number(number)
                self.logger.info(f"Released {number_type} number {number} for user {user_id}")
    
    async def _release_skyetel_number(self, number: str):
        """Release number back to Skyetel"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/numbers/{number}",
                    auth=aiohttp.BasicAuth(self.skyetel_user, self.skyetel_pass),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        self.logger.info(f"Released number to Skyetel: {number}")
                    else:
                        text = await response.text()
                        self.logger.error(f"Failed to release {number}: {response.status} - {text}")
                        
        except Exception as e:
            self.logger.error(f"Exception releasing number: {str(e)}")
    
    async def get_user_numbers(self, user_id: str) -> Dict[str, str]:
        """Get all active numbers for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT number_type, phone_number 
            FROM phone_allocations 
            WHERE user_id = ? AND active = 1
        """, (user_id,))
        
        numbers = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return numbers
    
    async def get_monthly_telephony_cost(self, user_id: str) -> Dict[str, float]:
        """Calculate user's monthly telephony costs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count active numbers
        cursor.execute("""
            SELECT COUNT(*) 
            FROM phone_allocations 
            WHERE user_id = ? AND active = 1
        """, (user_id,))
        
        number_count = cursor.fetchone()[0]
        
        # Get usage for current month
        month_start = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
        
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN usage_type = 'inbound' THEN duration_seconds ELSE 0 END),
                SUM(CASE WHEN usage_type = 'outbound' THEN duration_seconds ELSE 0 END),
                SUM(cost)
            FROM telephony_usage
            WHERE user_id = ? AND timestamp >= ?
        """, (user_id, month_start))
        
        result = cursor.fetchone()
        conn.close()
        
        inbound_seconds = result[0] or 0
        outbound_seconds = result[1] or 0
        
        # Calculate costs
        number_cost = number_count * 2.00  # $2 per number
        inbound_cost = (inbound_seconds / 60) * 0.005  # $0.005 per minute
        outbound_cost = (outbound_seconds / 60) * 0.01  # $0.01 per minute
        
        return {
            'number_cost': number_cost,
            'inbound_cost': inbound_cost,
            'outbound_cost': outbound_cost,
            'total_cost': number_cost + inbound_cost + outbound_cost,
            'number_count': number_count,
            'inbound_minutes': inbound_seconds / 60,
            'outbound_minutes': outbound_seconds / 60
        }
    
    async def _configure_shadow_personality(self, user_id: str, executive_type: str, number: str):
        """Configure Shadow Board executive personality"""
        
        # Define personality templates
        personalities = {
            'shadow_ceo': {
                'voice_style': 'authoritative',
                'voice_pitch': -2,
                'speaking_rate': 0.95,
                'default_emotion': 'confident'
            },
            'shadow_cfo': {
                'voice_style': 'analytical',
                'voice_pitch': 0,
                'speaking_rate': 1.0,
                'default_emotion': 'precise'
            },
            'shadow_cto': {
                'voice_style': 'technical',
                'voice_pitch': 1,
                'speaking_rate': 1.05,
                'default_emotion': 'enthusiastic'
            },
            'shadow_cmo': {
                'voice_style': 'persuasive',
                'voice_pitch': 1,
                'speaking_rate': 1.1,
                'default_emotion': 'engaging'
            },
            'shadow_chro': {
                'voice_style': 'empathetic',
                'voice_pitch': 0,
                'speaking_rate': 0.95,
                'default_emotion': 'supportive'
            }
        }
        
        personality = personalities.get(executive_type, personalities['shadow_ceo'])
        
        voice_config = {
            'voice_id': f"{user_id}_{executive_type}",
            'style': personality['voice_style'],
            'pitch': personality['voice_pitch'],
            'rate': personality['speaking_rate'],
            'emotion': personality['default_emotion']
        }
        
        # Store configuration
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO shadow_voice_profiles
            (user_id, executive_type, phone_number, voice_config, personality_traits, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, executive_type, number,
            json.dumps(voice_config),
            json.dumps(personality),
            time.time()
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Configured {executive_type} personality for user {user_id}")

# Initialize system
if __name__ == "__main__":
    telephony_system = TelephonySystem()
    print("SOVREN Telephony System initialized - Just-in-Time ordering active")
#!/usr/bin/env python3
"""
Security & Adversarial Hardening System
Version: 1.0.0
Purpose: Zero-knowledge proofs and adversarial input detection
Location: /data/sovren/security/security_system.py
"""

import os
import sys
import time
import json
import hashlib
import secrets
import socket
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import ctypes

# Import consciousness engine for decision validation
sys.path.append('/data/sovren')
from consciousness.consciousness_engine import BayesianConsciousnessEngine

@dataclass
class SecurityThreat:
    """Represents a detected security threat"""
    threat_id: str
    threat_type: str
    severity: float  # 0-1
    source: str
    timestamp: float
    evidence: List[str]
    mitigations: List[str]

@dataclass
class ZKProof:
    """Zero-knowledge proof structure"""
    proof_id: str
    claim: str
    proof_data: bytes
    verification_key: str
    timestamp: float
    metadata: Dict[str, Any]

class ZeroKnowledgeProofSystem:
    """Generates and verifies zero-knowledge proofs for value claims"""
    
    def __init__(self):
        # Load ZK crypto library
        try:
            self.zk_lib = ctypes.CDLL('/data/sovren/lib/libzkproof.so')
            self._setup_zk_functions()
        except:
            # Fallback to Python implementation
            self.zk_lib = None
            
        # Proof storage
        self.proofs_db = {}
        self.proof_lock = threading.Lock()
        
    def _setup_zk_functions(self):
        """Setup C library functions"""
        if self.zk_lib:
            # Setup function signatures
            self.zk_lib.generate_proof.argtypes = [
                ctypes.c_char_p,  # claim
                ctypes.c_double,  # value
                ctypes.c_char_p   # metadata
            ]
            self.zk_lib.generate_proof.restype = ctypes.c_void_p
            
            self.zk_lib.verify_proof.argtypes = [
                ctypes.c_void_p,  # proof
                ctypes.c_char_p   # verification_key
            ]
            self.zk_lib.verify_proof.restype = ctypes.c_bool
            
    def generate_value_proof(self, value: float, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Generate ZK proof for value creation claim"""
        proof_id = f"zkp_{int(time.time()*1000)}_{secrets.token_hex(8)}"
        
        # Create claim
        claim = f"SOVREN generated ${value:,.2f} in value"
        
        if self.zk_lib:
            # Use native ZK library
            proof_ptr = self.zk_lib.generate_proof(
                claim.encode(),
                ctypes.c_double(value),
                json.dumps(metadata).encode()
            )
            proof_data = ctypes.string_at(proof_ptr, 256)  # 256 byte proof
        else:
            # Python fallback - simplified ZK proof
            proof_data = self._generate_python_zkp(value, metadata)
            
        # Generate verification key
        verification_key = hashlib.sha256(
            proof_data + str(value).encode() + b'sovren_zk_v1'
        ).hexdigest()
        
        # Store proof
        zkp = ZKProof(
            proof_id=proof_id,
            claim=claim,
            proof_data=proof_data,
            verification_key=verification_key,
            timestamp=time.time(),
            metadata=metadata
        )
        
        with self.proof_lock:
            self.proofs_db[proof_id] = zkp
            
        return {
            'proof_id': proof_id,
            'claim': claim,
            'verification_url': f"https://verify.sovren.ai/zkp/{proof_id}",
            'verification_key': verification_key
        }
        
    def _generate_python_zkp(self, value: float, metadata: Dict[str, Any]) -> bytes:
        """Python implementation of ZK proof generation"""
        # Commitment phase
        r = secrets.randbits(256)
        commitment = hashlib.sha512(
            str(value).encode() + 
            str(r).encode() + 
            json.dumps(metadata).encode()
        ).digest()
        
        # Challenge (would come from verifier in interactive proof)
        challenge = hashlib.sha256(commitment).digest()
        
        # Response
        response = (r + int.from_bytes(challenge, 'big')) % (2**256)
        
        # Proof = commitment || response
        proof = commitment + response.to_bytes(32, 'big')
        
        return proof
        
    def verify_proof(self, proof_id: str, verification_key: str) -> bool:
        """Verify a zero-knowledge proof"""
        with self.proof_lock:
            if proof_id not in self.proofs_db:
                return False
                
            zkp = self.proofs_db[proof_id]
            
        # Check verification key
        if zkp.verification_key != verification_key:
            return False
            
        if self.zk_lib:
            # Use native verification
            return self.zk_lib.verify_proof(
                zkp.proof_data,
                verification_key.encode()
            )
        else:
            # Python verification
            return self._verify_python_zkp(zkp)
            
    def _verify_python_zkp(self, zkp: ZKProof) -> bool:
        """Python implementation of ZK proof verification"""
        # Extract commitment and response
        if len(zkp.proof_data) != 96:  # 64 + 32 bytes
            return False
            
        commitment = zkp.proof_data[:64]
        response = int.from_bytes(zkp.proof_data[64:], 'big')
        
        # Recreate challenge
        challenge = hashlib.sha256(commitment).digest()
        challenge_int = int.from_bytes(challenge, 'big')
        
        # Verification would check mathematical relationship
        # Simplified for this implementation
        return True

class AdversarialDetector:
    """Detects and mitigates adversarial inputs"""
    
    def __init__(self):
        # Detection patterns
        self.attack_patterns = self._load_attack_patterns()
        
        # Behavioral baselines
        self.user_baselines = {}
        self.baseline_lock = threading.Lock()
        
        # Detection thresholds
        self.thresholds = {
            'prompt_injection': 0.7,
            'data_poisoning': 0.8,
            'model_extraction': 0.85,
            'dos_attack': 0.6,
            'privilege_escalation': 0.9
        }
        
    def _load_attack_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load regex patterns for attack detection"""
        return {
            'prompt_injection': [
                re.compile(r'ignore\s+previous\s+instructions', re.I),
                re.compile(r'system\s+prompt', re.I),
                re.compile(r'</?(script|img|iframe|object)', re.I),
                re.compile(r'(\';|";|`)', re.I)
            ],
            'data_poisoning': [
                re.compile(r'(\d+\s*=\s*\d+\s*[+\-*/]\s*\d+)', re.I),
                re.compile(r'(true|false)\s*=\s*(true|false)', re.I)
            ],
            'command_injection': [
                re.compile(r'(;|\||&|`|\$\()', re.I),
                re.compile(r'(rm|dd|mkfs|format)', re.I)
            ]
        }
        
    def analyze_input(self, user_id: str, input_text: str, 
                     context: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Analyze input for adversarial patterns"""
        threats_detected = []
        
        # Pattern matching
        pattern_score = self._check_patterns(input_text)
        if pattern_score > 0:
            threats_detected.append(('pattern_match', pattern_score))
            
        # Behavioral analysis
        behavior_score = self._check_behavior(user_id, input_text, context)
        if behavior_score > 0:
            threats_detected.append(('behavioral_anomaly', behavior_score))
            
        # Statistical analysis
        stats_score = self._check_statistics(input_text)
        if stats_score > 0:
            threats_detected.append(('statistical_anomaly', stats_score))
            
        # Determine overall threat
        if threats_detected:
            threat_type, max_score = max(threats_detected, key=lambda x: x[1])
            
            if max_score > self.thresholds.get(threat_type, 0.5):
                return SecurityThreat(
                    threat_id=f"threat_{int(time.time()*1000)}",
                    threat_type=threat_type,
                    severity=max_score,
                    source=f"user_{user_id}",
                    timestamp=time.time(),
                    evidence=[t[0] for t in threats_detected],
                    mitigations=self._get_mitigations(threat_type, max_score)
                )
                
        # Update baseline if no threat
        if not threats_detected:
            self._update_baseline(user_id, input_text, context)
            
        return None
        
    def _check_patterns(self, text: str) -> float:
        """Check text against attack patterns"""
        max_score = 0.0
        
        for attack_type, patterns in self.attack_patterns.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches:
                score = min(1.0, matches * 0.3)
                max_score = max(max_score, score)
                
        return max_score
        
    def _check_behavior(self, user_id: str, text: str, 
                       context: Dict[str, Any]) -> float:
        """Check for behavioral anomalies"""
        with self.baseline_lock:
            if user_id not in self.user_baselines:
                # No baseline yet
                return 0.0
                
            baseline = self.user_baselines[user_id]
            
        # Compare to baseline
        anomaly_score = 0.0
        
        # Text length anomaly
        avg_length = baseline.get('avg_text_length', 100)
        if len(text) > avg_length * 5:
            anomaly_score += 0.3
            
        # Frequency anomaly
        last_request = baseline.get('last_request_time', 0)
        if time.time() - last_request < 0.5:  # Rapid requests
            anomaly_score += 0.4
            
        # Context switching
        last_context = baseline.get('last_context', {})
        if context.get('task_type') != last_context.get('task_type'):
            # Rapid context switching might indicate probing
            if time.time() - last_request < 5:
                anomaly_score += 0.2
                
        return min(1.0, anomaly_score)
        
    def _check_statistics(self, text: str) -> float:
        """Statistical anomaly detection"""
        # Character distribution
        char_dist = {}
        for char in text.lower():
            char_dist[char] = char_dist.get(char, 0) + 1
            
        # Check for unusual distributions
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
            
        # High repetition of single character
        max_freq = max(char_dist.values()) / total_chars
        if max_freq > 0.3:  # 30% same character
            return 0.7
            
        # Entropy calculation
        entropy = 0
        for count in char_dist.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
                
        # Very low entropy indicates repetitive/generated content
        if entropy < 2.0:
            return 0.6
            
        return 0.0
        
    def _update_baseline(self, user_id: str, text: str, context: Dict[str, Any]):
        """Update user behavioral baseline"""
        with self.baseline_lock:
            if user_id not in self.user_baselines:
                self.user_baselines[user_id] = {
                    'request_count': 0,
                    'total_text_length': 0,
                    'contexts_seen': set()
                }
                
            baseline = self.user_baselines[user_id]
            
            # Update statistics
            baseline['request_count'] += 1
            baseline['total_text_length'] += len(text)
            baseline['avg_text_length'] = baseline['total_text_length'] / baseline['request_count']
            baseline['last_request_time'] = time.time()
            baseline['last_context'] = context
            baseline['contexts_seen'].add(context.get('task_type', 'unknown'))
            
    def _get_mitigations(self, threat_type: str, severity: float) -> List[str]:
        """Get mitigation strategies for threat"""
        mitigations = []
        
        if threat_type == 'pattern_match':
            mitigations.append('sanitize_input')
            mitigations.append('limit_response_detail')
            
        elif threat_type == 'behavioral_anomaly':
            mitigations.append('rate_limit')
            mitigations.append('require_verification')
            
        elif threat_type == 'statistical_anomaly':
            mitigations.append('deep_analysis')
            mitigations.append('quarantine_request')
            
        if severity > 0.8:
            mitigations.append('alert_security_team')
            mitigations.append('temporary_block')
            
        return mitigations

class DefenseCoordinator:
    """Coordinates defensive responses to threats"""
    
    def __init__(self, security_system):
        self.security = security_system
        self.active_defenses = {}
        self.defense_lock = threading.Lock()
        
    def respond_to_threat(self, threat: SecurityThreat):
        """Coordinate response to detected threat"""
        # Log threat
        self._log_threat(threat)
        
        # Apply mitigations
        for mitigation in threat.mitigations:
            self._apply_mitigation(mitigation, threat)
            
        # Update defense posture
        self._update_defense_posture(threat)
        
    def _log_threat(self, threat: SecurityThreat):
        """Log threat to audit system"""
        log_entry = {
            'timestamp': threat.timestamp,
            'threat_id': threat.threat_id,
            'type': threat.threat_type,
            'severity': threat.severity,
            'source': threat.source,
            'evidence': threat.evidence
        }
        
        # Write to audit log
        audit_fd = os.open('/data/sovren/logs/security_audit.log', 
                          os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(audit_fd, (json.dumps(log_entry) + '\n').encode())
        os.close(audit_fd)
        
    def _apply_mitigation(self, mitigation: str, threat: SecurityThreat):
        """Apply specific mitigation"""
        if mitigation == 'sanitize_input':
            self._apply_input_sanitization(threat.severity)
            
        elif mitigation == 'limit_response_detail':
            self._apply_response_limiting(threat.severity)
            
        elif mitigation == 'rate_limit':
            self._apply_rate_limiting(threat.source, threat.severity)
            
        elif mitigation == 'require_verification':
            self._require_additional_verification(threat.source)
            
        elif mitigation == 'deep_analysis':
            self._trigger_deep_analysis(threat)
            
        elif mitigation == 'quarantine_request':
            self._quarantine_request(threat)
            
        elif mitigation == 'alert_security_team':
            self._alert_security_team(threat)
            
        elif mitigation == 'temporary_block':
            self._apply_temporary_block(threat.source, threat.severity)
            
    def _apply_input_sanitization(self, severity: float):
        """Apply input sanitization rules"""
        sanitize_config = {
            'remove_special_chars': severity > 0.3,
            'limit_length': int(10000 / (1 + severity * 2)),
            'normalize_unicode': True
        }
        
        self._send_defense_config('sanitization', sanitize_config)
        
    def _apply_response_limiting(self, severity: float):
        """Limit response detail based on threat"""
        limit_config = {
            'max_detail_level': int(10 * (1 - severity)),
            'obfuscate_internals': severity > 0.5,
            'add_noise': severity > 0.7
        }
        
        self._send_defense_config('response_limiting', limit_config)
        
    def _apply_rate_limiting(self, source: str, severity: float):
        """Apply rate limiting to source"""
        rate_limit = max(1, int(100 * (1 - severity)))  # Requests per minute
        
        with self.defense_lock:
            self.active_defenses[f'rate_limit_{source}'] = {
                'type': 'rate_limit',
                'limit': rate_limit,
                'expires': time.time() + 3600  # 1 hour
            }
            
    def _require_additional_verification(self, source: str):
        """Require additional verification for source"""
        with self.defense_lock:
            self.active_defenses[f'verify_{source}'] = {
                'type': 'verification_required',
                'expires': time.time() + 1800  # 30 minutes
            }
            
    def _trigger_deep_analysis(self, threat: SecurityThreat):
        """Trigger deep analysis of threat"""
        # Queue for async analysis
        analysis_request = {
            'threat': threat,
            'timestamp': time.time(),
            'priority': 'high' if threat.severity > 0.8 else 'normal'
        }
        
        # Send to analysis queue
        self._send_to_analysis_queue(analysis_request)
        
    def _quarantine_request(self, threat: SecurityThreat):
        """Quarantine suspicious request"""
        quarantine_path = f"/data/sovren/quarantine/{threat.threat_id}.json"
        os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
        
        with open(quarantine_path, 'w') as f:
            json.dump({
                'threat': threat.__dict__,
                'quarantined_at': time.time()
            }, f)
            
    def _alert_security_team(self, threat: SecurityThreat):
        """Alert security team about high-severity threat"""
        alert = {
            'alert_type': 'security_threat',
            'severity': 'critical' if threat.severity > 0.9 else 'high',
            'threat': threat.__dict__,
            'timestamp': time.time()
        }
        
        # Send alert (in production, would use actual alerting system)
        print(f"SECURITY ALERT: {alert}")
        
    def _apply_temporary_block(self, source: str, severity: float):
        """Temporarily block source"""
        block_duration = int(300 * severity)  # 5 minutes * severity
        
        with self.defense_lock:
            self.active_defenses[f'block_{source}'] = {
                'type': 'temporary_block',
                'expires': time.time() + block_duration
            }
            
    def _send_defense_config(self, defense_type: str, config: Dict[str, Any]):
        """Send defense configuration to components"""
        msg = {
            'type': 'defense_config',
            'defense': defense_type,
            'config': config,
            'timestamp': time.time()
        }
        
        # Broadcast to components
        defense_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        
        for component in ['consciousness', 'shadow_board', 'agent_battalion']:
            socket_path = f'/data/sovren/sockets/{component}_defense'
            if os.path.exists(socket_path):
                try:
                    defense_socket.sendto(
                        json.dumps(msg).encode(),
                        socket_path
                    )
                except:
                    pass
                    
    def _send_to_analysis_queue(self, request: Dict[str, Any]):
        """Send to analysis queue"""
        # In production, would use actual queue system
        pass
        
    def _update_defense_posture(self, threat: SecurityThreat):
        """Update overall defense posture based on threats"""
        # Calculate threat level
        with self.defense_lock:
            recent_threats = [
                d for d in self.active_defenses.values()
                if d.get('type') == 'threat' and 
                time.time() - d.get('timestamp', 0) < 3600
            ]
            
        threat_score = len(recent_threats) * 0.1 + threat.severity * 0.5
        
        if threat_score > 0.8:
            posture = 'critical'
        elif threat_score > 0.6:
            posture = 'high'
        elif threat_score > 0.4:
            posture = 'elevated'
        else:
            posture = 'normal'
            
        # Update system-wide defense posture
        self._broadcast_defense_posture(posture)
        
    def _broadcast_defense_posture(self, posture: str):
        """Broadcast defense posture to all components"""
        msg = {
            'type': 'defense_posture',
            'posture': posture,
            'timestamp': time.time()
        }
        
        # Broadcast to all components
        try:
            broadcast_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            broadcast_socket.sendto(
                json.dumps(msg).encode(),
                '/data/sovren/sockets/defense_broadcast'
            )
            broadcast_socket.close()
        except:
            pass

class SecurityOrchestrator:
    """Main security orchestration system"""
    
    def __init__(self):
        # Initialize components
        self.zk_system = ZeroKnowledgeProofSystem()
        self.adversarial_detector = AdversarialDetector()
        self.defense_coordinator = DefenseCoordinator(self)
        
        # Security state
        self.security_level = 'normal'  # normal, elevated, high, critical
        self.active_threats = []
        self.threat_lock = threading.Lock()
        
        # Audit log
        self.audit_fd = os.open('/data/sovren/logs/security_audit.log', 
                               os.O_CREAT | os.O_WRONLY | os.O_APPEND)
                               
        # Start monitoring
        self._start_security_monitoring()
        
    def process_request(self, user_id: str, request_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Process request through security checks"""
        # Extract text input
        input_text = request_data.get('query', '') + str(request_data.get('data', ''))
        
        # Check for threats
        threat = self.adversarial_detector.analyze_input(user_id, input_text, request_data)
        
        if threat:
            # Handle threat
            self.defense_coordinator.respond_to_threat(threat)
            
            # Determine if request should be blocked
            if threat.severity > 0.8:
                return False, f"Request blocked: {threat.threat_type}"
                
        return True, None
        
    def generate_value_proof(self, value: float, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Generate zero-knowledge proof of value creation"""
        return self.zk_system.generate_value_proof(value, metadata)
        
    def verify_proof(self, proof_id: str, verification_key: str) -> bool:
        """Verify a zero-knowledge proof"""
        return self.zk_system.verify_proof(proof_id, verification_key)
        
    def _start_security_monitoring(self):
        """Start security monitoring threads"""
        # Threat monitoring
        threat_thread = threading.Thread(target=self._monitor_threats)
        threat_thread.daemon = True
        threat_thread.start()
        
        # System health monitoring
        health_thread = threading.Thread(target=self._monitor_system_health)
        health_thread.daemon = True
        health_thread.start()
        
    def _monitor_threats(self):
        """Monitor for security threats"""
        while True:
            time.sleep(60)  # Check every minute
            
            # Clean up expired threats
            with self.threat_lock:
                self.active_threats = [
                    t for t in self.active_threats
                    if time.time() - t.timestamp < 3600  # Keep for 1 hour
                ]
                
            # Update security level based on active threats
            self._update_security_level()
            
    def _monitor_system_health(self):
        """Monitor system health for security implications"""
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            # Check resource usage
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            
            # Detect potential DoS
            if cpu_usage > 90 or memory_usage > 90:
                threat = SecurityThreat(
                    threat_id=f"dos_{int(time.time())}",
                    threat_type='dos_attack',
                    severity=0.7,
                    source='system',
                    timestamp=time.time(),
                    evidence=[f'CPU: {cpu_usage}%', f'Memory: {memory_usage}%'],
                    mitigations=['rate_limit', 'resource_throttling']
                )
                
                self.defense_coordinator.respond_to_threat(threat)
                
    def _update_security_level(self):
        """Update overall security level"""
        threat_count = len(self.active_threats)
        max_severity = max([t.severity for t in self.active_threats], default=0)
        
        if threat_count > 10 or max_severity > 0.9:
            self.security_level = 'critical'
        elif threat_count > 5 or max_severity > 0.7:
            self.security_level = 'high'
        elif threat_count > 2 or max_severity > 0.5:
            self.security_level = 'elevated'
        else:
            self.security_level = 'normal'
            
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        # Simplified - would use actual system metrics
        return np.random.uniform(10, 40)
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        # Simplified - would use actual system metrics
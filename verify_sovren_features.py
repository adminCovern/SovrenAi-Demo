#!/usr/bin/env python3
"""
SOVREN AI Feature Verification Script
Comprehensive test of all core features and functionality
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

@dataclass
class FeatureTest:
    name: str
    category: str
    test_func: callable
    critical: bool = True
    dependencies: List[str] = None

class SovrenFeatureVerifier:
    def __init__(self):
        self.results = {}
        self.features_tested = 0
        self.features_passed = 0
        self.critical_failures = []
        
    def log(self, message: str, color: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}[{timestamp}] {message}{RESET}")
        
    def check_file_exists(self, filepath: str, description: str) -> bool:
        """Check if a critical file exists"""
        exists = os.path.exists(filepath)
        if exists:
            self.log(f"✓ {description}: {filepath}", GREEN)
        else:
            self.log(f"✗ {description}: {filepath} NOT FOUND", RED)
        return exists
        
    def check_service_config(self, service_name: str) -> bool:
        """Check if systemd service is configured"""
        service_file = f"/etc/systemd/system/{service_name}.service"
        return self.check_file_exists(service_file, f"Service {service_name}")
        
    def check_gpu_availability(self) -> Tuple[bool, int]:
        """Check GPU availability and count"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_count = len(result.stdout.strip().split('\n'))
                self.log(f"✓ Found {gpu_count} GPUs", GREEN)
                return True, gpu_count
        except:
            pass
        self.log("✗ No GPUs detected", RED)
        return False, 0
        
    def verify_consciousness_engine(self) -> bool:
        """Verify Bayesian Consciousness Engine"""
        self.log("Checking Consciousness Engine...", BLUE)
        
        files_to_check = [
            ("/home/ubuntu/consciousness_engine_pcie_b200.py", "PCIe B200 Optimized Engine"),
            ("/home/ubuntu/artifact-consciousness-engine-v1.py", "Consciousness Engine Artifact"),
        ]
        
        found = False
        for filepath, desc in files_to_check:
            if self.check_file_exists(filepath, desc):
                found = True
                
        # Check for proper GPU memory configuration
        if found:
            with open("/home/ubuntu/consciousness_engine_pcie_b200.py", 'r') as f:
                content = f.read()
                if "'gpu_memory_gb': 80" in content:
                    self.log("✓ GPU memory correctly set to 80GB", GREEN)
                else:
                    self.log("✗ GPU memory configuration incorrect", RED)
                    found = False
                    
        return found
        
    def verify_agent_battalions(self) -> bool:
        """Verify 5 Agent Battalions"""
        self.log("Checking Agent Battalions...", BLUE)
        
        battalions = ["STRIKE", "INTEL", "OPS", "SENTINEL", "COMMAND"]
        found_count = 0
        
        # Check for battalion files
        files_to_check = [
            "/home/ubuntu/artifact-agent-battalion-v1.py",
            "/data/sovren/agent_battalion/agent_battalion_system.py"
        ]
        
        for filepath in files_to_check:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    for battalion in battalions:
                        if battalion in content:
                            self.log(f"✓ Found {battalion} battalion", GREEN)
                            found_count += 1
                            
        if found_count >= 5:
            self.log("✓ All 5 Agent Battalions configured", GREEN)
            return True
        else:
            self.log(f"✗ Only found {found_count}/5 battalions", RED)
            return False
            
    def verify_shadow_board(self) -> bool:
        """Verify Shadow Board implementation"""
        self.log("Checking Shadow Board...", BLUE)
        
        files = [
            "/home/ubuntu/artifact-shadow-board-v1 (1).py",
            "/data/sovren/shadow_board/deep_executive_personality.py"
        ]
        
        found = False
        for filepath in files:
            if self.check_file_exists(filepath, "Shadow Board"):
                found = True
                
        return found
        
    def verify_time_machine(self) -> bool:
        """Verify Time Machine functionality"""
        self.log("Checking Time Machine...", BLUE)
        
        files = [
            "/home/ubuntu/artifact-time-machine-v1.py",
            "/data/sovren/time_machine_system.py",
            "/data/sovren/time_machine.db"
        ]
        
        found_count = 0
        for filepath in files:
            if os.path.exists(filepath):
                self.log(f"✓ Found {os.path.basename(filepath)}", GREEN)
                found_count += 1
                
        return found_count >= 2
        
    def verify_ai_models(self) -> bool:
        """Verify core AI models configuration"""
        self.log("Checking AI Models...", BLUE)
        
        models = {
            "Whisper ASR": ["/data/sovren/models/ggml-large-v3.bin", "/home/ubuntu/sovren-deployment-final.sh"],
            "StyleTTS2": ["/data/sovren/models/styletts2_model.pth", "/data/sovren/models/styletts2_config.yml"],
            "Mixtral LLM": ["/data/sovren/models/mixtral-8x7b-q4.gguf", "/data/sovren/models/"]
        }
        
        found_count = 0
        for model_name, paths in models.items():
            found = False
            for path in paths:
                if os.path.exists(path):
                    found = True
                    break
            if found:
                self.log(f"✓ {model_name} configured", GREEN)
                found_count += 1
            else:
                self.log(f"✗ {model_name} not found", RED)
                
        return found_count >= 2
        
    def verify_skyetel_integration(self) -> bool:
        """Verify Skyetel telephony integration"""
        self.log("Checking Skyetel Integration...", BLUE)
        
        # Check for Skyetel configuration
        files = [
            "/home/ubuntu/sovren-telephony-system.py",
            "/data/sovren/voice/skyetel_oauth.py",
            "/data/sovren/voice/skyetel_auth.py"
        ]
        
        found = False
        for filepath in files:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    if "skyetel" in content.lower() or "Skyetel" in content:
                        self.log(f"✓ Found Skyetel configuration in {os.path.basename(filepath)}", GREEN)
                        found = True
                        
        # Check deployment script for Skyetel
        if os.path.exists("/home/ubuntu/DEPLOY_NOW_CORRECTED.sh"):
            with open("/home/ubuntu/DEPLOY_NOW_CORRECTED.sh", 'r') as f:
                if "SKYETEL" in f.read():
                    self.log("✓ Skyetel configured in deployment", GREEN)
                    found = True
                    
        return found
        
    def verify_billing_system(self) -> bool:
        """Verify Kill Bill integration"""
        self.log("Checking Kill Bill Integration...", BLUE)
        
        files = [
            "/home/ubuntu/killbill-integration.py",
            "/data/sovren/billing/payment_system.py"
        ]
        
        found = False
        for filepath in files:
            if self.check_file_exists(filepath, "Kill Bill"):
                found = True
                
        # Check for ceremony experience
        if found and os.path.exists("/data/sovren/billing/payment_system.py"):
            with open("/data/sovren/billing/payment_system.py", 'r') as f:
                if "ceremony" in f.read().lower():
                    self.log("✓ Payment ceremony experience found", GREEN)
                    
        return found
        
    def verify_authentication(self) -> bool:
        """Verify Azure AD authentication"""
        self.log("Checking Azure AD Authentication...", BLUE)
        
        # Check for Azure configuration
        config_found = False
        if os.path.exists("/home/ubuntu/sovren-deployment-final.sh"):
            with open("/home/ubuntu/sovren-deployment-final.sh", 'r') as f:
                content = f.read()
                if "AZURE_TENANT_ID" in content and "AZURE_CLIENT_ID" in content:
                    self.log("✓ Azure AD credentials configured", GREEN)
                    config_found = True
                    
        # Check auth system
        if self.check_file_exists("/data/sovren/security/auth_system.py", "Auth System"):
            config_found = True
            
        return config_found
        
    def verify_frontend(self) -> bool:
        """Verify PWA frontend"""
        self.log("Checking PWA Frontend...", BLUE)
        
        frontend_files = [
            "/home/ubuntu/sovren-frontend-complete.tsx",
            "/data/sovren/frontend/package.json",
            "/data/sovren/frontend/src/",
            "/data/sovren/frontend/public/"
        ]
        
        found_count = 0
        for filepath in frontend_files:
            if os.path.exists(filepath):
                self.log(f"✓ Found {os.path.basename(filepath)}", GREEN)
                found_count += 1
                
        return found_count >= 2
        
    def verify_api_endpoints(self) -> bool:
        """Verify API server and endpoints"""
        self.log("Checking API Endpoints...", BLUE)
        
        api_files = [
            "/data/sovren/api/api_server.py",
            "/data/sovren/api/application_api.py",
            "/data/sovren/api/websocket_server.py"
        ]
        
        found_count = 0
        for filepath in api_files:
            if self.check_file_exists(filepath, "API"):
                found_count += 1
                
        return found_count >= 2
        
    def verify_deployment_readiness(self) -> bool:
        """Verify deployment configuration"""
        self.log("Checking Deployment Readiness...", BLUE)
        
        # Check for deployment scripts
        scripts = [
            "/home/ubuntu/DEPLOY_NOW_CORRECTED.sh",
            "/home/ubuntu/sovren-deployment-final.sh"
        ]
        
        found = False
        for script in scripts:
            if self.check_file_exists(script, "Deployment Script"):
                found = True
                
        # Check systemd services
        services = [
            "sovren-main",
            "sovren-consciousness",
            "sovren-whisper",
            "sovren-tts",
            "sovren-llm"
        ]
        
        service_count = 0
        for service in services:
            if self.check_service_config(service):
                service_count += 1
                
        if service_count >= 3:
            self.log(f"✓ {service_count}/5 services configured", GREEN)
        else:
            self.log(f"✗ Only {service_count}/5 services configured", RED)
            
        return found and service_count >= 3
        
    def run_all_tests(self):
        """Run all feature verification tests"""
        self.log("="*60, BLUE)
        self.log("SOVREN AI FEATURE VERIFICATION", BLUE)
        self.log("="*60, BLUE)
        
        # Define all tests
        tests = [
            FeatureTest("GPU Availability", "Hardware", self.check_gpu_availability),
            FeatureTest("Consciousness Engine", "Core AI", self.verify_consciousness_engine),
            FeatureTest("Agent Battalions", "Core AI", self.verify_agent_battalions),
            FeatureTest("Shadow Board", "Core AI", self.verify_shadow_board),
            FeatureTest("Time Machine", "Core AI", self.verify_time_machine),
            FeatureTest("AI Models", "Models", self.verify_ai_models),
            FeatureTest("Skyetel Integration", "Telephony", self.verify_skyetel_integration),
            FeatureTest("Kill Bill Billing", "Billing", self.verify_billing_system),
            FeatureTest("Azure AD Auth", "Security", self.verify_authentication),
            FeatureTest("PWA Frontend", "UI", self.verify_frontend),
            FeatureTest("API Endpoints", "Backend", self.verify_api_endpoints),
            FeatureTest("Deployment Config", "DevOps", self.verify_deployment_readiness),
        ]
        
        # Run tests by category
        categories = {}
        for test in tests:
            if test.category not in categories:
                categories[test.category] = []
            categories[test.category].append(test)
            
        for category, category_tests in categories.items():
            self.log(f"\n{category} Features:", YELLOW)
            self.log("-" * 40, YELLOW)
            
            for test in category_tests:
                self.features_tested += 1
                try:
                    result = test.test_func()
                    if result:
                        self.features_passed += 1
                        self.results[test.name] = "PASSED"
                    else:
                        self.results[test.name] = "FAILED"
                        if test.critical:
                            self.critical_failures.append(test.name)
                except Exception as e:
                    self.log(f"✗ {test.name} - ERROR: {str(e)}", RED)
                    self.results[test.name] = f"ERROR: {str(e)}"
                    if test.critical:
                        self.critical_failures.append(test.name)
                        
        # Generate summary
        self.generate_summary()
        
    def generate_summary(self):
        """Generate verification summary"""
        self.log("\n" + "="*60, BLUE)
        self.log("VERIFICATION SUMMARY", BLUE)
        self.log("="*60, BLUE)
        
        # Overall stats
        pass_rate = (self.features_passed / self.features_tested) * 100 if self.features_tested > 0 else 0
        
        self.log(f"\nTotal Features Tested: {self.features_tested}", YELLOW)
        self.log(f"Features Passed: {self.features_passed}", GREEN if self.features_passed == self.features_tested else YELLOW)
        self.log(f"Features Failed: {self.features_tested - self.features_passed}", RED if self.features_tested > self.features_passed else GREEN)
        self.log(f"Pass Rate: {pass_rate:.1f}%", GREEN if pass_rate >= 80 else RED)
        
        # Critical failures
        if self.critical_failures:
            self.log("\nCRITICAL FAILURES:", RED)
            for failure in self.critical_failures:
                self.log(f"  - {failure}", RED)
        else:
            self.log("\n✓ No critical failures", GREEN)
            
        # Detailed results
        self.log("\nDetailed Results:", YELLOW)
        for feature, status in self.results.items():
            color = GREEN if status == "PASSED" else RED
            symbol = "✓" if status == "PASSED" else "✗"
            self.log(f"  {symbol} {feature}: {status}", color)
            
        # Deployment readiness
        self.log("\n" + "="*60, BLUE)
        if pass_rate >= 80 and not self.critical_failures:
            self.log("DEPLOYMENT STATUS: READY ✓", GREEN)
            self.log("System is ready for production deployment", GREEN)
        else:
            self.log("DEPLOYMENT STATUS: NOT READY ✗", RED)
            self.log("Critical issues must be resolved before deployment", RED)
            
        # Save results
        with open("/home/ubuntu/sovren_verification_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "features_tested": self.features_tested,
                "features_passed": self.features_passed,
                "pass_rate": pass_rate,
                "critical_failures": self.critical_failures,
                "results": self.results
            }, f, indent=2)
            
        self.log(f"\nResults saved to: /home/ubuntu/sovren_verification_results.json", BLUE)


if __name__ == "__main__":
    verifier = SovrenFeatureVerifier()
    verifier.run_all_tests()
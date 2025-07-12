#!/usr/bin/env python3
"""
SOVREN AI Deployment Test Suite
Comprehensive testing for 24-hour deployment validation
"""

import os
import sys
import time
import json
import asyncio
import subprocess
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeploymentTests')

# Test Configuration
TEST_CONFIG = {
    'api_url': 'https://sovrenai.app/api',
    'timeout': 30,
    'performance_thresholds': {
        'decision_latency_ms': 100,
        'voice_latency_ms': 200,
        'api_response_ms': 500,
        'concurrent_users': 500,
        'simulations_per_second': 100000
    },
    'required_services': [
        'nginx',
        'postgresql'
    ]
}

class DeploymentTestSuite:
    """Complete deployment test suite"""
    
    def __init__(self):
        self.results = {
            'infrastructure': {},
            'services': {},
            'performance': {},
            'integration': {},
            'security': {},
            'timestamp': datetime.now().isoformat()
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("Starting SOVREN AI Deployment Tests...")
        
        # Phase 1: Infrastructure Tests
        self.test_infrastructure()
        
        # Phase 2: Service Tests
        self.test_services()
        
        # Phase 3: Performance Tests
        self.test_performance()
        
        # Phase 4: Integration Tests
        self.test_integration()
        
        # Phase 5: Security Tests
        self.test_security()
        
        # Generate report
        self.generate_report()
        
        return self.results
        
    def test_infrastructure(self):
        """Test infrastructure requirements"""
        logger.info("Testing infrastructure...")
        
        tests = {
            'gpu_check': self._check_gpus(),
            'memory_check': self._check_memory(),
            'cpu_check': self._check_cpu(),
            'storage_check': self._check_storage(),
            'network_check': self._check_network()
        }
        
        for test_name, result in tests.items():
            self.results['infrastructure'][test_name] = result
            
    def _check_gpus(self) -> Dict[str, Any]:
        """Check GPU availability"""
        try:
            # Check nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        name, memory = parts[0], parts[1]
                        gpus.append({
                            'name': name,
                            'memory': memory
                        })
                    
            # Verify B200 GPUs
            b200_count = sum(1 for gpu in gpus if 'B200' in gpu['name'])
            
            return {
                'status': 'pass' if len(gpus) >= 1 else 'fail',
                'gpu_count': len(gpus),
                'b200_count': b200_count,
                'gpus': gpus
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                
            total_gb = 0
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                    total_gb = total_kb / (1024 * 1024)
                    break
                    
            return {
                'status': 'pass' if total_gb >= 16 else 'fail',  # 16GB minimum
                'total_gb': round(total_gb, 2),
                'required_gb': 16
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU cores"""
        try:
            cpu_count = os.cpu_count()
            
            return {
                'status': 'pass' if cpu_count >= 4 else 'fail',
                'cores': cpu_count,
                'required_cores': 4
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _check_storage(self) -> Dict[str, Any]:
        """Check storage space"""
        try:
            stat = os.statvfs('/data/sovren')
            
            # Calculate available space
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            return {
                'status': 'pass' if total_gb >= 50 else 'fail',  # 50GB minimum
                'total_gb': round(total_gb, 2),
                'available_gb': round(available_gb, 2),
                'required_gb': 50
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Test external connectivity
            result = subprocess.run(['ping', '-c', '1', 'google.com'], 
                                 capture_output=True, text=True)
            external_connectivity = result.returncode == 0
            
            # Test local services
            api_test = subprocess.run(['curl', '-s', '-I', 'http://localhost:8000'], 
                                    capture_output=True, text=True)
            api_connectivity = 'HTTP' in api_test.stdout
            
            frontend_test = subprocess.run(['curl', '-s', '-I', 'http://localhost:3000'], 
                                         capture_output=True, text=True)
            frontend_connectivity = 'HTTP' in frontend_test.stdout
            
            return {
                'status': 'pass' if external_connectivity else 'fail',
                'external_connectivity': external_connectivity,
                'api_connectivity': api_connectivity,
                'frontend_connectivity': frontend_connectivity
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def test_services(self):
        """Test all SOVREN services"""
        logger.info("Testing services...")
        
        for service in TEST_CONFIG['required_services']:
            self.results['services'][service] = self._check_service(service)
            
    def _check_service(self, service_name: str) -> Dict[str, Any]:
        """Check individual service status"""
        try:
            # Check systemd service
            result = subprocess.run(
                ['systemctl', 'is-active', service_name],
                capture_output=True,
                text=True
            )
            
            is_active = result.stdout.strip() == 'active'
            
            return {
                'status': 'pass' if is_active else 'fail',
                'active': is_active,
                'service': service_name
            }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def test_performance(self):
        """Test performance metrics"""
        logger.info("Testing performance...")
        
        # File system performance
        self.results['performance']['filesystem'] = self._test_filesystem()
        
        # Python import performance
        self.results['performance']['python_imports'] = self._test_python_imports()
        
    def _test_filesystem(self) -> Dict[str, Any]:
        """Test filesystem performance"""
        try:
            test_file = '/tmp/sovren_test_file'
            start = time.time()
            
            # Write test
            with open(test_file, 'w') as f:
                f.write('x' * 1000000)  # 1MB
            
            write_time = time.time() - start
            
            # Read test
            start = time.time()
            with open(test_file, 'r') as f:
                data = f.read()
            
            read_time = time.time() - start
            
            # Cleanup
            os.unlink(test_file)
            
            return {
                'status': 'pass',
                'write_time_ms': round(write_time * 1000, 2),
                'read_time_ms': round(read_time * 1000, 2)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _test_python_imports(self) -> Dict[str, Any]:
        """Test Python import performance"""
        try:
            start = time.time()
            
            # Test basic imports
            import torch
            import numpy as np
            import asyncio
            
            import_time = time.time() - start
            
            return {
                'status': 'pass',
                'import_time_ms': round(import_time * 1000, 2),
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def test_integration(self):
        """Test system integration"""
        logger.info("Testing integration...")
        
        # Test SOVREN file structure
        self.results['integration']['file_structure'] = self._test_file_structure()
        
        # Test Python module imports
        self.results['integration']['module_imports'] = self._test_module_imports()
        
    def _test_file_structure(self) -> Dict[str, Any]:
        """Test SOVREN file structure"""
        required_paths = [
            '/data/sovren',
            '/data/sovren/api',
            '/data/sovren/frontend',
            '/data/sovren/consciousness',
            '/data/sovren/agent_battalion',
            '/data/sovren/shadow_board',
            '/data/sovren/time_machine'
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
                
        return {
            'status': 'pass' if not missing_paths else 'fail',
            'required_paths': len(required_paths),
            'found_paths': len(required_paths) - len(missing_paths),
            'missing_paths': missing_paths
        }
        
    def _test_module_imports(self) -> Dict[str, Any]:
        """Test SOVREN module imports"""
        try:
            sys.path.insert(0, '/data/sovren')
            
            import_results = {}
            
            # Test consciousness engine
            try:
                from consciousness.consciousness_engine import PCIeB200ConsciousnessEngine
                import_results['consciousness'] = 'pass'
            except Exception as e:
                import_results['consciousness'] = f'fail: {e}'
                
            # Test shadow board
            try:
                from shadow_board.shadow_board import ShadowBoard
                import_results['shadow_board'] = 'pass'
            except Exception as e:
                import_results['shadow_board'] = f'fail: {e}'
                
            # Test time machine
            try:
                from time_machine.time_machine_system import TimeMachine
                import_results['time_machine'] = 'pass'
            except Exception as e:
                import_results['time_machine'] = f'fail: {e}'
                
            # Test agent battalion
            try:
                from agent_battalion.agent_battalion_system import AgentBattalion
                import_results['agent_battalion'] = 'pass'
            except Exception as e:
                import_results['agent_battalion'] = f'fail: {e}'
                
            passed = sum(1 for result in import_results.values() if result == 'pass')
            total = len(import_results)
            
            return {
                'status': 'pass' if passed == total else 'fail',
                'passed_imports': passed,
                'total_imports': total,
                'import_results': import_results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def test_security(self):
        """Test security features"""
        logger.info("Testing security...")
        
        # Test SSL certificate
        self.results['security']['ssl_certificate'] = self._test_ssl_certificate()
        
        # Test nginx configuration
        self.results['security']['nginx_config'] = self._test_nginx_config()
        
    def _test_ssl_certificate(self) -> Dict[str, Any]:
        """Test SSL certificate"""
        try:
            result = subprocess.run([
                'openssl', 's_client', '-connect', 'sovrenai.app:443', '-servername', 'sovrenai.app'
            ], input='', capture_output=True, text=True, timeout=10)
            
            cert_valid = 'Verify return code: 0 (ok)' in result.stdout
            
            return {
                'status': 'pass' if cert_valid else 'fail',
                'certificate_valid': cert_valid
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _test_nginx_config(self) -> Dict[str, Any]:
        """Test nginx configuration"""
        try:
            result = subprocess.run(['nginx', '-t'], capture_output=True, text=True)
            config_valid = result.returncode == 0
            
            return {
                'status': 'pass' if config_valid else 'fail',
                'config_valid': config_valid,
                'output': result.stderr
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def generate_report(self):
        """Generate test report"""
        logger.info("Generating test report...")
        
        # Calculate summary
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        for category, tests in self.results.items():
            if isinstance(tests, dict) and category != 'timestamp':
                for test_name, result in tests.items():
                    if isinstance(result, dict) and 'status' in result:
                        total_tests += 1
                        if result['status'] == 'pass':
                            passed_tests += 1
                        elif result['status'] == 'fail':
                            failed_tests += 1
                        elif result['status'] == 'error':
                            error_tests += 1
                            
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': round(passed_tests / total_tests * 100, 2) if total_tests > 0 else 0
        }
        
        # Save report
        report_path = f'/tmp/sovren_deployment_test_{int(time.time())}.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Test report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SOVREN AI DEPLOYMENT TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({self.results['summary']['success_rate']}%)")
        print(f"Failed: {failed_tests}")
        print(f"Errors: {error_tests}")
        print("="*60)
        
        if failed_tests > 0 or error_tests > 0:
            print("\nFAILED TESTS:")
            for category, tests in self.results.items():
                if isinstance(tests, dict) and category not in ['timestamp', 'summary']:
                    for test_name, result in tests.items():
                        if isinstance(result, dict) and result.get('status') in ['fail', 'error']:
                            print(f"  - {category}.{test_name}: {result.get('error', 'Failed')}")
                            
        print(f"\nFull report: {report_path}")
        print("\n")

def main():
    """Run deployment tests"""
    tester = DeploymentTestSuite()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
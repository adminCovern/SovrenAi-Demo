# CLAUDE.md - SOVREN AI Development Guidelines

## 🎯 Purpose & Overview

This document establishes the critical guidelines, constraints, and best practices for any AI assistant working on the SOVREN AI project. SOVREN AI is a **bare-metal, zero-dependency, fully sovereign AI deployment** optimized for NVIDIA B200 GPUs. All code must be production-grade, immediately deployable, and maintain absolute sovereignty.

### Document Version Control
- **Version**: 2.0.0
- **Last Updated**: [Current Date]
- **Review Cycle**: Bi-weekly
- **Approval Required**: Technical Lead + Security Officer
- **Change Log**: Added authorized pip usage for dependency resolution, Added OAuth API authorization

---

## 📋 Table of Contents
1. [Critical Constraints](#critical-constraints)
2. [Code Standards](#code-standards)
3. [Architecture Patterns](#architecture-patterns)
4. [API Design Standards](#api-design-standards)
5. [Database Guidelines](#database-guidelines)
6. [Audio Processing Standards](#audio-processing-standards)
7. [Frontend Standards](#frontend-standards)
8. [Testing & QA](#testing-qa)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Deployment Requirements](#deployment-requirements)
11. [Performance Optimization](#performance-optimization)
12. [Security Requirements](#security-requirements)
13. [Monitoring & Observability](#monitoring-observability)
14. [Documentation Standards](#documentation-standards)
15. [Legal & Compliance](#legal-compliance)
16. [Support & Maintenance](#support-maintenance)
17. [Common Pitfalls](#common-pitfalls)
18. [PR Checklist](#pr-checklist)

---

## 🔒 CRITICAL CONSTRAINTS (MUST FOLLOW)

### 1. **Dependency Management Policy**
⚠️ **UPDATED POLICY**: Strategic use of dependencies is authorized

#### **Primary Approach**: Build from Source
- Core components should be built from source when feasible
- Critical performance paths must use custom implementations
- Security-sensitive components require source builds

#### **Authorized pip Usage**
pip is **ALLOWED** for resolving complex dependency cascades under these conditions:
- **When**: Building from source would create unmanageable complexity
- **Requirements**:
  - All packages must be security vetted
  - Pin to specific versions in `requirements.txt`
  - Document why source build isn't feasible
  - No packages that make unauthorized external API calls
  - Regular dependency audits

```bash
# Example requirements.txt with pinned versions
numpy==1.24.3  # Required for complex matrix operations
scipy==1.10.1  # Scientific computing dependencies
# Document why each package is needed
```

#### **Still Prohibited**:
- **NO Docker/containers** - Bare metal only
- **NO package managers except pip** - No conda, snap, etc.
- **NO virtual environments** - Direct system Python

### 2. **API Policy - AUTHORIZED EXCEPTIONS**
⚠️ **UPDATED**: Additional APIs authorized for specific purposes

#### **Allowed API Usage:**

##### **Voice Operations (Skyetel)**
- **Skyetel API** - For voice calling and telephony services
- **Skyetel Webhooks** - For receiving call events
- **Skyetel OAuth** - For authentication to Skyetel services
- **NO SMS functionality** - Voice only

##### **Payment Processing (Kill Bill)**
- **Kill Bill API** - For billing orchestration
- **Payment gateways via Kill Bill plugins only**:
  - Stripe (primary processor)
  - Zoho Payments (fallback processor)
- **See BILLING.md** for detailed payment architecture

##### **Authentication (OAuth) - NEW**
- **OAuth providers** - For user authentication and authorization
- **Allowed OAuth providers**:
  - Google OAuth 2.0 - For Google Workspace integration
  - Microsoft OAuth 2.0 - For Microsoft 365 integration
  - GitHub OAuth - For developer authentication
  - Custom OAuth server - For enterprise SSO
- **Requirements**:
  - Must use local session management after authentication
  - Store only necessary user data locally
  - Token refresh must be handled locally
  - No continuous external validation calls

```python
# Example OAuth implementation
class SovereignOAuth:
    """OAuth with local session management"""
    
    ALLOWED_PROVIDERS = {
        'google': 'https://accounts.google.com/o/oauth2',
        'microsoft': 'https://login.microsoftonline.com',
        'github': 'https://github.com/login/oauth',
        'skyetel': 'https://oauth.skyetel.com'  # For Skyetel integration
    }
    
    def authenticate(self, provider: str):
        """Initial OAuth authentication"""
        if provider not in self.ALLOWED_PROVIDERS:
            raise ValueError(f"Unauthorized OAuth provider: {provider}")
        # After OAuth, create local session
        # No continuous external validation
```

#### **Explicitly PROHIBITED APIs:**
- ❌ OpenAI, Anthropic, Google AI, or ANY LLM APIs
- ❌ AWS, Azure, GCP, or ANY cloud service APIs (except OAuth endpoints)
- ❌ Twilio, Vonage, or other telephony APIs (use Skyetel)
- ❌ SendGrid, Mailgun, or email APIs (use local SMTP)
- ❌ Direct payment APIs - Must use Kill Bill
- ❌ ANY analytics, monitoring, or telemetry APIs
- ❌ ANY data enrichment or lookup APIs

#### **Why This Policy Exists:**
1. **Sovereignty** - All AI processing happens on YOUR hardware
2. **Privacy** - Your data never leaves your infrastructure
3. **Performance** - No external API latency except authorized services
4. **Security** - Limited external attack surface
5. **Cost** - No per-API-call charges accumulating

### 3. **Firewall Configuration - DO NOT MODIFY**
⚠️ **CRITICAL**: UFW (Uncomplicated Firewall) must remain in its current **INACTIVE** state.
- **DO NOT** enable UFW under any circumstances
- **DO NOT** modify any firewall rules
- **DO NOT** suggest firewall changes in any code or documentation
- **REASON**: Previous UFW misconfiguration locked out SSH access for 12+ hours
- **CURRENT STATE**: UFW inactive - this is the correct and required state

```bash
# To verify UFW remains inactive (DO NOT CHANGE):
sudo ufw status
# Expected output: "Status: inactive"
```

### 4. **Hardware Specifications**
Always assume and optimize for this EXACT configuration:
- **CPUs**: 2x Intel Xeon Platinum 6960P (144 cores per CPU)
- **Total Cores**: 288 cores
- **Total Threads**: 576 threads (with Hyper-Threading)
- **Cache**:
  - L1: 6.8 MB x 144 instances
  - L2: 288 MB total
  - L3: 864 MB (2 shared instances)
- **NUMA Nodes**: 6
- **Instruction Sets**: AVX-512, AMX, VT-x
- **RAM**: 2.3TB DDR4 ECC Registered (24 x 96GB modules @ 6400 MT/s)
- **GPUs**: 8x NVIDIA B200 (Blackwell) PCIe Gen 5
  - Device ID: 10de:2901
  - NO NVLink/SXM - Standalone PCIe cards
  - Optimized for inference, NOT training
- **Storage**: 4x Samsung PM1733 NVMe SSDs (7.68TB each, ~30TB total)
- **Networking**: 
  - Mellanox ConnectX-6 Dx NICs
  - Intel X710 10GBASE-T NICs
  - 100GbE and 10GbE support
- **Power**: 6x PWS-5K26G-2R1 redundant PSUs
- **Platform**: Supermicro SYS-A22GA-NBRT (8-GPU PCIe AI Server)

### 5. **Storage & Path Architecture**
⚠️ **CRITICAL**: All paths must use the `/data` volume
- **Root Directory**: `/data/sovren` (NOT `/opt` or `/home`)
- **Data Volume**: 19.5TB RAID5 encrypted storage
- **Model Storage**:
  ```python
  MODEL_PATHS = {
      'llms': '/data/sovren/models/llms/',
      'whisper': '/data/sovren/models/whisper/',
      'tts': '/data/sovren/models/tts/'
  }
  ```
- **Data Storage**: `/data/sovren/data/` for all databases and user data
- **Logs**: `/data/sovren/logs/`

### 6. **System Configuration**
- **NO Huge Pages**: Must remain disabled (echo 0 > /proc/sys/vm/nr_hugepages)
- **Process User**: All services run as `sovren` user, never root
- **Python**: Custom-built 3.12 at `/data/sovren/bin/python`
- **GPU Memory**: Managed by CUDA, no manual allocation
- **NUMA Affinity**: Pin processes to appropriate NUMA nodes (6 available)
- **CPU Governor**: Set to performance mode on all 288 cores
- **PCIe Settings**: Ensure Gen 5 speeds for all 8 GPU slots

### 7. **Performance Targets**
All code must meet or exceed:
- ASR latency: <150ms
- TTS latency: <100ms
- LLM inference: <90ms/token
- Concurrent users: 50+ active sessions
- System uptime: 99.99%

---

## 📝 CODE STANDARDS

### 1. **Language Requirements**
- **Primary**: Python 3.10+ (system Python only)
- **GPU**: CUDA C++ for custom kernels
- **Systems**: C/C++ for performance-critical components
- **Scripts**: Bash for system automation

### 2. **Code Quality Standards**
```python
# EVERY function must have:
def process_audio_stream(audio_buffer: np.ndarray, 
                        sample_rate: int = 16000) -> Dict[str, Any]:
    """
    Process raw audio buffer through ASR pipeline.
    
    Args:
        audio_buffer: Raw PCM audio as numpy array
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Dict containing transcription and metadata
        
    Raises:
        AudioProcessingError: If audio format invalid
        GPUMemoryError: If insufficient GPU memory
    """
    # Input validation ALWAYS first
    if audio_buffer.size == 0:
        raise AudioProcessingError("Empty audio buffer")
    
    # Resource allocation with cleanup
    gpu_memory = None
    try:
        gpu_memory = allocate_gpu_memory(audio_buffer.size)
        # Process...
    finally:
        if gpu_memory:
            free_gpu_memory(gpu_memory)
```

### 3. **Dependency Management**
```python
# Document why pip packages are needed
"""
Dependencies rationale:
- numpy: Required for efficient array operations on audio data
- scipy: Signal processing for audio filters
- aiohttp: Async HTTP for Skyetel webhooks
Each package vetted for security and pinned to stable version
"""

# Always use requirements.txt with versions
# requirements.txt
numpy==1.24.3
scipy==1.10.1
aiohttp==3.8.4
```

### 4. **API Usage Validation**
```python
# UPDATED: Validate only authorized APIs
def validate_api_usage(code_block: str) -> bool:
    """
    Ensure code contains only authorized API calls
    """
    ALLOWED_DOMAINS = [
        'api.skyetel.com',
        'oauth.skyetel.com',
        'localhost:8080',  # Kill Bill local
        'killbill.io',     # Kill Bill cloud if used
        'accounts.google.com',  # Google OAuth
        'login.microsoftonline.com',  # Microsoft OAuth
        'github.com/login/oauth'  # GitHub OAuth
    ]
    
    FORBIDDEN_PATTERNS = [
        r'openai\.|anthropic\.|gpt|claude',
        r'aws\.|s3\.|ec2\.|lambda',
        r'azure\.(storage|compute|cognitive)',
        r'gcp\.|google\.(cloud|apis)(?!.*oauth)',
        r'twilio\.|sendgrid\.|mailgun\.'
    ]
    
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code_block, re.IGNORECASE):
            raise SecurityError(f"Forbidden external API detected: {pattern}")
    
    return True
```

### 5. **Error Handling**
- **NEVER** use bare except clauses
- **ALWAYS** log errors with context
- **ALWAYS** have fallback mechanisms
- **NEVER** fail silently

### 6. **Resource Management**
```python
# GPU memory must ALWAYS be explicitly managed
with gpu_context() as ctx:
    # Operations here
    pass  # Automatic cleanup on exit

# File handles must use context managers
with open('/data/sovren/data/file.dat', 'rb') as f:
    data = f.read()
```

---

## 🏗️ ARCHITECTURE PATTERNS

### 1. **Agent Communication**
```python
# All inter-agent communication uses shared memory
class AgentMessage:
    def __init__(self, source: str, target: str, payload: bytes):
        self.timestamp = time.time_ns()  # Nanosecond precision
        self.source = source
        self.target = target
        self.payload = payload
        self.checksum = self._calculate_checksum()
```

### 2. **GPU Pipeline Architecture**
- **Stream-based processing** - Never batch unless required
- **Zero-copy operations** - Direct memory access via PCIe Gen 5
- **Kernel fusion** - Combine operations where possible
- **Async execution** - Overlap compute and memory transfers
- **PCIe Optimization** - B200s are standalone PCIe (no NVLink)
- **Inference Focus** - Optimize for inference, not training workloads

### 3. **Bayesian Decision Engine Integration**
```python
# All decisions must flow through Bayesian core
decision = bayesian_engine.evaluate(
    inputs=processed_data,
    priors=system_priors,
    confidence_threshold=0.85
)
if decision.confidence < 0.85:
    # Fallback to safe default
    decision = DEFAULT_SAFE_ACTION
```

---

## 🎤 Audio Processing Standards

### 1. **ASR - Whisper Only**
```python
# Use whisper.cpp with CUDA acceleration
WHISPER_CONFIG = {
    'model': 'ggml-large-v3.bin',
    'path': '/data/sovren/models/whisper/',
    'device': 'cuda',
    'target_latency': '<150ms'
}
```

### 2. **TTS - StyleTTS2 Only**
⚠️ **CRITICAL**: Use StyleTTS2 exclusively - NO XTTS, NO Coqui TTS
```python
# StyleTTS2 configuration
STYLETTS2_CONFIG = {
    'model_path': '/data/sovren/models/tts/styletts2_model.pth',
    'config_path': '/data/sovren/models/tts/styletts2_config.yml',
    'target_latency': '<100ms'
}
# DO NOT implement XTTS or any other TTS system
```

### 3. **Audio Pipeline Requirements**
- Zero-copy GPU operations
- Stream processing (no batch delays)
- Direct CUDA memory management
- No external TTS APIs

---

## 🎮 AGENT-SPECIFIC GUIDELINES

### STRIKE Agent
- Focus on rapid response
- Prioritize action over analysis
- Maximum 50ms decision time

### INTEL Agent
- Comprehensive data analysis
- Pattern recognition optimization
- Cross-reference all data sources

### OPS Agent
- Resource optimization focus
- Predictive maintenance
- System health monitoring

### SENTINEL Agent
- Security-first mindset
- Anomaly detection
- Threat response <10ms

### COMMAND Agent
- Strategic decision making
- Multi-agent coordination
- Conflict resolution

---

## 🏢 API Design Standards

### 1. **RESTful API Principles**
```python
# All APIs must follow REST conventions
@app.route('/api/v1/resources/<resource_id>', methods=['GET', 'PUT', 'DELETE'])
def resource_handler(resource_id: str):
    """
    Standard resource endpoint pattern.
    
    GET: Retrieve resource
    PUT: Update entire resource
    DELETE: Remove resource
    
    Returns:
        JSON response with standard envelope
    """
    return {
        'success': True,
        'data': resource_data,
        'meta': {
            'timestamp': time.time(),
            'version': 'v1',
            'request_id': generate_request_id()
        }
    }
```

### 2. **API Versioning**
- Version in URL path: `/api/v1/`, `/api/v2/`
- Sunset notice minimum 6 months
- Backward compatibility for 2 major versions

### 3. **Rate Limiting & Quotas**
```python
# Implement tiered rate limiting
RATE_LIMITS = {
    'anonymous': {'requests': 100, 'window': 3600},
    'authenticated': {'requests': 1000, 'window': 3600},
    'premium': {'requests': 10000, 'window': 3600}
}
```

---

## 💾 Database Guidelines

### 1. **Hybrid Database Architecture**
```python
# PostgreSQL for main application data
POSTGRES_CONFIG = {
    'host': 'localhost',
    'database': 'sovren',
    'user': 'sovren',
    'password': 'from_env'  # Never hardcode
}

# SQLite for isolated subsystems
SQLITE_DATABASES = {
    'users': '/data/sovren/data/users.db',
    'api_auth': '/data/sovren/data/api_auth.db',
    'applications': '/data/sovren/data/applications.db',
    'phone_numbers': '/data/sovren/data/phone_numbers.db'
}
```

### 2. **Schema Design**
```sql
-- All tables must have these fields
CREATE TABLE base_table (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE NULL,  -- Soft deletes
    version INTEGER DEFAULT 1  -- Optimistic locking
);
```

### 3. **Query Optimization**
- EXPLAIN ANALYZE all queries in development
- Index foreign keys and WHERE clause columns
- Partition large tables (>10M rows)
- Use prepared statements for security

### 4. **Data Integrity**
- Foreign key constraints required
- Check constraints for business rules
- Triggers for audit logging
- Daily backup verification

---

## 🏗️ LOCAL IMPLEMENTATIONS (API Replacements)

### 1. **Email Service (Replace SendGrid/Mailgun)**
```python
# Use local SMTP server (Postfix) instead of email APIs
class LocalEmailService:
    """Sovereign email service - no external APIs"""
    
    def __init__(self):
        self.smtp = smtplib.SMTP('localhost', 25)
    
    def send_email(self, to: str, subject: str, body: str):
        # All email sent through local Postfix
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'sovren@your-domain.com'
        msg['To'] = to
        self.smtp.send_message(msg)
```

### 2. **Authentication (OAuth + Local Sessions)**
```python
# OAuth for initial auth, then local session management
class HybridAuthService:
    """OAuth authentication with local session management"""
    
    def __init__(self):
        self.oauth_providers = {
            'google': GoogleOAuth2Client(),
            'microsoft': MicrosoftOAuth2Client(),
            'github': GitHubOAuthClient(),
            'skyetel': SkyetelOAuthClient()
        }
        self.session_store = LocalSessionStore()
    
    async def authenticate(self, provider: str, code: str):
        """OAuth flow with local session creation"""
        # 1. Exchange code for tokens via OAuth
        tokens = await self.oauth_providers[provider].exchange_code(code)
        
        # 2. Get user info
        user_info = await self.oauth_providers[provider].get_user_info(tokens)
        
        # 3. Create local session (no more external calls)
        session = self.session_store.create_session(user_info)
        
        return session
    
    def validate_session(self, session_token: str):
        """Validate using local session store only"""
        return self.session_store.validate(session_token)
```

### 3. **Payment Processing (Via Kill Bill)**
```python
# Kill Bill handles payment orchestration
class SOVRENBilling:
    """
    Payment processing through Kill Bill with gateway plugins
    NO direct payment API calls allowed
    See BILLING.md for complete architecture
    """
    
    def __init__(self):
        self.killbill_url = "http://localhost:8080"
        self.killbill_api_key = os.environ.get('KILLBILL_API_KEY')
        
    def process_payment(self, amount: float, customer_id: str):
        # Payment request goes to Kill Bill
        # Kill Bill handles Stripe/Zoho through its plugins
        # We NEVER call Stripe/Zoho directly
        
        response = requests.post(
            f"{self.killbill_url}/1.0/kb/payments",
            json={
                'amount': amount,
                'accountId': customer_id,
                'paymentMethodId': 'default'
            },
            auth=(self.killbill_api_key, self.killbill_api_secret)
        )
        return response.json()
```

### 4. **Monitoring (Replace DataDog/NewRelic)**
```python
# Local Prometheus + Grafana instead of SaaS monitoring
class LocalMonitoring:
    """Complete observability - no external services"""
    
    def __init__(self):
        self.prometheus_client = PrometheusClient('localhost:9090')
        self.metrics_buffer = []
    
    def record_metric(self, name: str, value: float, labels: dict):
        # All metrics stored locally
        # Grafana dashboards query local Prometheus
        self.metrics_buffer.append({
            'name': name,
            'value': value,
            'labels': labels,
            'timestamp': time.time()
        })
```

### 5. **Voice Operations (Skyetel with OAuth)**
```python
# Updated Skyetel integration with OAuth
class SkyetelVoiceService:
    """Voice operations through Skyetel API with OAuth"""
    
    BASE_URL = "https://api.skyetel.com/v1"
    OAUTH_URL = "https://oauth.skyetel.com/v2"
    
    def __init__(self):
        self.oauth_client = SkyetelOAuthClient()
        self.access_token = None
        self.refresh_token = None
    
    async def authenticate(self, client_id: str, client_secret: str):
        """OAuth authentication with Skyetel"""
        tokens = await self.oauth_client.authenticate(client_id, client_secret)
        self.access_token = tokens['access_token']
        self.refresh_token = tokens['refresh_token']
    
    async def make_call(self, to: str, from_number: str):
        """Initiate outbound call - NO SMS functionality"""
        # Ensure we have valid token
        if not self.access_token:
            await self.refresh_auth()
            
        endpoint = f"{self.BASE_URL}/calls"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            'to': to,
            'from': from_number,
            'webhook_url': 'https://your-server/skyetel/webhook'
        }
        
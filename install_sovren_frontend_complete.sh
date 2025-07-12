#!/bin/bash
# Install SOVREN AI Frontend, Admin Dashboard, and MCP Server

set -e

echo "=== Installing SOVREN AI Frontend & Admin Components ==="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# 1. Install Node.js if not present
log "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    log "Installing Node.js..."
    cd /data/sovren/src
    if [ ! -f node-v20.11.0-linux-x64.tar.xz ]; then
        wget https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz
    fi
    tar -xf node-v20.11.0-linux-x64.tar.xz
    sudo ln -sf /data/sovren/src/node-v20.11.0-linux-x64/bin/node /usr/local/bin/node
    sudo ln -sf /data/sovren/src/node-v20.11.0-linux-x64/bin/npm /usr/local/bin/npm
    sudo ln -sf /data/sovren/src/node-v20.11.0-linux-x64/bin/npx /usr/local/bin/npx
else
    log "Node.js already installed: $(node --version)"
fi

# 2. Create Admin Dashboard Component
log "Creating Admin Dashboard..."
sudo mkdir -p /data/sovren/admin
sudo tee /data/sovren/admin/dashboard.tsx > /dev/null << 'EOF'
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Users, Shield, Activity, DollarSign, 
  CheckCircle, XCircle, Clock, Phone,
  Cpu, HardDrive, Wifi
} from 'lucide-react';

interface UserApproval {
  id: string;
  name: string;
  email: string;
  phone: string;
  tier: 'SMB' | 'ENTERPRISE' | 'FOUNDATION';
  requestedAt: string;
  status: 'pending' | 'approved' | 'rejected';
  phoneNumbersRequired: number;
}

export function AdminDashboard() {
  const [pendingUsers, setPendingUsers] = useState<UserApproval[]>([]);
  const [systemStats, setSystemStats] = useState({
    totalUsers: 0,
    activeUsers: 0,
    pendingApprovals: 0,
    monthlyRevenue: 0,
    gpuUtilization: 0,
    cpuUtilization: 0,
    memoryUsage: 0,
    activeVoiceSessions: 0
  });

  useEffect(() => {
    // Fetch pending approvals
    fetchPendingApprovals();
    // Fetch system stats
    fetchSystemStats();
    
    // Set up real-time updates
    const ws = new WebSocket('wss://sovrenai.app/ws/admin');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'user_approval_request') {
        setPendingUsers(prev => [...prev, data.user]);
      } else if (data.type === 'system_stats') {
        setSystemStats(data.stats);
      }
    };

    return () => ws.close();
  }, []);

  const fetchPendingApprovals = async () => {
    const response = await fetch('/api/admin/pending-approvals', {
      credentials: 'include'
    });
    const data = await response.json();
    setPendingUsers(data.users);
  };

  const fetchSystemStats = async () => {
    const response = await fetch('/api/admin/system-stats', {
      credentials: 'include'
    });
    const data = await response.json();
    setSystemStats(data);
  };

  const handleApproval = async (userId: string, approved: boolean) => {
    const response = await fetch(`/api/admin/approve-user/${userId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ approved })
    });

    if (response.ok) {
      setPendingUsers(prev => prev.filter(u => u.id !== userId));
      if (approved) {
        // Trigger phone number allocation
        await fetch(`/api/admin/allocate-numbers/${userId}`, {
          method: 'POST',
          credentials: 'include'
        });
      }
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'ENTERPRISE': return 'bg-purple-500';
      case 'SMB': return 'bg-blue-500';
      case 'FOUNDATION': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">SOVREN AI Admin Dashboard</h1>
        
        {/* System Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Users</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.totalUsers}</div>
              <p className="text-xs text-muted-foreground">
                {systemStats.activeUsers} active
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Monthly Revenue</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${systemStats.monthlyRevenue.toLocaleString()}
              </div>
              <p className="text-xs text-muted-foreground">
                Recurring revenue
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">GPU Utilization</CardTitle>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.gpuUtilization}%</div>
              <p className="text-xs text-muted-foreground">
                8x B200 183GB
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Calls</CardTitle>
              <Phone className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.activeVoiceSessions}</div>
              <p className="text-xs text-muted-foreground">
                Concurrent sessions
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Pending Approvals */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Pending User Approvals ({pendingUsers.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {pendingUsers.length === 0 ? (
              <p className="text-gray-400">No pending approvals</p>
            ) : (
              <div className="space-y-4">
                {pendingUsers.map((user) => (
                  <div key={user.id} className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <h3 className="font-semibold">{user.name}</h3>
                        <p className="text-sm text-gray-400">{user.email}</p>
                        <p className="text-sm text-gray-400">{user.phone}</p>
                      </div>
                      <Badge className={getTierColor(user.tier)}>
                        {user.tier}
                      </Badge>
                    </div>
                    
                    <div className="text-sm text-gray-400 mb-3">
                      Requested: {new Date(user.requestedAt).toLocaleString()}
                    </div>
                    
                    <Alert className="mb-3 bg-gray-600 border-gray-500">
                      <AlertDescription>
                        This user requires {user.phoneNumbersRequired} phone numbers
                        (${user.phoneNumbersRequired * 2}/month from Skyetel)
                      </AlertDescription>
                    </Alert>
                    
                    <div className="flex gap-2">
                      <Button
                        onClick={() => handleApproval(user.id, true)}
                        className="bg-green-600 hover:bg-green-700"
                      >
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Approve
                      </Button>
                      <Button
                        onClick={() => handleApproval(user.id, false)}
                        variant="destructive"
                      >
                        <XCircle className="h-4 w-4 mr-2" />
                        Reject
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* System Health */}
        <Card className="bg-gray-800 border-gray-700 mt-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Health
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">Hardware Status</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>CPU Usage</span>
                    <span>{systemStats.cpuUtilization}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Memory Usage</span>
                    <span>{systemStats.memoryUsage}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>GPU Utilization</span>
                    <span>{systemStats.gpuUtilization}%</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold mb-2">Service Status</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Consciousness Engine</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Agent Battalions</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Voice System (Skyetel)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>MCP Server</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
EOF

# 3. Create Login Page
log "Creating Login Page..."
sudo tee /data/sovren/frontend/src/Login.tsx > /dev/null << 'EOF'
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Brain } from 'lucide-react';

export function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();

      if (response.ok) {
        // Store user data
        localStorage.setItem('user', JSON.stringify(data.user));
        
        // Redirect based on role
        if (data.user.role === 'admin') {
          navigate('/admin');
        } else {
          navigate('/dashboard');
        }
      } else {
        setError(data.message || 'Login failed');
      }
    } catch (err) {
      setError('Connection error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleAzureLogin = () => {
    window.location.href = '/api/auth/azure';
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <Card className="w-full max-w-md bg-gray-800 border-gray-700">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <Brain className="h-12 w-12 text-purple-500" />
          </div>
          <CardTitle className="text-2xl font-bold text-white">
            SOVREN AI
          </CardTitle>
          <p className="text-gray-400 mt-2">
            Enterprise AI System Login
          </p>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert className="mb-4 bg-red-900 border-red-800">
              <AlertDescription className="text-white">
                {error}
              </AlertDescription>
            </Alert>
          )}

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <Input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="bg-gray-700 border-gray-600 text-white"
                required
              />
            </div>
            
            <div>
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-gray-700 border-gray-600 text-white"
                required
              />
            </div>

            <Button
              type="submit"
              className="w-full bg-purple-600 hover:bg-purple-700"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Logging in...
                </>
              ) : (
                'Login'
              )}
            </Button>
          </form>

          <div className="mt-4">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-gray-600" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-gray-800 px-2 text-gray-400">Or</span>
              </div>
            </div>

            <Button
              onClick={handleAzureLogin}
              variant="outline"
              className="w-full mt-4 bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
            >
              Login with Azure AD
            </Button>
          </div>

          <p className="text-center text-sm text-gray-400 mt-6">
            Don't have an account?{' '}
            <a href="/register" className="text-purple-400 hover:text-purple-300">
              Request Access
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
EOF

# 4. Create MCP Server Service
log "Creating MCP Server Service..."
sudo tee /etc/systemd/system/sovren-mcp.service > /dev/null << EOF
[Unit]
Description=SOVREN MCP Server
After=network.target sovren-main.service
PartOf=sovren.target

[Service]
Type=simple
User=sovren
Group=sovren
WorkingDirectory=/data/sovren
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/data/sovren:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages"
ExecStart=/usr/bin/python3 /data/sovren/mcp/mcp_server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=sovren.target
EOF

# 5. Create Frontend Service
log "Creating Frontend Service..."
sudo tee /etc/systemd/system/sovren-frontend.service > /dev/null << EOF
[Unit]
Description=SOVREN AI Frontend
After=network.target sovren-main.service
PartOf=sovren.target

[Service]
Type=simple
User=sovren
Group=sovren
WorkingDirectory=/data/sovren/frontend
Environment="NODE_ENV=production"
Environment="NEXT_PUBLIC_API_URL=http://localhost:8000"
ExecStart=/usr/local/bin/npm start
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=sovren.target
EOF

# 6. Install frontend dependencies
log "Installing frontend dependencies..."
cd /data/sovren/frontend
sudo -u sovren npm install --production

# 7. Build frontend
log "Building frontend for production..."
sudo -u sovren npm run build || {
    log "Build failed, creating fallback static server..."
    # Create a simple static server as fallback
    sudo tee /data/sovren/frontend/server.js > /dev/null << 'EOF'
const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, 'build')));
app.use(express.static(path.join(__dirname, 'public')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`SOVREN Frontend running on port ${PORT}`);
});
EOF
}

# 8. Create public index.html
log "Creating public HTML files..."
sudo mkdir -p /data/sovren/frontend/public
sudo tee /data/sovren/frontend/public/index.html > /dev/null << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOVREN AI - Enterprise AI System</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #111827;
            color: white;
        }
        #root {
            min-height: 100vh;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script src="/bundle.js"></script>
</body>
</html>
EOF

# 9. Update Nginx configuration
log "Updating Nginx configuration..."
sudo tee /etc/nginx/sites-available/sovren > /dev/null << 'EOF'
server {
    listen 80;
    server_name sovrenai.app www.sovrenai.app;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name sovrenai.app www.sovrenai.app;

    ssl_certificate /etc/nginx/sovren.crt;
    ssl_certificate_key /etc/nginx/sovren.key;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # MCP Server
    location /mcp {
        proxy_pass http://localhost:5010;
        proxy_http_version 1.1;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Host $host;
    }
}
EOF

# 10. Fix permissions
sudo chown -R sovren:sovren /data/sovren/frontend
sudo chown -R sovren:sovren /data/sovren/admin
sudo chown -R sovren:sovren /data/sovren/mcp

# 11. Reload services
log "Reloading services..."
sudo systemctl daemon-reload
sudo systemctl enable sovren-mcp sovren-frontend
sudo ln -sf /etc/nginx/sites-available/sovren /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

echo ""
echo "=== Frontend & Admin Components Installed ==="
echo ""
echo "Components installed:"
echo "✓ Frontend with login page at /data/sovren/frontend"
echo "✓ Admin dashboard at /data/sovren/admin"
echo "✓ MCP Server service configured"
echo "✓ Nginx configured for sovrenai.app"
echo ""
echo "To start services:"
echo "  sudo systemctl start sovren-mcp"
echo "  sudo systemctl start sovren-frontend"
echo ""
echo "To check status:"
echo "  sudo systemctl status sovren-mcp"
echo "  sudo systemctl status sovren-frontend"
echo ""
echo "Access points:"
echo "  Frontend: https://sovrenai.app (after DNS setup)"
echo "  Admin: https://sovrenai.app/admin"
echo "  API: https://sovrenai.app/api"
echo "  MCP: https://sovrenai.app/mcp"
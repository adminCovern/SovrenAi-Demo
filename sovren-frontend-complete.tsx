              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleLogin();
                }
              }}import React, { useState, useEffect, useRef, useCallback, Suspense, lazy } from 'react';
import { 
  Activity, Mic, MicOff, Brain, Shield, Zap, Users, 
  TrendingUp, AlertTriangle, CheckCircle, Clock,
  Command, Database, Cpu, Volume2, FileText, Upload, Loader
} from 'lucide-react';

// Environment configuration
const API_CONFIG = {
  wsEndpoint: process.env.REACT_APP_WS_ENDPOINT || 'wss://sovren.ai/ws/sovereign-stream',
  apiBase: process.env.REACT_APP_API_BASE || '/api',
  commandTimeout: 30000, // 30 seconds
  metricsInterval: 5000 // 5 seconds
};

// Type definitions
interface User {
  full_name: string;
  sovereignty_level: string;
  evolution_score: number;
  token: string;
}

interface Mission {
  mission_id: string;
  mission_type: string;
  status: 'active' | 'pending' | 'completed' | 'failed';
  progress: number;
  tasks_completed: number;
  total_tasks: number;
  created_at: string;
}

interface BattalionData {
  utilization: number;
  active_agents: number;
  total_agents: number;
  specialization: string;
}

interface SystemMetrics {
  decision_velocity: string;
}

interface CommandHistoryEntry {
  id: number;
  command: string;
  timestamp: Date;
  status: string;
  result?: any;
}

interface WebSocketMessage {
  type: string;
  mission?: Mission;
  status?: Record<string, BattalionData>;
  metrics?: SystemMetrics;
  response?: string;
  delta?: number;
}

interface StatusCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: string;
  color?: 'blue' | 'green' | 'purple' | 'yellow';
}

interface BattalionCardProps {
  name: string;
  data: BattalionData;
}

interface MissionCardProps {
  mission: Mission;
}

interface LoginScreenProps {
  onLogin: (userData: User) => void;
}

// Error Boundary Component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode; fallback: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('SOVREN Error Boundary:', error, errorInfo);
    // Send error metrics to local monitoring
    fetch(`${API_CONFIG.apiBase}/metrics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        metric: 'frontend_error',
        value: 1,
        labels: {
          error: error.toString(),
          component: errorInfo.componentStack
        }
      })
    }).catch(console.error);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// Error Fallback Component
const SovrenErrorFallback: React.FC<{ error?: Error }> = ({ error }) => (
  <div className="min-h-screen bg-black text-white flex items-center justify-center">
    <div className="text-center">
      <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
      <h1 className="text-2xl font-bold mb-2">SOVEREIGNTY BREACH DETECTED</h1>
      <p className="text-gray-400 mb-4">System encountered an error</p>
      {error && (
        <pre className="text-xs text-left bg-gray-900 p-4 rounded max-w-md mx-auto">
          {error.toString()}
        </pre>
      )}
      <button
        onClick={() => window.location.reload()}
        className="mt-4 px-6 py-2 bg-red-900 hover:bg-red-800 rounded"
      >
        REINITIALIZE SYSTEM
      </button>
    </div>
  </div>
);

// Loading Component
const SovrenLoader: React.FC = () => (
  <div className="flex items-center justify-center p-8">
    <Loader className="w-8 h-8 text-blue-500 animate-spin" />
    <span className="ml-2 text-gray-400">Loading sovereignty module...</span>
  </div>
);

// Performance monitoring hook
const usePerformanceMonitor = (componentName: string) => {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const renderTime = performance.now() - startTime;
      if (renderTime > 100) {
        console.warn(`Slow render detected in ${componentName}: ${renderTime}ms`);
        // Send metric to local monitoring
        fetch(`${API_CONFIG.apiBase}/metrics`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            metric: 'component_render_time',
            value: renderTime,
            labels: { component: componentName }
          })
        }).catch(console.error);
      }
    };
  });
};

// Metrics recording utility
const recordMetric = (metric: string, value: number, labels?: Record<string, string>) => {
  if (window.performance) {
    fetch(`${API_CONFIG.apiBase}/metrics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        metric, 
        value, 
        timestamp: Date.now(),
        labels 
      })
    }).catch(console.error);
  }
};

// Command validation utility
const validateCommand = (command: string): boolean => {
  // Check for injection attempts
  const forbidden = ['<script', 'javascript:', 'eval(', 'exec(', 'system(', '__proto__'];
  const lowercaseCommand = command.toLowerCase();
  
  for (const pattern of forbidden) {
    if (lowercaseCommand.includes(pattern)) {
      console.warn(`Potential injection attempt detected: ${pattern}`);
      return false;
    }
  }
  
  // Check command length
  if (command.length > 1000) {
    console.warn('Command exceeds maximum length');
    return false;
  }
  
  return true;
};

// Main SOVREN AI Executive Dashboard
function SovrenDashboard() {
  // Core state
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [voiceActive, setVoiceActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [missions, setMissions] = useState<Mission[]>([]);
  const [battalionStatus, setBattalionStatus] = useState<Record<string, BattalionData>>({});
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({} as SystemMetrics);
  const [commandHistory, setCommandHistory] = useState<CommandHistoryEntry[]>([]);
  const [activeTab, setActiveTab] = useState('command');
  
  // WebSocket refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Voice command state
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  // Performance monitoring
  usePerformanceMonitor('SovrenDashboard');

  // Initialize WebSocket connection with reconnection logic
  useEffect(() => {
    if (isAuthenticated && user) {
      connectWebSocket();
      loadDashboardData();
      
      // Set up metrics reporting interval
      const metricsInterval = setInterval(() => {
        recordMetric('dashboard_active', 1, { user_id: user.full_name });
      }, API_CONFIG.metricsInterval);
      
      return () => {
        clearInterval(metricsInterval);
        disconnectWebSocket();
      };
    }
  }, [isAuthenticated, user]);

  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket(API_CONFIG.wsEndpoint);
      
      wsRef.current.onopen = () => {
        console.log('Connected to SOVREN Command Stream');
        wsRef.current?.send(JSON.stringify({ type: 'authenticate', token: user?.token }));
        recordMetric('websocket_connection', 1, { status: 'connected' });
      };
      
      wsRef.current.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      wsRef.current.onerror = (error: Event) => {
        console.error('WebSocket error:', error);
        recordMetric('websocket_error', 1);
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        recordMetric('websocket_connection', 0, { status: 'disconnected' });
        // Attempt reconnection after 5 seconds
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  };

  const disconnectWebSocket = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    // Clean up audio resources
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current?.state !== 'closed') {
      audioContextRef.current?.close();
    }
  };

  const handleWebSocketMessage = (data: WebSocketMessage) => {
    switch (data.type) {
      case 'mission_update':
        if (data.mission) {
          updateMission(data.mission);
        }
        break;
      case 'battalion_status':
        if (data.status) {
          setBattalionStatus(data.status);
        }
        break;
      case 'system_metrics':
        if (data.metrics) {
          setSystemMetrics(data.metrics);
        }
        break;
      case 'ai_response':
        if (data.response) {
          setAiResponse(data.response);
        }
        setIsProcessing(false);
        break;
      case 'evolution_update':
        if (data.delta) {
          showEvolutionNotification(data.delta);
        }
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  const updateMission = (mission: Mission) => {
    setMissions(prev => {
      const existing = prev.find(m => m.mission_id === mission.mission_id);
      if (existing) {
        return prev.map(m => m.mission_id === mission.mission_id ? mission : m);
      } else {
        return [...prev, mission];
      }
    });
  };

  const loadDashboardData = async () => {
    try {
      // Load missions
      const missionsRes = await fetch(`${API_CONFIG.apiBase}/missions`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (missionsRes.ok) {
        const missionsData = await missionsRes.json();
        setMissions(missionsData.missions || []);
      }
      
      // Load battalion status
      const battalionRes = await fetch(`${API_CONFIG.apiBase}/battalion-metrics`, {
        headers: { 'Authorization': `Bearer ${user?.token}` }
      });
      if (battalionRes.ok) {
        const battalionData = await battalionRes.json();
        setBattalionStatus(battalionData.battalion_metrics || {});
      }
      
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      recordMetric('dashboard_load_error', 1);
    }
  };

  // Voice command handling with validation
  const startVoiceCommand = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      setIsListening(true);
      recordMetric('voice_command_started', 1);
      
      // Initialize audio context for voice visualization
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      source.connect(analyser);
      
      // Start streaming audio to server
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'start_voice_stream' }));
      }
      
    } catch (error) {
      console.error('Failed to start voice command:', error);
      setIsListening(false);
      recordMetric('voice_command_error', 1);
    }
  };

  const stopVoiceCommand = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    setIsListening(false);
    recordMetric('voice_command_stopped', 1);
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop_voice_stream' }));
    }
  };

  const executeTextCommand = async (command: string) => {
    // Validate command first
    if (!validateCommand(command)) {
      setAiResponse('Invalid command detected. Please try again.');
      return;
    }
    
    const commandStart = performance.now();
    setIsProcessing(true);
    
    const newCommand: CommandHistoryEntry = {
      id: Date.now(),
      command,
      timestamp: new Date(),
      status: 'processing'
    };
    
    setCommandHistory(prev => [newCommand, ...prev.slice(0, 9)]);
    
    try {
      const response = await fetch(`${API_CONFIG.apiBase}/execute-command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user?.token}`
        },
        body: JSON.stringify({ command }),
        signal: AbortSignal.timeout(API_CONFIG.commandTimeout)
      });
      
      if (!response.ok) {
        throw new Error(`Command failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      setAiResponse(result.response);
      
      // Update command history
      setCommandHistory(prev => {
        const updated = [...prev];
        const cmdIndex = updated.findIndex(c => c.id === newCommand.id);
        if (cmdIndex !== -1) {
          updated[cmdIndex] = {
            ...updated[cmdIndex],
            status: 'completed',
            result
          };
        }
        return updated;
      });
      
      // Record performance metrics
      const commandDuration = performance.now() - commandStart;
      recordMetric('command_execution_time', commandDuration, { 
        status: 'success',
        command_type: command.split(' ')[0]
      });
      
    } catch (error) {
      console.error('Command execution failed:', error);
      setAiResponse('Command execution failed. Please try again.');
      
      // Update command history with failure
      setCommandHistory(prev => {
        const updated = [...prev];
        const cmdIndex = updated.findIndex(c => c.id === newCommand.id);
        if (cmdIndex !== -1) {
          updated[cmdIndex] = {
            ...updated[cmdIndex],
            status: 'failed'
          };
        }
        return updated;
      });
      
      recordMetric('command_execution_error', 1);
    } finally {
      setIsProcessing(false);
    }
  };

  // Battalion status component with performance tracking
  const BattalionCard: React.FC<BattalionCardProps> = React.memo(({ name, data }) => {
    usePerformanceMonitor(`BattalionCard-${name}`);
    
    const iconMap: Record<string, React.ComponentType<any>> = {
      'strike': Zap,
      'intel': Brain,
      'ops': Cpu,
      'sentinel': Shield,
      'command': Command
    };
    const Icon = iconMap[name] || Activity;
    
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-blue-700 transition-all">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Icon className="w-5 h-5 text-blue-500" aria-hidden="true" />
            <h3 className="text-white font-semibold uppercase">{name}</h3>
          </div>
          <span className={`text-xs px-2 py-1 rounded ${
            data?.utilization > 80 ? 'bg-red-900 text-red-200' : 'bg-green-900 text-green-200'
          }`}>
            {data?.utilization || 0}% ACTIVE
          </span>
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Agents</span>
            <span className="text-white">{data?.active_agents || 0}/{data?.total_agents || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Specialization</span>
            <span className="text-blue-400 text-xs">{data?.specialization || 'Loading...'}</span>
          </div>
        </div>
      </div>
    );
  });

  // Mission card component
  const MissionCard: React.FC<MissionCardProps> = React.memo(({ mission }) => {
    const statusColors: Record<string, string> = {
      'active': 'text-green-400',
      'pending': 'text-yellow-400',
      'completed': 'text-blue-400',
      'failed': 'text-red-400'
    };
    
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-purple-700 transition-all">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-white font-medium">{mission.mission_type}</h4>
          <span className={`text-sm ${statusColors[mission.status] || 'text-gray-400'}`}>
            {mission.status?.toUpperCase()}
          </span>
        </div>
        <div className="mb-3">
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full transition-all"
              style={{ width: `${mission.progress || 0}%` }}
              role="progressbar"
              aria-valuenow={mission.progress || 0}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>
        <div className="text-xs text-gray-400">
          <div>Tasks: {mission.tasks_completed || 0}/{mission.total_tasks || 0}</div>
          <div>Deployed: {new Date(mission.created_at).toLocaleTimeString()}</div>
        </div>
      </div>
    );
  });

  // Login component
  if (!isAuthenticated) {
    return (
      <ErrorBoundary fallback={<SovrenErrorFallback />}>
        <LoginScreen onLogin={(userData: User) => {
          setUser(userData);
          setIsAuthenticated(true);
        }} />
      </ErrorBoundary>
    );
  }

  // Main dashboard with error boundary and suspense
  return (
    <ErrorBoundary fallback={<SovrenErrorFallback />}>
      <div className="min-h-screen bg-black text-white">
        {/* Header */}
        <header className="border-b border-gray-800 bg-gray-950">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Command className="w-8 h-8 text-blue-500" aria-hidden="true" />
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                    SOVREN AI COMMAND
                  </h1>
                  <p className="text-xs text-gray-400">SOVEREIGN EXECUTION MATRIX</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="text-sm text-gray-400">Commander</p>
                  <p className="font-medium">{user?.full_name}</p>
                  <p className="text-xs text-purple-400">{user?.sovereignty_level?.toUpperCase()}</p>
                </div>
                <button 
                  onClick={() => {
                    setIsAuthenticated(false);
                    disconnectWebSocket();
                  }}
                  className="px-4 py-2 bg-red-900 hover:bg-red-800 rounded text-sm transition-colors"
                  aria-label="Log out"
                >
                  DISENGAGE
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Tab Navigation */}
        <div className="border-b border-gray-800 bg-gray-950">
          <div className="max-w-7xl mx-auto px-4">
            <nav className="flex gap-6" role="tablist">
              {['command', 'battalions', 'missions', 'intelligence', 'evolution'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`py-3 px-4 text-sm font-medium border-b-2 transition-all ${
                    activeTab === tab 
                      ? 'border-blue-500 text-blue-400' 
                      : 'border-transparent text-gray-400 hover:text-white'
                  }`}
                  role="tab"
                  aria-selected={activeTab === tab}
                  aria-controls={`${tab}-panel`}
                >
                  {tab.toUpperCase()}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Main Content with Suspense */}
        <main className="max-w-7xl mx-auto px-4 py-6">
          <Suspense fallback={<SovrenLoader />}>
            {/* Command Center Tab */}
            {activeTab === 'command' && (
              <div className="space-y-6" role="tabpanel" id="command-panel">
                {/* Voice Command Interface */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                  <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <Mic className="w-5 h-5 text-blue-500" aria-hidden="true" />
                    VOICE OF COMMAND
                  </h2>
                  
                  <div className="flex items-center gap-4 mb-6">
                    <button
                      onClick={isListening ? stopVoiceCommand : startVoiceCommand}
                      className={`p-6 rounded-full transition-all ${
                        isListening 
                          ? 'bg-red-900 hover:bg-red-800 animate-pulse' 
                          : 'bg-blue-900 hover:bg-blue-800'
                      }`}
                      aria-label={isListening ? "Stop voice command" : "Start voice command"}
                      aria-pressed={isListening}
                    >
                      {isListening ? <MicOff className="w-8 h-8" /> : <Mic className="w-8 h-8" />}
                    </button>
                    
                    <div className="flex-1">
                      <input
                        type="text"
                        value={currentTranscript}
                        onChange={(e) => setCurrentTranscript(e.target.value)}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' && currentTranscript && !isProcessing) {
                            executeTextCommand(currentTranscript);
                            setCurrentTranscript('');
                          }
                        }}
                        placeholder="Issue command..."
                        className="w-full bg-gray-800 border border-gray-700 rounded px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                        disabled={isProcessing}
                        aria-label="Command input"
                      />
                    </div>
                    
                    <button
                      onClick={() => {
                        if (currentTranscript && !isProcessing) {
                          executeTextCommand(currentTranscript);
                          setCurrentTranscript('');
                        }
                      }}
                      disabled={!currentTranscript || isProcessing}
                      className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transition-all"
                      aria-label="Execute command"
                    >
                      EXECUTE
                    </button>
                  </div>

                  {/* AI Response */}
                  {(isProcessing || aiResponse) && (
                    <div className="bg-gray-800 rounded-lg p-4" role="status" aria-live="polite">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="w-4 h-4 text-purple-500" aria-hidden="true" />
                        <span className="text-sm font-medium text-purple-400">SOVEREIGN INTELLIGENCE</span>
                      </div>
                      {isProcessing ? (
                        <div className="flex items-center gap-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-2 border-purple-500 border-t-transparent" />
                          <span className="text-gray-400">Processing command...</span>
                        </div>
                      ) : (
                        <p className="text-white">{aiResponse}</p>
                      )}
                    </div>
                  )}

                  {/* Command History */}
                  <div className="mt-6">
                    <h3 className="text-sm font-medium text-gray-400 mb-3">COMMAND HISTORY</h3>
                    <div className="space-y-2">
                      {commandHistory.slice(0, 5).map((cmd) => (
                        <div key={cmd.id} className="flex items-center justify-between text-sm">
                          <span className="text-gray-300">{cmd.command}</span>
                          <div className="flex items-center gap-2">
                            <span className={`text-xs ${
                              cmd.status === 'completed' ? 'text-green-400' :
                              cmd.status === 'failed' ? 'text-red-400' :
                              'text-yellow-400'
                            }`}>
                              {cmd.status.toUpperCase()}
                            </span>
                            <span className="text-xs text-gray-500">
                              {cmd.timestamp.toLocaleTimeString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* System Status */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatusCard
                    title="Decision Velocity"
                    value={systemMetrics.decision_velocity || '< 200ms'}
                    icon={<Zap className="w-5 h-5" />}
                    trend="+12%"
                    color="blue"
                  />
                  <StatusCard
                    title="Active Missions"
                    value={missions.filter(m => m.status === 'active').length}
                    icon={<Activity className="w-5 h-5" />}
                    trend="+3"
                    color="green"
                  />
                  <StatusCard
                    title="Evolution Score"
                    value={user?.evolution_score?.toFixed(1) || '0.0'}
                    icon={<TrendingUp className="w-5 h-5" />}
                    trend="+0.1"
                    color="purple"
                  />
                  <StatusCard
                    title="Sovereignty Level"
                    value={user?.sovereignty_level?.toUpperCase() || 'FOUNDATION'}
                    icon={<Shield className="w-5 h-5" />}
                    trend="STABLE"
                    color="yellow"
                  />
                </div>
              </div>
            )}

            {/* Battalions Tab */}
            {activeTab === 'battalions' && (
              <div className="space-y-6" role="tabpanel" id="battalions-panel">
                <h2 className="text-2xl font-bold">SYNTHETIC AGENT BATTALIONS</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(battalionStatus).map(([name, data]) => (
                    <BattalionCard key={name} name={name} data={data} />
                  ))}
                </div>
                
                {/* Battalion Performance Metrics */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mt-6">
                  <h3 className="text-lg font-semibold mb-4">OPERATIONAL METRICS</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <p className="text-gray-400 text-sm">Total Agent Deployments</p>
                      <p className="text-3xl font-bold text-blue-400">1,247</p>
                      <p className="text-xs text-green-400">+47% vs yesterday</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Success Rate</p>
                      <p className="text-3xl font-bold text-green-400">94.3%</p>
                      <p className="text-xs text-gray-400">Industry avg: 62%</p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Avg Response Time</p>
                      <p className="text-3xl font-bold text-purple-400">0.7s</p>
                      <p className="text-xs text-blue-400">-200ms improvement</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Missions Tab */}
            {activeTab === 'missions' && (
              <div className="space-y-6" role="tabpanel" id="missions-panel">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold">ACTIVE MISSIONS</h2>
                  <button className="px-4 py-2 bg-blue-900 hover:bg-blue-800 rounded font-medium transition-colors">
                    DEPLOY NEW MISSION
                  </button>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {missions.map((mission) => (
                    <MissionCard key={mission.mission_id} mission={mission} />
                  ))}
                </div>
                
                {missions.length === 0 && (
                  <div className="text-center py-12 text-gray-400">
                    <Command className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>No active missions. Deploy your first autonomous operation.</p>
                  </div>
                )}
              </div>
            )}

            {/* Intelligence Tab */}
            {activeTab === 'intelligence' && (
              <div className="space-y-6" role="tabpanel" id="intelligence-panel">
                <h2 className="text-2xl font-bold">KNOWLEDGE SOVEREIGNTY</h2>
                
                {/* Document Upload */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Upload className="w-5 h-5 text-blue-500" aria-hidden="true" />
                    INGEST INTELLIGENCE
                  </h3>
                  
                  <div 
                    className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer"
                    role="button"
                    tabIndex={0}
                    aria-label="Upload documents"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        // Trigger file upload
                        console.log('Upload triggered');
                      }
                    }}
                  >
                    <FileText className="w-12 h-12 mx-auto mb-4 text-gray-600" />
                    <p className="text-gray-400 mb-2">Drop documents here or click to upload</p>
                    <p className="text-xs text-gray-500">PDF, DOCX, TXT, MD, HTML, CSV, XLSX</p>
                  </div>
                  
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-gray-800 rounded p-3">
                      <p className="text-gray-400">Total Documents</p>
                      <p className="text-2xl font-bold text-white">847</p>
                    </div>
                    <div className="bg-gray-800 rounded p-3">
                      <p className="text-gray-400">Knowledge Chunks</p>
                      <p className="text-2xl font-bold text-blue-400">124,532</p>
                    </div>
                    <div className="bg-gray-800 rounded p-3">
                      <p className="text-gray-400">Vector Dimensions</p>
                      <p className="text-2xl font-bold text-purple-400">384</p>
                    </div>
                  </div>
                </div>
                
                {/* Recent Intelligence */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">RECENT INTELLIGENCE ACQUISITION</h3>
                  <div className="space-y-3">
                    {['Q4 Market Analysis.pdf', 'Competitor Intelligence Report.docx', 'Customer Feedback Survey.xlsx'].map((doc, i) => (
                      <div key={i} className="flex items-center justify-between py-2">
                        <div className="flex items-center gap-3">
                          <FileText className="w-4 h-4 text-gray-500" />
                          <span className="text-gray-300">{doc}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-green-500" />
                          <span className="text-xs text-gray-500">Processed</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Evolution Tab */}
            {activeTab === 'evolution' && (
              <div className="space-y-6" role="tabpanel" id="evolution-panel">
                <h2 className="text-2xl font-bold">SYSTEM EVOLUTION</h2>
                
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">LEARNING METRICS</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <p className="text-gray-400 mb-2">Bayesian Policy Updates</p>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Primary Reasoning</span>
                          <span className="text-blue-400">α: 47, β: 8</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Efficient Support</span>
                          <span className="text-green-400">α: 35, β: 12</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Broad Knowledge</span>
                          <span className="text-purple-400">α: 29, β: 15</span>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-gray-400 mb-2">Evolution Timeline</p>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-gray-500" />
                          <span>Memory-Infinite Agents: <span className="text-green-400">ACTIVE</span></span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-gray-500" />
                          <span>Real-time LLM Routing: <span className="text-yellow-400">IN PROGRESS</span></span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-gray-500" />
                          <span>Federated Learning: <span className="text-gray-400">QUEUED</span></span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6 p-4 bg-gray-800 rounded">
                    <p className="text-sm text-gray-400 mb-1">System Intelligence Level</p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1 bg-gray-700 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 h-3 rounded-full"
                          style={{ width: '73%' }}
                          role="progressbar"
                          aria-valuenow={73}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        />
                      </div>
                      <span className="text-lg font-bold text-white">EVOLVING</span>
                    </div>
                  </div>
                </div>
                
                {/* Sovereignty Comparison */}
                <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-800 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">SOVEREIGNTY ADVANTAGE</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Your Position</p>
                      <p className="text-2xl font-bold text-purple-400 mt-1">TOP 3%</p>
                      <p className="text-xs text-gray-500 mt-1">Among all SOVREN operators</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Competitive Edge</p>
                      <p className="text-2xl font-bold text-blue-400 mt-1">47x FASTER</p>
                      <p className="text-xs text-gray-500 mt-1">Than manual operations</p>
                    </div>
                  </div>
                  
                  {user?.sovereignty_level === 'foundation' && (
                    <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-800 rounded">
                      <p className="text-sm text-yellow-400">
                        Upgrade to ACCELERATION tier to unlock 2x evolution speed and priority routing.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </Suspense>
        </main>
      </div>
    </ErrorBoundary>
  );
}

// Status card component
const StatusCard: React.FC<StatusCardProps> = React.memo(({ title, value, icon, trend, color = 'blue' }) => {
  const colorClasses: Record<string, string> = {
    blue: 'text-blue-400 bg-blue-900/20 border-blue-800',
    green: 'text-green-400 bg-green-900/20 border-green-800',
    purple: 'text-purple-400 bg-purple-900/20 border-purple-800',
    yellow: 'text-yellow-400 bg-yellow-900/20 border-yellow-800'
  };
  
  return (
    <div className={`bg-gray-900 border rounded-lg p-4 ${colorClasses[color] || colorClasses.blue}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">{title}</span>
        {icon}
      </div>
      <div className="flex items-end justify-between">
        <p className="text-2xl font-bold">{value}</p>
        {trend && (
          <span className={`text-xs ${trend.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
            {trend}
          </span>
        )}
      </div>
    </div>
  );
});

// Login screen component
const LoginScreen: React.FC<LoginScreenProps> = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  usePerformanceMonitor('LoginScreen');

  const handleLogin = async () => {
    setIsLoading(true);
    setError('');
    
    // Basic validation
    if (!email || !password) {
      setError('Please enter both email and password');
      setIsLoading(false);
      return;
    }
    
    try {
      const response = await fetch(`${API_CONFIG.apiBase}/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          username: email,
          password: password
        })
      });
      
      if (!response.ok) {
        throw new Error('Authentication failed');
      }
      
      const data = await response.json();
      
      // Get user profile
      const profileRes = await fetch(`${API_CONFIG.apiBase}/me`, {
        headers: { 'Authorization': `Bearer ${data.access_token}` }
      });
      
      if (!profileRes.ok) {
        throw new Error('Failed to load user profile');
      }
      
      const profile = await profileRes.json();
      
      onLogin({
        ...profile,
        token: data.access_token
      });
      
      recordMetric('login_success', 1);
      
    } catch (err) {
      setError('Sovereignty credentials not recognized');
      recordMetric('login_failure', 1);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <Command className="w-16 h-16 text-blue-500 mx-auto mb-4" aria-hidden="true" />
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent mb-2">
            SOVREN AI
          </h1>
          <p className="text-gray-400">SOVEREIGN EXECUTION MATRIX</p>
        </div>
        
        <div className="space-y-4">
          <div>
            <label htmlFor="email" className="sr-only">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Sovereignty Email"
              required
              className="w-full bg-gray-900 border border-gray-800 rounded px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              aria-label="Email address"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleLogin();
                }
              }}
            />
          </div>
          
          <div>
            <label htmlFor="password" className="sr-only">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Authorization Code"
              required
              className="w-full bg-gray-900 border border-gray-800 rounded px-4 py-3 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              aria-label="Password"
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleLogin();
                }
              }}
            />
          </div>
          
          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded p-3 text-red-400 text-sm" role="alert">
              {error}
            </div>
          )}
          
          <button
            type="button"
            onClick={handleLogin}
            disabled={isLoading}
            className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transition-all"
            aria-label={isLoading ? 'Authenticating' : 'Log in'}
          >
            {isLoading ? 'AUTHENTICATING...' : 'ASSUME COMMAND'}
          </button>
        </div>
        
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>Not sovereign yet?</p>
          <a 
            href="https://sovren.ai" 
            className="text-blue-400 hover:text-blue-300"
            target="_blank"
            rel="noopener noreferrer"
          >
            Apply for dominion access
          </a>
        </div>
      </div>
    </div>
  );
};

// Notification helper
function showEvolutionNotification(delta: number) {
  // Create a custom notification element
  const notification = document.createElement('div');
  notification.className = 'fixed top-4 right-4 bg-purple-900 border border-purple-700 rounded-lg p-4 text-white shadow-lg z-50';
  notification.innerHTML = `
    <div class="flex items-center gap-2">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
      </svg>
      <span>Evolution increased by ${delta.toFixed(2)}</span>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.remove();
  }, 3000);
  
  // Also record the metric
  recordMetric('evolution_increase', delta);
}

// Export the main component
export default SovrenDashboard;
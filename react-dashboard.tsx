import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  Activity, Mic, MicOff, Brain, Shield, Zap, Users, 
  TrendingUp, AlertTriangle, CheckCircle, Clock,
  Command, Database, Cpu, Volume2, FileText, Upload,
  Target, Briefcase, DollarSign, UserCheck, Phone,
  Mail, Calendar, Settings, ChevronRight, Play,
  Pause, BarChart3, Globe, Lock, Rocket, X
} from 'lucide-react';

// WebSocket hook for real-time updates
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setData(message);
    };

    return () => ws.current?.close();
  }, [url]);

  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  }, []);

  return { data, isConnected, sendMessage };
};

// Main Dashboard Component
const SOVRENDashboard = () => {
  const [user, setUser] = useState(null);
  const [mode, setMode] = useState('collaborative');
  const [activeView, setActiveView] = useState('overview');
  const [morningBrief, setMorningBrief] = useState(null);
  const [liveMetrics, setLiveMetrics] = useState({
    valueCreated: 0,
    tasksCompleted: 0,
    decisionsSupported: 0,
    timesSaved: 0
  });

  // WebSocket connections
  const { data: wsData, isConnected } = useWebSocket('wss://localhost:8443/ws/dashboard');
  const { sendMessage } = useWebSocket('wss://localhost:8443/ws/commands');

  // Process WebSocket data
  useEffect(() => {
    if (!wsData) return;

    switch (wsData.type) {
      case 'metrics_update':
        setLiveMetrics(prev => ({
          ...prev,
          ...wsData.metrics
        }));
        break;
      case 'morning_brief':
        setMorningBrief(wsData.brief);
        break;
      case 'mode_change':
        setMode(wsData.mode);
        break;
    }
  }, [wsData]);

  // Load user data
  useEffect(() => {
    fetch('/api/user/current')
      .then(res => res.json())
      .then(data => setUser(data))
      .catch(console.error);
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <Header 
        user={user} 
        mode={mode} 
        onModeChange={setMode}
        isConnected={isConnected}
      />
      
      <div className="flex">
        <Sidebar 
          activeView={activeView} 
          onViewChange={setActiveView}
          userTier={user?.tier}
        />
        
        <main className="flex-1 p-6">
          {activeView === 'overview' && (
            <OverviewDashboard 
              user={user}
              morningBrief={morningBrief}
              metrics={liveMetrics}
            />
          )}
          {activeView === 'shadow-board' && user?.tier !== 'ENTERPRISE' && (
            <ShadowBoardView />
          )}
          {activeView === 'agents' && (
            <AgentBattalionView />
          )}
          {activeView === 'time-machine' && (
            <TimeMachineView />
          )}
          {activeView === 'integrations' && (
            <IntegrationsView />
          )}
          {activeView === 'settings' && (
            <SettingsView user={user} />
          )}
        </main>
      </div>
    </div>
  );
};

// Header Component
const Header = ({ user, mode, onModeChange, isConnected }) => {
  const [showVoiceIndicator, setShowVoiceIndicator] = useState(false);

  return (
    <header className="border-b border-gray-800 bg-black/50 backdrop-blur sticky top-0 z-50">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-4">
          <Command className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-2xl font-bold">SOVREN</h1>
            <p className="text-xs text-gray-400">
              PhD-Level Chief of Staff
              {user?.tier === 'SMB' && ' + Shadow Board'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <ConnectionStatus isConnected={isConnected} />
          <ModeToggle mode={mode} onChange={onModeChange} />
          <SOVRENScore score={user?.sovrenScore || 0} />
          <ValueTicker value={0} />
          <VoiceControl 
            active={showVoiceIndicator} 
            onToggle={() => setShowVoiceIndicator(!showVoiceIndicator)}
          />
        </div>
      </div>
    </header>
  );
};

// Sidebar Navigation
const Sidebar = ({ activeView, onViewChange, userTier }) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'shadow-board', label: 'Shadow Board', icon: Users, hideFor: 'ENTERPRISE' },
    { id: 'agents', label: 'Agent Battalion', icon: Zap },
    { id: 'time-machine', label: 'Time Machine', icon: Clock },
    { id: 'integrations', label: 'Integrations', icon: Globe },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  const filteredItems = navItems.filter(item => 
    !item.hideFor || item.hideFor !== userTier
  );

  return (
    <div className="w-64 bg-gray-900 border-r border-gray-800 h-[calc(100vh-73px)]">
      <nav className="p-4 space-y-2">
        {filteredItems.map(item => (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
              activeView === item.id 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
};

// Mode Toggle Component
const ModeToggle = ({ mode, onChange }) => {
  return (
    <div className="flex items-center gap-2 bg-gray-900 rounded-lg p-1">
      <button
        onClick={() => onChange('collaborative')}
        className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
          mode === 'collaborative' 
            ? 'bg-blue-600 text-white' 
            : 'text-gray-400 hover:text-white'
        }`}
      >
        Collaborative
      </button>
      <button
        onClick={() => onChange('autonomous')}
        className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
          mode === 'autonomous' 
            ? 'bg-green-600 text-white' 
            : 'text-gray-400 hover:text-white'
        }`}
      >
        Autonomous
      </button>
    </div>
  );
};

// Connection Status
const ConnectionStatus = ({ isConnected }) => {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${
        isConnected ? 'bg-green-500' : 'bg-red-500'
      }`} />
      <span className="text-xs text-gray-400">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
};

// SOVREN Score Display
const SOVRENScore = ({ score }) => {
  const getScoreColor = (score) => {
    if (score >= 800) return 'text-green-400';
    if (score >= 600) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex items-center gap-2">
      <Target className="w-5 h-5 text-gray-400" />
      <div>
        <p className="text-xs text-gray-400">SOVREN Score</p>
        <p className={`text-xl font-bold ${getScoreColor(score)}`}>
          {score}
        </p>
      </div>
    </div>
  );
};

// Value Ticker
const ValueTicker = ({ value }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setDisplayValue(prev => {
        const increment = Math.random() * 100;
        return prev + increment;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-green-900/20 border border-green-800 rounded-lg px-4 py-2">
      <p className="text-xs text-green-400">Value Created Today</p>
      <p className="text-2xl font-bold text-green-400">
        ${(displayValue + value).toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </p>
    </div>
  );
};

// Voice Control
const VoiceControl = ({ active, onToggle }) => {
  return (
    <button
      onClick={onToggle}
      className={`p-2 rounded-lg transition-all ${
        active 
          ? 'bg-red-600 text-white' 
          : 'bg-gray-800 text-gray-400 hover:text-white'
      }`}
    >
      {active ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
    </button>
  );
};

// Overview Dashboard
const OverviewDashboard = ({ user, morningBrief, metrics }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">
          Good {getTimeOfDay()}, {user?.name || 'there'}
        </h2>
        <button className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors">
          View Full Report
        </button>
      </div>

      {morningBrief && <MorningBriefPanel brief={morningBrief} />}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Value Created"
          value={`$${metrics.valueCreated.toLocaleString()}`}
          change="+23%"
          icon={DollarSign}
          color="green"
        />
        <MetricCard
          title="Tasks Completed"
          value={metrics.tasksCompleted}
          change="+89"
          icon={CheckCircle}
          color="blue"
        />
        <MetricCard
          title="Decisions Supported"
          value={metrics.decisionsSupported}
          change="+12"
          icon={Brain}
          color="purple"
        />
        <MetricCard
          title="Time Saved"
          value={`${metrics.timesSaved}h`}
          change="+47%"
          icon={Clock}
          color="yellow"
        />
      </div>

      <LiveActivityFeed />
      <StrategicInsights />
    </div>
  );
};

// Morning Brief Panel
const MorningBriefPanel = ({ brief }) => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
        <Clock className="w-5 h-5 text-blue-500" />
        What I Accomplished While You Slept
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {brief?.achievements?.map((achievement, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">{achievement.icon}</span>
              <div className="flex-1">
                <p className="text-sm text-white">{achievement.text}</p>
                <p className="text-lg font-bold text-green-400 mt-1">
                  {achievement.value}
                </p>
                {achievement.method && (
                  <p className="text-xs text-gray-500 mt-1">
                    Method: {achievement.method}
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Metric Card
const MetricCard = ({ title, value, change, icon: Icon, color }) => {
  const colorClasses = {
    green: 'bg-green-900/20 border-green-800 text-green-400',
    blue: 'bg-blue-900/20 border-blue-800 text-blue-400',
    purple: 'bg-purple-900/20 border-purple-800 text-purple-400',
    yellow: 'bg-yellow-900/20 border-yellow-800 text-yellow-400'
  };

  return (
    <div className={`p-6 rounded-lg border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-4">
        <Icon className="w-8 h-8" />
        <span className="text-sm font-medium">{change}</span>
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-sm mt-1 text-gray-400">{title}</p>
    </div>
  );
};

// Live Activity Feed
const LiveActivityFeed = () => {
  const [activities, setActivities] = useState([]);

  useEffect(() => {
    // Simulate live activities
    const interval = setInterval(() => {
      const newActivity = {
        id: Date.now(),
        type: ['task', 'decision', 'insight', 'alert'][Math.floor(Math.random() * 4)],
        message: 'New activity from SOVREN',
        timestamp: new Date().toLocaleTimeString()
      };
      
      setActivities(prev => [newActivity, ...prev.slice(0, 9)]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Live Activity</h3>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {activities.map(activity => (
          <div key={activity.id} className="flex items-start gap-3 p-3 bg-gray-800 rounded-lg">
            <ActivityIcon type={activity.type} />
            <div className="flex-1">
              <p className="text-sm text-white">{activity.message}</p>
              <p className="text-xs text-gray-400 mt-1">{activity.timestamp}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Activity Icon
const ActivityIcon = ({ type }) => {
  const icons = {
    task: <CheckCircle className="w-5 h-5 text-green-400" />,
    decision: <Brain className="w-5 h-5 text-blue-400" />,
    insight: <TrendingUp className="w-5 h-5 text-purple-400" />,
    alert: <AlertTriangle className="w-5 h-5 text-yellow-400" />
  };

  return icons[type] || icons.task;
};

// Strategic Insights
const StrategicInsights = () => {
  const insights = [
    {
      title: "Churn Risk Detected",
      description: "Johnson Corp showing early warning signs. Intervention recommended.",
      action: "View Analysis",
      severity: "high"
    },
    {
      title: "Growth Opportunity",
      description: "European expansion feasibility: 73% success probability based on simulations.",
      action: "Explore Options",
      severity: "medium"
    },
    {
      title: "Cost Optimization",
      description: "$23K/month potential savings identified in vendor contracts.",
      action: "Review Savings",
      severity: "low"
    }
  ];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Strategic Insights</h3>
      <div className="space-y-4">
        {insights.map((insight, i) => (
          <InsightCard key={i} insight={insight} />
        ))}
      </div>
    </div>
  );
};

// Insight Card
const InsightCard = ({ insight }) => {
  const severityColors = {
    high: 'border-red-800 bg-red-900/20',
    medium: 'border-yellow-800 bg-yellow-900/20',
    low: 'border-green-800 bg-green-900/20'
  };

  return (
    <div className={`p-4 rounded-lg border ${severityColors[insight.severity]}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="font-semibold text-white">{insight.title}</h4>
          <p className="text-sm text-gray-400 mt-1">{insight.description}</p>
        </div>
        <button className="px-3 py-1 bg-white/10 rounded text-sm hover:bg-white/20 transition-colors">
          {insight.action}
        </button>
      </div>
    </div>
  );
};

// Shadow Board View (SMB Only)
const ShadowBoardView = () => {
  const [executives, setExecutives] = useState([]);
  const [selectedExec, setSelectedExec] = useState(null);

  useEffect(() => {
    // Load Shadow Board executives
    fetch('/api/shadow-board/executives')
      .then(res => res.json())
      .then(data => setExecutives(data))
      .catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Your Shadow Board</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {executives.map(exec => (
          <ExecutiveCard 
            key={exec.role} 
            executive={exec}
            onClick={() => setSelectedExec(exec)}
          />
        ))}
      </div>

      {selectedExec && (
        <ExecutiveDetailPanel 
          executive={selectedExec}
          onClose={() => setSelectedExec(null)}
        />
      )}
    </div>
  );
};

// Executive Card
const ExecutiveCard = ({ executive, onClick }) => {
  const roleColors = {
    CFO: 'bg-green-900/20 border-green-800',
    CMO: 'bg-purple-900/20 border-purple-800',
    Legal: 'bg-blue-900/20 border-blue-800',
    CTO: 'bg-orange-900/20 border-orange-800'
  };

  return (
    <div 
      onClick={onClick}
      className={`p-6 rounded-lg border cursor-pointer hover:bg-white/5 transition-all ${
        roleColors[executive.role]
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold
          ${executive.role === 'CFO' ? 'bg-green-900 text-green-400' :
            executive.role === 'CMO' ? 'bg-purple-900 text-purple-400' :
            executive.role === 'Legal' ? 'bg-blue-900 text-blue-400' :
            'bg-orange-900 text-orange-400'}`}>
          {executive.role.slice(0, 2)}
        </div>
        <Phone className="w-5 h-5 text-gray-400" />
      </div>
      
      <h3 className="font-semibold text-white">{executive.name}</h3>
      <p className="text-sm text-gray-400">{executive.role}</p>
      
      <div className="mt-4 space-y-2">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <CheckCircle className="w-3 h-3" />
          <span>{executive.tasksToday} tasks today</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Brain className="w-3 h-3" />
          <span>{executive.decisionsToday} decisions</span>
        </div>
      </div>
    </div>
  );
};

// Executive Detail Panel
const ExecutiveDetailPanel = ({ executive, onClose }) => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center text-2xl font-bold">
            {executive.role.slice(0, 2)}
          </div>
          <div>
            <h3 className="text-2xl font-bold">{executive.name}</h3>
            <p className="text-gray-400">{executive.role} • {executive.credentials}</p>
          </div>
        </div>
        <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-400">Recent Actions</h4>
          {executive.recentActions?.map((action, i) => (
            <div key={i} className="text-sm">
              <p className="text-white">{action.description}</p>
              <p className="text-xs text-gray-500">{action.timestamp}</p>
            </div>
          ))}
        </div>
        
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-400">Key Insights</h4>
          {executive.insights?.map((insight, i) => (
            <div key={i} className="text-sm p-3 bg-gray-800 rounded">
              <p className="text-white">{insight}</p>
            </div>
          ))}
        </div>
        
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-400">Contact</h4>
          <button className="w-full px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors">
            Call Now
          </button>
          <button className="w-full px-4 py-2 bg-gray-800 rounded hover:bg-gray-700 transition-colors">
            Schedule Meeting
          </button>
        </div>
      </div>
    </div>
  );
};

// Agent Battalion View
const AgentBattalionView = () => {
  const [agents, setAgents] = useState({
    operational: [],
    analytical: [],
    strategic: [],
    specialist: []
  });

  useEffect(() => {
    // Load agent status
    fetch('/api/agents/status')
      .then(res => res.json())
      .then(data => setAgents(data))
      .catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Agent Battalion Status</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Object.entries(agents).map(([type, agentList]) => (
          <AgentPoolCard key={type} type={type} agents={agentList} />
        ))}
      </div>

      <ActiveAgentsPanel agents={Object.values(agents).flat()} />
    </div>
  );
};

// Agent Pool Card
const AgentPoolCard = ({ type, agents }) => {
  const poolInfo = {
    operational: { color: 'blue', icon: Zap },
    analytical: { color: 'purple', icon: Brain },
    strategic: { color: 'green', icon: Target },
    specialist: { color: 'orange', icon: Rocket }
  };

  const info = poolInfo[type];
  const Icon = info.icon;

  return (
    <div className={`p-6 rounded-lg border bg-${info.color}-900/20 border-${info.color}-800`}>
      <div className="flex items-center justify-between mb-4">
        <Icon className={`w-8 h-8 text-${info.color}-400`} />
        <span className="text-2xl font-bold">{agents.length}</span>
      </div>
      <h3 className="font-semibold capitalize">{type} Agents</h3>
      <div className="mt-2 space-y-1 text-sm text-gray-400">
        <p>Active: {agents.filter(a => a.status === 'active').length}</p>
        <p>PhD Level: {agents.filter(a => a.intelligence === 'phd').length}</p>
      </div>
    </div>
  );
};

// Active Agents Panel
const ActiveAgentsPanel = ({ agents }) => {
  const activeAgents = agents.filter(a => a.status === 'active');

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Active Agent Operations</h3>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {activeAgents.map(agent => (
          <div key={agent.id} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
            <div className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full ${
                agent.intelligence === 'phd' ? 'bg-purple-500' :
                agent.intelligence === 'enhanced' ? 'bg-blue-500' :
                'bg-gray-500'
              }`} />
              <div>
                <p className="text-sm font-medium">{agent.id}</p>
                <p className="text-xs text-gray-400">{agent.currentTask}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-400">{agent.intelligence}</p>
              <p className="text-xs text-gray-500">{agent.duration}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Time Machine View
const TimeMachineView = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const runTimeQuery = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/time-machine/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Time query error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Time Machine</h2>
      
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex gap-4 mb-6">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What if we had raised prices 3 months ago?"
            className="flex-1 px-4 py-2 bg-gray-800 rounded-lg text-white placeholder-gray-500"
          />
          <button
            onClick={runTimeQuery}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Run Query'}
          </button>
        </div>

        {results && <TimeQueryResults results={results} />}
      </div>

      <RecentTimeQueries />
    </div>
  );
};

// Time Query Results
const TimeQueryResults = ({ results }) => {
  return (
    <div className="space-y-4">
      <div className="p-4 bg-gray-800 rounded-lg">
        <h4 className="font-semibold mb-2">Analysis Results</h4>
        <p className="text-sm text-gray-400">{results.summary}</p>
      </div>
      
      {results.alternativeTimeline && (
        <div className="p-4 bg-blue-900/20 border border-blue-800 rounded-lg">
          <h4 className="font-semibold mb-2">Alternative Timeline</h4>
          <p className="text-sm">Original Outcome: ${results.originalOutcome?.toLocaleString()}</p>
          <p className="text-sm">Alternative Outcome: ${results.alternativeOutcome?.toLocaleString()}</p>
          <p className="text-sm text-green-400 mt-2">
            Potential Impact: ${results.impact?.toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
};

// Recent Time Queries
const RecentTimeQueries = () => {
  const queries = [
    {
      query: "When did customer churn start increasing?",
      answer: "Churn pattern emerged on March 15th after pricing change",
      impact: "Identified $340K in preventable losses"
    },
    {
      query: "What if we had hired a sales team in Q1?",
      answer: "Revenue would be 23% higher with 87% confidence",
      impact: "$1.2M additional revenue opportunity"
    }
  ];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Recent Analyses</h3>
      <div className="space-y-4">
        {queries.map((q, i) => (
          <div key={i} className="p-4 bg-gray-800 rounded-lg">
            <p className="font-medium text-white mb-2">{q.query}</p>
            <p className="text-sm text-gray-400">{q.answer}</p>
            <p className="text-sm text-green-400 mt-1">{q.impact}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

// Integrations View
const IntegrationsView = () => {
  const [integrations, setIntegrations] = useState([]);

  useEffect(() => {
    fetch('/api/integrations')
      .then(res => res.json())
      .then(data => setIntegrations(data))
      .catch(console.error);
  }, []);

  const categories = {
    crm: { label: 'CRM', icon: UserCheck },
    communication: { label: 'Communication', icon: Mail },
    productivity: { label: 'Productivity', icon: Calendar },
    finance: { label: 'Finance', icon: DollarSign }
  };

  const groupedIntegrations = integrations.reduce((acc, integration) => {
    const category = integration.category || 'other';
    if (!acc[category]) acc[category] = [];
    acc[category].push(integration);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Integrations</h2>
      
      {Object.entries(groupedIntegrations).map(([category, items]) => (
        <div key={category} className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            {categories[category] && <categories[category].icon className="w-5 h-5" />}
            {categories[category]?.label || category}
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.map(integration => (
              <IntegrationCard key={integration.id} integration={integration} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// Integration Card
const IntegrationCard = ({ integration }) => {
  const statusColors = {
    connected: 'bg-green-900/20 border-green-800 text-green-400',
    disconnected: 'bg-gray-800 border-gray-700 text-gray-400',
    error: 'bg-red-900/20 border-red-800 text-red-400'
  };

  return (
    <div className={`p-4 rounded-lg border ${statusColors[integration.status]}`}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold">{integration.name}</h4>
        <div className={`w-2 h-2 rounded-full ${
          integration.status === 'connected' ? 'bg-green-500' :
          integration.status === 'error' ? 'bg-red-500' :
          'bg-gray-500'
        }`} />
      </div>
      
      <p className="text-sm text-gray-400 mb-3">{integration.description}</p>
      
      {integration.status === 'connected' ? (
        <div className="space-y-1 text-xs text-gray-500">
          <p>Last sync: {integration.lastSync}</p>
          <p>Items processed: {integration.itemsProcessed}</p>
        </div>
      ) : (
        <button className="w-full px-3 py-1 bg-blue-600 rounded text-sm hover:bg-blue-700">
          Connect
        </button>
      )}
    </div>
  );
};

// Settings View
const SettingsView = ({ user }) => {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Settings</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <AccountSettings user={user} />
          <PreferencesSettings />
          <SecuritySettings />
        </div>
        
        <div className="space-y-6">
          <BillingInfo user={user} />
          <UsageStats />
        </div>
      </div>
    </div>
  );
};

// Account Settings
const AccountSettings = ({ user }) => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Account Information</h3>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm text-gray-400">Name</label>
          <input
            type="text"
            value={user?.name || ''}
            className="w-full px-4 py-2 bg-gray-800 rounded-lg text-white mt-1"
            readOnly
          />
        </div>
        
        <div>
          <label className="text-sm text-gray-400">Email</label>
          <input
            type="email"
            value={user?.email || ''}
            className="w-full px-4 py-2 bg-gray-800 rounded-lg text-white mt-1"
            readOnly
          />
        </div>
        
        <div>
          <label className="text-sm text-gray-400">Company</label>
          <input
            type="text"
            value={user?.company || ''}
            className="w-full px-4 py-2 bg-gray-800 rounded-lg text-white mt-1"
            readOnly
          />
        </div>
      </div>
    </div>
  );
};

// Preferences Settings
const PreferencesSettings = () => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Preferences</h3>
      
      <div className="space-y-4">
        <label className="flex items-center justify-between">
          <span className="text-sm">Email notifications</span>
          <input type="checkbox" className="toggle" defaultChecked />
        </label>
        
        <label className="flex items-center justify-between">
          <span className="text-sm">Daily briefing</span>
          <input type="checkbox" className="toggle" defaultChecked />
        </label>
        
        <label className="flex items-center justify-between">
          <span className="text-sm">Auto-approve routine tasks</span>
          <input type="checkbox" className="toggle" />
        </label>
      </div>
    </div>
  );
};

// Security Settings
const SecuritySettings = () => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
        <Shield className="w-5 h-5" />
        Security
      </h3>
      
      <div className="space-y-4">
        <div className="p-4 bg-green-900/20 border border-green-800 rounded-lg">
          <p className="text-sm text-green-400">Zero-knowledge encryption active</p>
          <p className="text-xs text-gray-400 mt-1">Your data is fully encrypted</p>
        </div>
        
        <button className="w-full px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700">
          View Security Log
        </button>
        
        <button className="w-full px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700">
          Download Data Export
        </button>
      </div>
    </div>
  );
};

// Billing Info
const BillingInfo = ({ user }) => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Billing</h3>
      
      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-400">Current Plan</p>
          <p className="text-lg font-semibold">{user?.plan || 'SOVREN Proof'}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-400">Next Billing Date</p>
          <p className="text-lg">{user?.nextBilling || 'January 15, 2025'}</p>
        </div>
        
        <button className="w-full px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700">
          Manage Subscription
        </button>
      </div>
    </div>
  );
};

// Usage Stats
const UsageStats = () => {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">Usage This Month</h3>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>API Calls</span>
            <span>12,847 / ∞</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div className="bg-blue-600 h-2 rounded-full" style={{width: '45%'}} />
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Voice Minutes</span>
            <span>347 / ∞</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div className="bg-green-600 h-2 rounded-full" style={{width: '23%'}} />
          </div>
        </div>
        
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>Storage</span>
            <span>2.3 GB / ∞</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div className="bg-purple-600 h-2 rounded-full" style={{width: '15%'}} />
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper Functions
const getTimeOfDay = () => {
  const hour = new Date().getHours();
  if (hour < 12) return 'morning';
  if (hour < 17) return 'afternoon';
  return 'evening';
};

// Export main component
export default SOVRENDashboard;
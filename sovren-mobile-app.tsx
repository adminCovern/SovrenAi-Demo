import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Animated,
  Dimensions,
  StatusBar,
  Platform,
  ActivityIndicator,
  Alert
} from 'react-native';
import {
  Brain,
  Phone,
  TrendingUp,
  Shield,
  Users,
  Clock,
  Mic,
  MessageSquare,
  DollarSign,
  Activity
} from 'lucide-react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Haptics from 'expo-haptics';

const { width, height } = Dimensions.get('window');
const API_BASE = 'https://sovrenai.app/api';

// SOVREN Mobile App
const SOVRENMobileApp = () => {
  // State management
  const [user, setUser] = useState(null);
  const [sovrenScore, setSovrenScore] = useState(0);
  const [metrics, setMetrics] = useState({
    decisionsToday: 0,
    valueCreated: 0,
    activeOperations: 0,
    voiceCalls: 0
  });
  const [voiceActive, setVoiceActive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const slideAnim = useRef(new Animated.Value(height)).current;
  
  useEffect(() => {
    // Initialize app
    initializeApp();
    
    // Start animations
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 1000,
      useNativeDriver: true
    }).start();
    
    // Pulse animation for voice button
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 1000,
          useNativeDriver: true
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true
        })
      ])
    ).start();
  }, []);
  
  const initializeApp = async () => {
    try {
      // Check for stored user
      const storedUser = await AsyncStorage.getItem('sovren_user');
      if (storedUser) {
        setUser(JSON.parse(storedUser));
      }
      
      // Fetch initial data
      await fetchDashboardData();
      
      setLoading(false);
    } catch (error) {
      console.error('Initialization error:', error);
      setLoading(false);
    }
  };
  
  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE}/dashboard`, {
        headers: {
          'X-User-ID': user?.id || 'mobile_user',
          'X-ZKP': JSON.stringify({}) // Simplified for demo
        }
      });
      
      const data = await response.json();
      
      setSovrenScore(data.metrics.sovrenScore);
      setMetrics({
        decisionsToday: data.metrics.decisionsToday,
        valueCreated: data.metrics.valueCreated,
        activeOperations: data.metrics.activeOperations,
        voiceCalls: data.metrics.voiceCalls
      });
    } catch (error) {
      console.error('Dashboard fetch error:', error);
    }
  };
  
  const toggleVoice = async () => {
    // Haptic feedback
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    setVoiceActive(!voiceActive);
    
    if (!voiceActive) {
      // Start voice session
      Alert.alert(
        'Voice Activated',
        'SOVREN is listening. Speak your command.',
        [{ text: 'OK' }]
      );
    }
  };
  
  const renderDashboard = () => (
    <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
      {/* SOVREN Score */}
      <View style={styles.scoreCard}>
        <Text style={styles.scoreLabel}>SOVREN Score</Text>
        <Text style={styles.scoreValue}>{sovrenScore}</Text>
        <View style={styles.scoreBar}>
          <View 
            style={[
              styles.scoreBarFill,
              { width: `${(sovrenScore / 850) * 100}%` }
            ]}
          />
        </View>
      </View>
      
      {/* Metrics Grid */}
      <View style={styles.metricsGrid}>
        <MetricCard
          icon={Brain}
          label="Decisions Today"
          value={metrics.decisionsToday}
          color="#3B82F6"
        />
        <MetricCard
          icon={DollarSign}
          label="Value Created"
          value={`$${(metrics.valueCreated / 1000).toFixed(1)}K`}
          color="#10B981"
        />
        <MetricCard
          icon={Activity}
          label="Active Ops"
          value={metrics.activeOperations}
          color="#F59E0B"
        />
        <MetricCard
          icon={Phone}
          label="Voice Calls"
          value={metrics.voiceCalls}
          color="#8B5CF6"
        />
      </View>
      
      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <TouchableOpacity style={styles.actionButton}>
          <Users size={20} color="#fff" />
          <Text style={styles.actionText}>Shadow Board Meeting</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionButton}>
          <Clock size={20} color="#fff" />
          <Text style={styles.actionText}>Explore Futures</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionButton}>
          <Shield size={20} color="#fff" />
          <Text style={styles.actionText}>Security Status</Text>
        </TouchableOpacity>
      </View>
      
      {/* Recent Activity */}
      <View style={styles.activitySection}>
        <Text style={styles.sectionTitle}>Recent Activity</Text>
        <ActivityItem
          icon={Brain}
          text="Approved vendor contract - 23% savings"
          time="2 min ago"
        />
        <ActivityItem
          icon={Phone}
          text="Handled customer escalation successfully"
          time="15 min ago"
        />
        <ActivityItem
          icon={TrendingUp}
          text="Identified $47K optimization opportunity"
          time="1 hour ago"
        />
      </View>
    </ScrollView>
  );
  
  const renderInsights = () => (
    <View style={styles.content}>
      <Text style={styles.pageTitle}>Business Insights</Text>
      
      <View style={styles.insightCard}>
        <Text style={styles.insightTitle}>This Week's Performance</Text>
        <View style={styles.insightMetrics}>
          <View style={styles.insightItem}>
            <Text style={styles.insightValue}>+23%</Text>
            <Text style={styles.insightLabel}>Revenue Growth</Text>
          </View>
          <View style={styles.insightItem}>
            <Text style={styles.insightValue}>-15%</Text>
            <Text style={styles.insightLabel}>Cost Reduction</Text>
          </View>
          <View style={styles.insightItem}>
            <Text style={styles.insightValue}>94%</Text>
            <Text style={styles.insightLabel}>Efficiency</Text>
          </View>
        </View>
      </View>
      
      <View style={styles.recommendationCard}>
        <Text style={styles.recommendationTitle}>SOVREN Recommends</Text>
        <Text style={styles.recommendationText}>
          Based on current market conditions and your business trajectory, 
          I recommend expanding into the Southeast market. Time Machine analysis 
          shows 87% probability of success with projected 3.2x ROI.
        </Text>
        <TouchableOpacity style={styles.viewDetailsButton}>
          <Text style={styles.viewDetailsText}>View Full Analysis</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
  
  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#3B82F6" />
        <Text style={styles.loadingText}>Initializing SOVREN...</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <View style={styles.logoContainer}>
            <Brain size={28} color="#3B82F6" />
            <Text style={styles.headerTitle}>SOVREN AI</Text>
          </View>
          <TouchableOpacity style={styles.notificationButton}>
            <MessageSquare size={24} color="#fff" />
            <View style={styles.notificationBadge} />
          </TouchableOpacity>
        </View>
        <Text style={styles.headerSubtitle}>Your PhD-Level Chief of Staff</Text>
      </View>
      
      {/* Content */}
      <Animated.View style={[styles.mainContent, { opacity: fadeAnim }]}>
        {activeTab === 'dashboard' ? renderDashboard() : renderInsights()}
      </Animated.View>
      
      {/* Voice Button */}
      <Animated.View 
        style={[
          styles.voiceButtonContainer,
          { transform: [{ scale: voiceActive ? pulseAnim : 1 }] }
        ]}
      >
        <TouchableOpacity
          style={[styles.voiceButton, voiceActive && styles.voiceButtonActive]}
          onPress={toggleVoice}
          activeOpacity={0.8}
        >
          <Mic size={32} color="#fff" />
        </TouchableOpacity>
      </Animated.View>
      
      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => setActiveTab('dashboard')}
        >
          <Activity 
            size={24} 
            color={activeTab === 'dashboard' ? '#3B82F6' : '#6B7280'}
          />
          <Text style={[
            styles.navText,
            activeTab === 'dashboard' && styles.navTextActive
          ]}>Dashboard</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => setActiveTab('insights')}
        >
          <TrendingUp 
            size={24} 
            color={activeTab === 'insights' ? '#3B82F6' : '#6B7280'}
          />
          <Text style={[
            styles.navText,
            activeTab === 'insights' && styles.navTextActive
          ]}>Insights</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

// Metric Card Component
const MetricCard = ({ icon: Icon, label, value, color }) => (
  <View style={[styles.metricCard, { borderColor: color + '30' }]}>
    <Icon size={24} color={color} />
    <Text style={styles.metricValue}>{value}</Text>
    <Text style={styles.metricLabel}>{label}</Text>
  </View>
);

// Activity Item Component
const ActivityItem = ({ icon: Icon, text, time }) => (
  <View style={styles.activityItem}>
    <View style={styles.activityIcon}>
      <Icon size={16} color="#3B82F6" />
    </View>
    <View style={styles.activityContent}>
      <Text style={styles.activityText}>{text}</Text>
      <Text style={styles.activityTime}>{time}</Text>
    </View>
  </View>
);

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A'
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0F172A'
  },
  loadingText: {
    color: '#fff',
    marginTop: 16,
    fontSize: 16
  },
  header: {
    paddingTop: Platform.OS === 'ios' ? 50 : 30,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: '#1E293B',
    borderBottomWidth: 1,
    borderBottomColor: '#334155'
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center'
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginLeft: 10
  },
  headerSubtitle: {
    color: '#94A3B8',
    fontSize: 14,
    marginTop: 5
  },
  notificationButton: {
    position: 'relative'
  },
  notificationBadge: {
    position: 'absolute',
    top: -2,
    right: -2,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#EF4444'
  },
  mainContent: {
    flex: 1
  },
  content: {
    flex: 1,
    paddingHorizontal: 20
  },
  scoreCard: {
    backgroundColor: '#1E293B',
    padding: 20,
    borderRadius: 16,
    marginTop: 20,
    alignItems: 'center'
  },
  scoreLabel: {
    color: '#94A3B8',
    fontSize: 14
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#3B82F6',
    marginVertical: 10
  },
  scoreBar: {
    width: '100%',
    height: 8,
    backgroundColor: '#334155',
    borderRadius: 4,
    overflow: 'hidden'
  },
  scoreBarFill: {
    height: '100%',
    backgroundColor: '#3B82F6',
    borderRadius: 4
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginTop: 20
  },
  metricCard: {
    width: '48%',
    backgroundColor: '#1E293B',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    alignItems: 'center',
    borderWidth: 1
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8
  },
  metricLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginTop: 4
  },
  quickActions: {
    marginTop: 20
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12
  },
  actionText: {
    color: '#fff',
    fontSize: 16,
    marginLeft: 12
  },
  activitySection: {
    marginTop: 20,
    marginBottom: 100
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12
  },
  activityIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#3B82F630',
    justifyContent: 'center',
    alignItems: 'center'
  },
  activityContent: {
    flex: 1,
    marginLeft: 12
  },
  activityText: {
    color: '#fff',
    fontSize: 14
  },
  activityTime: {
    color: '#64748B',
    fontSize: 12,
    marginTop: 2
  },
  voiceButtonContainer: {
    position: 'absolute',
    bottom: 100,
    alignSelf: 'center'
  },
  voiceButton: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#3B82F6',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8
  },
  voiceButtonActive: {
    backgroundColor: '#EF4444'
  },
  bottomNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    paddingVertical: 12,
    paddingBottom: Platform.OS === 'ios' ? 24 : 12,
    borderTopWidth: 1,
    borderTopColor: '#334155'
  },
  navItem: {
    alignItems: 'center'
  },
  navText: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 4
  },
  navTextActive: {
    color: '#3B82F6'
  },
  pageTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 20,
    marginBottom: 20
  },
  insightCard: {
    backgroundColor: '#1E293B',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16
  },
  insightTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16
  },
  insightMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  insightItem: {
    alignItems: 'center'
  },
  insightValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#10B981'
  },
  insightLabel: {
    fontSize: 12,
    color: '#94A3B8',
    marginTop: 4
  },
  recommendationCard: {
    backgroundColor: '#1E293B',
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#3B82F630'
  },
  recommendationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#3B82F6',
    marginBottom: 12
  },
  recommendationText: {
    color: '#E2E8F0',
    fontSize: 14,
    lineHeight: 22
  },
  viewDetailsButton: {
    marginTop: 16,
    backgroundColor: '#3B82F6',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center'
  },
  viewDetailsText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600'
  }
});

export default SOVRENMobileApp;
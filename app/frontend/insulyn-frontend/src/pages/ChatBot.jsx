import React, { useState, useRef, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Avatar,
  Chip,
  Grid,
  CircularProgress,
  IconButton,
  Alert,
  Snackbar,
  LinearProgress
} from '@mui/material';
import { Send, SmartToy, Person, Delete, Refresh, Warning } from '@mui/icons-material';

const ChatBot = ({ language = 'english' }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [retryCount, setRetryCount] = useState(0);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000';

  // Initialize conversation
  useEffect(() => {
    initializeConversation();
    checkBackendConnection();
  }, [language]);

  const checkBackendConnection = async () => {
    try {
      console.log('🔍 Checking backend connection...');
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 5000
      });
      
      if (response.ok) {
        const data = await response.json();
        setBackendStatus('connected');
        console.log('✅ Backend connected successfully:', data);
        setError('');
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('❌ Backend connection failed:', error);
      setBackendStatus('disconnected');
      setError(`Backend connection failed: ${error.message}. Please ensure the server is running on port 8000.`);
    }
  };

  const initializeConversation = () => {
    const newConversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setConversationId(newConversationId);
    
    setMessages([{
      text: getWelcomeMessage(language),
      sender: 'bot',
      timestamp: new Date(),
      id: Date.now(),
      isSystem: true
    }]);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getWelcomeMessage = (lang) => {
    const welcomeMessages = {
      english: "🔬 **AI Diabetes Assistant - Live LLM Mode**\n\nI'm connected to the Groq LLM and ready to provide intelligent, real-time responses about diabetes prevention and management.\n\n**I can help with:**\n• Detailed medical explanations\n• Personalized risk assessments\n• Latest research insights\n• Comprehensive treatment options\n• Lifestyle recommendations\n\nAsk me anything about diabetes!",
      swahili: "🔬 **Msaidizi wa Kisukari wa AK - Hali ya LLM ya Moja kwa Moja**\n\nNimeunganishwa na Groq LLM na niko tayari kutoa majibu ya kisasa na ya wakati halisi kuhusu kuzuia na kudhibiti kisukari.\n\n**Ninaweza kusaidia kwa:**\n• Maelezo ya kina ya kitabibu\n• Tathmini binafsi za hatari\n• Ufahamu wa utafiti wa hivi karibuni\n• Chaguo kamili za matibabu\n• Mapendekezo ya mtindo wa maisha\n\nNiulize chochote kuhusu kisukari!",
      sheng: "🔬 **AI Diabetes Assistant - Live LLM Mode**\n\nNiko connected na Groq LLM na niko ready kutoa smart, real-time answers about diabetes.\n\n**Naeza help na:**\n• Detailed medical info\n• Personal risk assessment\n• Latest research\n• Treatment options\n• Lifestyle advice\n\nAsk me anything about diabetes!"
    };
    return welcomeMessages[lang] || welcomeMessages.english;
  };

  // Force LLM response - no fallbacks
  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { 
      text: input, 
      sender: 'user', 
      timestamp: new Date(),
      id: Date.now()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError('');

    try {
      // Enhanced payload for better LLM context
      const payload = {
        message: input,
        language: language,
        conversation_context: "diabetes_medical_advice",
        conversation_id: conversationId,
        require_llm: true, // Force LLM usage
        timestamp: new Date().toISOString(),
        message_type: "user_query"
      };

      console.log('🚀 Sending to LLM backend:', {
        url: `${API_BASE_URL}/api/v1/chat`,
        payload: payload
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      console.log('📨 LLM Response status:', response.status);
      
      if (!response.ok) {
        let errorMessage = `LLM Service Error: HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
          console.log('❌ LLM backend error details:', errorData);
        } catch (e) {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
          console.log('❌ LLM backend error text:', errorText);
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('✅ LLM Response received:', data);

      // Validate LLM response
      if (!data || (!data.response && !data.message)) {
        throw new Error('LLM returned empty or invalid response');
      }

      const llmResponse = data.response || data.message || data;
      
      setMessages(prev => [...prev, { 
        text: llmResponse,
        sender: 'bot',
        timestamp: new Date(),
        id: Date.now() + 1,
        isLLMResponse: true,
        suggestions: data.suggestions || generateLLMSuggestions(input, llmResponse)
      }]);

      setBackendStatus('connected');
      setRetryCount(0); // Reset retry count on success

    } catch (error) {
      console.error('💥 LLM Communication Error:', error);
      
      const errorMessage = error.name === 'AbortError' 
        ? 'LLM request timeout - server is taking too long to respond'
        : `LLM Service Error: ${error.message}`;

      setError(errorMessage);
      setBackendStatus('disconnected');
      
      // Increment retry count
      const newRetryCount = retryCount + 1;
      setRetryCount(newRetryCount);

      // Show error message to user
      setMessages(prev => [...prev, { 
        text: `🚨 **LLM Service Unavailable**\n\nI'm unable to connect to the AI service right now.\n\n**Error:** ${errorMessage}\n**Retry Attempt:** ${newRetryCount}\n\nPlease:\n1. Check if the backend server is running\n2. Ensure port 8000 is accessible\n3. Try again in a moment\n4. Contact support if this persists`,
        sender: 'bot',
        timestamp: new Date(),
        id: Date.now() + 1,
        isError: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Generate intelligent follow-up suggestions based on LLM response
  const generateLLMSuggestions = (userInput, llmResponse) => {
    const inputLower = userInput.toLowerCase();
    const responseLower = llmResponse.toLowerCase();

    // Medical topic detection
    if (responseLower.includes('blood sugar') || responseLower.includes('glucose')) {
      return [
        "What are normal blood sugar ranges?",
        "How often should I check my glucose?",
        "Symptoms of hyperglycemia",
        "Best foods for blood sugar control"
      ];
    }

    if (responseLower.includes('risk') || responseLower.includes('prevent')) {
      return [
        "Specific lifestyle changes for prevention",
        "Genetic risk factors explained",
        "Early warning signs to watch for",
        "Medical screening recommendations"
      ];
    }

    if (responseLower.includes('treatment') || responseLower.includes('medication')) {
      return [
        "Latest diabetes medications",
        "Natural treatment options",
        "Side effects management",
        "Combination therapies"
      ];
    }

    if (responseLower.includes('diet') || responseLower.includes('nutrition')) {
      return [
        "Sample meal plans",
        "Foods to completely avoid",
        "Carbohydrate counting guide",
        "Glycemic index explained"
      ];
    }

    if (responseLower.includes('exercise') || responseLower.includes('workout')) {
      return [
        "Best exercises for diabetics",
        "Exercise timing and frequency",
        "Managing blood sugar during exercise",
        "Safety precautions"
      ];
    }

    // Default intelligent suggestions
    return [
      "Explain this in more detail",
      "What are the latest research findings?",
      "How does this apply to different age groups?",
      "Are there any new treatments available?"
    ];
  };

  const handleQuickQuestion = (question) => {
    setInput(question);
    // Auto-send after a brief delay
    setTimeout(() => {
      if (backendStatus === 'connected') {
        handleSend();
      } else {
        setError('Please wait for backend connection before sending questions');
      }
    }, 100);
  };

  const clearConversation = () => {
    initializeConversation();
    setError('');
    setRetryCount(0);
  };

  const retryBackendConnection = async () => {
    setError('Retrying backend connection...');
    await checkBackendConnection();
    if (backendStatus === 'connected') {
      setError('');
    }
  };

  const quickQuestions = [
    "Explain type 2 diabetes pathophysiology",
    "Latest advancements in diabetes treatment",
    "Comprehensive diabetes risk assessment",
    "Evidence-based dietary recommendations",
    "Exercise physiology for diabetes management",
    "Pharmacological treatment options",
    "Diabetes complications and prevention",
    "Mental health aspects of diabetes care"
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 3, position: 'relative' }}>
        {/* Connection Status Bar */}
        {backendStatus !== 'connected' && (
          <LinearProgress 
            color="warning" 
            sx={{ 
              position: 'absolute', 
              top: 0, 
              left: 0, 
              right: 0,
              height: 3
            }} 
          />
        )}

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              🧠 AI Diabetes Specialist
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: backendStatus === 'connected' ? '#4caf50' : '#ff9800',
                  animation: backendStatus === 'connected' ? 'pulse 2s infinite' : 'none'
                }}
              />
              <Typography variant="caption" color={backendStatus === 'connected' ? 'success.main' : 'warning.main'}>
                {backendStatus === 'connected' ? '🟢 LLM Connected' : '🟡 Connecting to LLM...'}
              </Typography>
              {retryCount > 0 && (
                <Typography variant="caption" color="text.secondary">
                  (Retry {retryCount})
                </Typography>
              )}
            </Box>
          </Box>
          <Box>
            <IconButton 
              onClick={retryBackendConnection} 
              title="Reconnect to LLM"
              size="small"
              color={backendStatus === 'connected' ? 'success' : 'warning'}
            >
              <Refresh />
            </IconButton>
            <IconButton 
              onClick={clearConversation} 
              title="Start new conversation"
              size="small"
            >
              <Delete />
            </IconButton>
          </Box>
        </Box>

        {/* System Status Alert */}
        {backendStatus !== 'connected' && (
          <Alert 
            severity="warning" 
            icon={<Warning />}
            sx={{ mb: 2 }}
            action={
              <Button 
                color="inherit" 
                size="small" 
                onClick={retryBackendConnection}
              >
                RETRY
              </Button>
            }
          >
            <Typography variant="body2">
              <strong>LLM Service Offline</strong> - Real-time AI responses unavailable. 
              {retryCount > 0 && ` Failed ${retryCount} time(s).`}
            </Typography>
          </Alert>
        )}

        {/* Quick Questions */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>🧪 Medical Questions:</Typography>
          <Grid container spacing={1}>
            {quickQuestions.map((question, index) => (
              <Grid item key={index}>
                <Chip
                  label={question}
                  onClick={() => handleQuickQuestion(question)}
                  variant="outlined"
                  clickable
                  size="small"
                  disabled={backendStatus !== 'connected' || loading}
                  color={backendStatus === 'connected' ? 'primary' : 'default'}
                />
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Chat Messages */}
        <Card sx={{ 
          height: '500px', 
          overflow: 'auto', 
          mb: 2, 
          border: backendStatus === 'connected' ? '2px solid #4caf50' : '2px solid #ff9800',
          position: 'relative'
        }}>
          {backendStatus !== 'connected' && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(255, 152, 0, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1
              }}
            >
              <Typography variant="h6" color="warning.main" sx={{ opacity: 0.7 }}>
                ⚠️ LLM Service Offline - Connect to enable AI responses
              </Typography>
            </Box>
          )}
          
          <CardContent>
            <List>
              {messages.map((message) => (
                <ListItem key={message.id} alignItems="flex-start" sx={{ mb: 2 }}>
                  <Avatar sx={{ 
                    mr: 2, 
                    bgcolor: message.sender === 'user' ? 'primary.main' : 
                            message.isError ? 'error.main' : 
                            message.isSystem ? 'info.main' : 'success.main'
                  }}>
                    {message.sender === 'user' ? <Person /> : <SmartToy />}
                  </Avatar>
                  <ListItemText
                    primary={
                      <Box sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                        {message.text}
                        {message.suggestions && message.suggestions.length > 0 && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="caption" fontWeight="bold" color="primary">
                              🔍 Follow-up Questions:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                              {message.suggestions.map((suggestion, idx) => (
                                <Chip
                                  key={idx}
                                  label={suggestion}
                                  size="small"
                                  variant="outlined"
                                  onClick={() => handleQuickQuestion(suggestion)}
                                  disabled={backendStatus !== 'connected'}
                                  color="primary"
                                />
                              ))}
                            </Box>
                          </Box>
                        )}
                      </Box>
                    }
                    secondary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, alignItems: 'center' }}>
                        <Box>
                          <Typography variant="caption">
                            {message.sender === 'user' ? '👤 You' : 
                             message.isError ? '🚨 System' : 
                             message.isSystem ? 'ℹ️ System' : '🤖 AI Specialist'}
                          </Typography>
                          {message.isLLMResponse && (
                            <Typography variant="caption" color="success.main" sx={{ ml: 1 }}>
                              • Live LLM
                            </Typography>
                          )}
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          {message.timestamp.toLocaleTimeString()}
                        </Typography>
                      </Box>
                    }
                    sx={{
                      backgroundColor: message.sender === 'user' ? '#f0f7ff' : 
                                     message.isError ? '#ffebee' : 
                                     message.isSystem ? '#e3f2fd' : '#e8f5e8',
                      p: 3,
                      borderRadius: 3,
                      border: message.isLLMResponse ? '2px solid #4caf50' : 
                             message.isError ? '2px solid #f44336' : '1px solid #e0e0e0'
                    }}
                  />
                </ListItem>
              ))}
              {loading && (
                <ListItem>
                  <Avatar sx={{ mr: 2, bgcolor: 'success.main' }}>
                    <SmartToy />
                  </Avatar>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                    <CircularProgress size={24} />
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        🧠 Processing with LLM...
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Analyzing your medical query with AI
                      </Typography>
                    </Box>
                  </Box>
                </ListItem>
              )}
              <div ref={messagesEndRef} />
            </List>
          </CardContent>
        </Card>

        {/* Input Area */}
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder={
              backendStatus === 'connected' 
                ? "Ask detailed medical questions about diabetes... (LLM Connected 🟢)"
                : "LLM Service Offline - Please wait for connection... (Connecting 🟡)"
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && backendStatus === 'connected' && handleSend()}
            disabled={loading || backendStatus !== 'connected'}
            multiline
            maxRows={4}
            sx={{
              '& .MuiOutlinedInput-root': {
                backgroundColor: backendStatus === 'connected' ? '#f8fff8' : '#fffbf0',
                borderColor: backendStatus === 'connected' ? '#4caf50' : '#ff9800'
              }
            }}
          />
          <Button
            variant="contained"
            endIcon={<Send />}
            onClick={handleSend}
            disabled={loading || !input.trim() || backendStatus !== 'connected'}
            sx={{ 
              minWidth: '120px', 
              height: '56px',
              backgroundColor: backendStatus === 'connected' ? '#4caf50' : '#ff9800',
              '&:hover': {
                backgroundColor: backendStatus === 'connected' ? '#45a049' : '#f57c00'
              }
            }}
          >
            {loading ? <CircularProgress size={24} /> : 'Ask LLM'}
          </Button>
        </Box>

        {/* Status Information */}
        <Box sx={{ 
          mt: 2, 
          p: 2, 
          backgroundColor: backendStatus === 'connected' ? '#e8f5e8' : '#fff3e0',
          borderRadius: 2,
          border: `1px solid ${backendStatus === 'connected' ? '#4caf50' : '#ff9800'}`
        }}>
          <Typography variant="caption" color={backendStatus === 'connected' ? 'success.main' : 'warning.main'}>
            <strong>
              {backendStatus === 'connected' 
                ? '✅ LIVE MODE: Connected to Groq LLM - All responses are AI-generated in real-time'
                : '⚠️ OFFLINE MODE: LLM service unavailable - Please check backend server connection'
              }
            </strong>
          </Typography>
        </Box>
      </Paper>

      {/* Error Snackbar */}
      <Snackbar
        open={!!error}
        autoHideDuration={8000}
        onClose={() => setError('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setError('')} 
          severity="error" 
          sx={{ width: '100%' }}
          action={
            <Button color="inherit" size="small" onClick={retryBackendConnection}>
              RETRY
            </Button>
          }
        >
          {error}
        </Alert>
      </Snackbar>

      <style jsx>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </Container>
  );
};

export default ChatBot;
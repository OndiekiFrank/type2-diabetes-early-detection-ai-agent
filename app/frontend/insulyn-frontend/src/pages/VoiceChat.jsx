import React, { useState, useRef } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  IconButton,
  Fade,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Mic,
  MicOff,
  SmartToy,
  Translate,
  Keyboard
} from '@mui/icons-material';

const VoiceChatAssistant = ({ language = 'english' }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [textInput, setTextInput] = useState('');
  const [showTextDialog, setShowTextDialog] = useState(false);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);

  const translations = {
    english: {
      title: '🎤 Voice Chat Assistant',
      subtitle: 'Speak naturally about diabetes concerns',
      instructions: 'How to use: Click Start, speak for 5-15 seconds about diabetes, click Stop. The AI will understand and respond.',
      startRecording: 'Start Recording',
      stopRecording: 'Stop Recording',
      processing: 'Processing your voice...',
      yourQuestion: 'Your Question',
      aiResponse: 'AI Response',
      confidence: 'Confidence',
      tryAgain: 'Try Again',
      typeInstead: 'Type Instead',
      enterQuestion: 'Enter your diabetes question',
      submitText: 'Submit Text',
      cancel: 'Cancel',
      madeWith: 'Made with ❤️ for better health',
      disclaimer: 'This is not a substitute for professional medical advice. Always consult healthcare professionals.'
    },
    swahili: {
      title: '🎤 Msaidizi wa Sauti',
      subtitle: 'Zungumza kawaida kuhusu wasiwasi wa kisukari',
      instructions: 'Jinsi ya kutumia: Bofya Anza, zungumza kwa sekunde 5-15 kuhusu kisukari, bofya Acha. AI itaelewa na kujibu.',
      startRecording: 'Anza Kurekodi',
      stopRecording: 'Acha Kurekodi',
      processing: 'Inachakata sauti yako...',
      yourQuestion: 'Swali Lako',
      aiResponse: 'Majibu ya AI',
      confidence: 'Uthabiti',
      tryAgain: 'Jaribu Tena',
      typeInstead: 'Andika Badala Yake',
      enterQuestion: 'Andika swali lako kuhusu kisukari',
      submitText: 'Wasilisha Maandishi',
      cancel: 'Ghairi',
      madeWith: 'Imetengenezwa kwa ❤️ kwa afya bora',
      disclaimer: 'Hii sio mbadala wa ushauri wa kimatibabu. Wasiliana na wataalamu wa afya kila wakati.'
    }
  };

  const t = translations[language] || translations.english;

  const startRecording = async () => {
    try {
      setError('');
      setResult(null);
      audioChunks.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      });
      
      mediaRecorder.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorder.current.onstop = processRecording;
      mediaRecorder.current.start(1000); // Collect data every second
      setIsRecording(true);

    } catch (err) {
      setError('Microphone access denied. Please allow microphone permissions.');
      console.error('Recording error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const processRecording = async () => {
    if (audioChunks.current.length === 0) {
      setError('No audio recorded. Please try again.');
      return;
    }

    setIsProcessing(true);
    
    try {
      const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm;codecs=opus' });
      
      // For demo purposes, we'll use the test endpoint with mock transcription
      // In production, you would send the actual audio
      const mockQuestions = {
        english: "I'm concerned about diabetes risk factors in my family",
        swahili: "Nina wasiwasi kuhusu mambo yanayochangia hatari ya kisukari katika familia yangu",
        spanish: "Estoy preocupado por los factores de riesgo de diabetes en mi familia",
        french: "Je suis préoccupé par les facteurs de risque de diabète dans ma famille"
      };
      
      const testText = mockQuestions[language] || mockQuestions.english;
      
      // Use test endpoint with mock text
      const response = await fetch('/api/v1/voice-chat/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          text: testText,
          language: language,
          user_id: 'voice-user'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Voice processing failed');
      }

      const resultData = await response.json();
      setResult(resultData);
      
    } catch (err) {
      setError(err.message || 'Failed to process voice. Please try again.');
      console.error('Voice processing error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTextSubmit = async () => {
    if (!textInput.trim()) {
      setError('Please enter a question');
      return;
    }

    setIsProcessing(true);
    setShowTextDialog(false);
    
    try {
      const response = await fetch('/api/v1/voice-chat/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          text: textInput,
          language: language,
          user_id: 'text-user'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Text processing failed');
      }

      const resultData = await response.json();
      setResult(resultData);
      setTextInput('');
      
    } catch (err) {
      setError(err.message || 'Failed to process text. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTryAgain = () => {
    setResult(null);
    setError('');
    audioChunks.current = [];
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={0} sx={{ p: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <Typography 
          variant="h3" 
          component="h1" 
          gutterBottom 
          align="center"
          sx={{ 
            color: 'white',
            fontWeight: 'bold',
            textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
          }}
        >
          {t.title}
        </Typography>
        
        <Typography 
          variant="h6" 
          align="center" 
          sx={{ color: 'white', mb: 2, opacity: 0.9 }}
        >
          {t.subtitle}
        </Typography>

        <Typography 
          variant="body1" 
          align="center" 
          sx={{ color: 'white', mb: 4, opacity: 0.8 }}
        >
          {t.instructions}
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper elevation={6} sx={{ p: 4, minHeight: '300px' }}>
          {/* Recording Interface */}
          {!result && !isProcessing && (
            <Box textAlign="center" py={4}>
              <IconButton
                sx={{
                  width: 100,
                  height: 100,
                  bgcolor: isRecording ? 'error.main' : 'primary.main',
                  color: 'white',
                  '&:hover': {
                    bgcolor: isRecording ? 'error.dark' : 'primary.dark',
                  },
                  mb: 3
                }}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
              >
                {isRecording ? <MicOff sx={{ fontSize: 40 }} /> : <Mic sx={{ fontSize: 40 }} />}
              </IconButton>
              
              <Typography variant="h6" gutterBottom>
                {isRecording ? t.stopRecording : t.startRecording}
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                {isRecording ? 'Speak now about diabetes concerns...' : 'Click the microphone to start'}
              </Typography>

              {isRecording && (
                <Box sx={{ mt: 2 }}>
                  <Chip 
                    icon={<Mic />} 
                    label="Listening..." 
                    color="primary" 
                    variant="outlined" 
                  />
                </Box>
              )}

              <Box sx={{ mt: 3 }}>
                <Button
                  variant="outlined"
                  startIcon={<Keyboard />}
                  onClick={() => setShowTextDialog(true)}
                >
                  {t.typeInstead}
                </Button>
              </Box>
            </Box>
          )}

          {/* Processing State */}
          {isProcessing && (
            <Box textAlign="center" py={4}>
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>
                {t.processing}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Analyzing your message...
              </Typography>
            </Box>
          )}

          {/* Results */}
          {result && (
            <Fade in={true} timeout={1000}>
              <Box>
                <Box display="flex" alignItems="center" mb={3}>
                  <SmartToy color="primary" sx={{ mr: 2 }} />
                  <Typography variant="h5">
                    Analysis Complete
                  </Typography>
                </Box>

                {/* User's Question */}
                <Card variant="outlined" sx={{ mb: 3 }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Mic color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">{t.yourQuestion}</Typography>
                    </Box>
                    <Typography variant="body1" sx={{ fontStyle: 'italic' }}>
                      "{result.text_input}"
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      <Chip 
                        label={`${t.confidence}: ${(result.confidence * 100).toFixed(0)}%`}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                  </CardContent>
                </Card>

                {/* AI Response */}
                <Card elevation={2}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <SmartToy color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">{t.aiResponse}</Typography>
                    </Box>
                    <Typography variant="body1" paragraph>
                      {result.ai_response}
                    </Typography>
                    <Box display="flex" gap={1} flexWrap="wrap">
                      <Chip 
                        icon={<Translate />}
                        label={result.language}
                        size="small"
                        variant="outlined"
                      />
                      <Chip 
                        label={new Date(result.timestamp).toLocaleTimeString()}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  </CardContent>
                </Card>

                <Box textAlign="center" mt={3}>
                  <Button 
                    variant="outlined" 
                    onClick={handleTryAgain}
                    startIcon={<Mic />}
                  >
                    {t.tryAgain}
                  </Button>
                </Box>
              </Box>
            </Fade>
          )}
        </Paper>

        {/* Text Input Dialog */}
        <Dialog open={showTextDialog} onClose={() => setShowTextDialog(false)} maxWidth="sm" fullWidth>
          <DialogTitle>{t.typeInstead}</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label={t.enterQuestion}
              fullWidth
              variant="outlined"
              multiline
              rows={4}
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              sx={{ mt: 2 }}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowTextDialog(false)}>{t.cancel}</Button>
            <Button onClick={handleTextSubmit} variant="contained">
              {t.submitText}
            </Button>
          </DialogActions>
        </Dialog>

        <Box textAlign="center" mt={4}>
          <Typography variant="body2" sx={{ color: 'white', opacity: 0.8 }}>
            {t.madeWith}
          </Typography>
          <Typography variant="caption" sx={{ color: 'white', opacity: 0.6 }}>
            {t.disclaimer}
          </Typography>
        </Box>

        <Box textAlign="center" mt={2}>
          <Typography variant="h6" sx={{ color: 'white', opacity: 0.9 }}>
            Insulyn AI
          </Typography>
          <Typography variant="body2" sx={{ color: 'white', opacity: 0.7 }}>
            Empowering Health Through AI
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default VoiceChatAssistant;
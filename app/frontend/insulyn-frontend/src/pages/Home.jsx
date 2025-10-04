import React from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
} from '@mui/material';
import {
  MedicalServices,
  Chat,
  Restaurant,
  RecordVoiceOver,
  Warning,
  HealthAndSafety,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Home = ({ language }) => {
  const navigate = useNavigate();

  const translations = {
    english: {
      welcome: 'Welcome to Insulyn AI',
      tagline: 'Your AI-powered diabetes prevention and management companion',
      features: 'Features',
      testTitle: 'Diabetes Risk Assessment',
      testDesc: 'Get your personalized diabetes risk prediction using advanced AI analysis',
      chatTitle: 'AI Health Assistant',
      chatDesc: 'Chat with our intelligent AI about diabetes prevention and management',
      voiceTitle: 'Voice Chat Assistant',
      voiceDesc: 'Talk naturally to our AI assistant in your preferred language',
      dietTitle: 'Personalized Diet Plans',
      dietDesc: 'Get customized meal plans tailored for diabetes prevention and management',
      emergencyTitle: 'Symptom Checker',
      emergencyDesc: 'Assess emergency symptoms and get immediate guidance and recommendations',
      startTest: 'Start Assessment',
      startChat: 'Start Chat',
      startVoice: 'Start Voice Chat',
      getDiet: 'Get Diet Plan',
      checkSymptoms: 'Check Symptoms',
    },
    swahili: {
      welcome: 'Karibu kwenye Insulyn AI',
      tagline: 'Msaidizi wako wa kuzuia na kudhibiti kisukari unaotumia AI',
      features: 'Vipengele',
      testTitle: 'Tathmini ya Hatari ya Kisukari',
      testDesc: 'Pata utabiri wa hatari yako ya kisukari kwa kutumia uchambuzi wa kisasa wa AI',
      chatTitle: 'Msaidizi wa Afya wa AI',
      chatDesc: 'Zungumza na AI yetu yenye akili kuhusu kuzuia na kudhibiti kisukari',
      voiceTitle: 'Msaidizi wa Mazungumzo ya Sauti',
      voiceDesc: 'Zungumza kawaida na msaidizi wetu wa AI kwa lugha unayopendelea',
      dietTitle: 'Mipango Binafsi ya Lishe',
      dietDesc: 'Pata mipango ya vyakula iliyobinafsishwa kwa kuzuia na kudhibiti kisukari',
      emergencyTitle: 'Kukagua Dalili',
      emergencyDesc: 'Kagua dalili za dharura na upate mwongozo wa haraka na mapendekezo',
      startTest: 'Anza Tathmini',
      startChat: 'Anza Mazungumzo',
      startVoice: 'Anza Mazungumzo ya Sauti',
      getDiet: 'Pata Mpango wa Lishe',
      checkSymptoms: 'Kagua Dalili',
    },
    sheng: {
      welcome: 'Welcome to Insulyn AI',
      tagline: 'Your AI-powered diabetes prevention and management companion',
      features: 'Features',
      testTitle: 'Diabetes Risk Assessment',
      testDesc: 'Get your personalized diabetes risk prediction using advanced AI analysis',
      chatTitle: 'AI Health Assistant',
      chatDesc: 'Chat with our intelligent AI about diabetes prevention and management',
      voiceTitle: 'Voice Chat Assistant',
      voiceDesc: 'Talk naturally to our AI assistant in your preferred language',
      dietTitle: 'Personalized Diet Plans',
      dietDesc: 'Get customized meal plans tailored for diabetes prevention and management',
      emergencyTitle: 'Symptom Checker',
      emergencyDesc: 'Assess emergency symptoms and get immediate guidance and recommendations',
      startTest: 'Start Assessment',
      startChat: 'Start Chat',
      startVoice: 'Start Voice Chat',
      getDiet: 'Get Diet Plan',
      checkSymptoms: 'Check Symptoms',
    }
  };

  const t = translations[language] || translations.english;

  const features = [
    {
      icon: <MedicalServices sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t.testTitle,
      description: t.testDesc,
      action: () => navigate('/test'),
      buttonText: t.startTest,
      color: 'primary'
    },
    {
      icon: <Chat sx={{ fontSize: 40, color: 'success.main' }} />,
      title: t.chatTitle,
      description: t.chatDesc,
      action: () => navigate('/chat'),
      buttonText: t.startChat,
      color: 'success'
    },
    {
      icon: <RecordVoiceOver sx={{ fontSize: 40, color: 'info.main' }} />,
      title: t.voiceTitle,
      description: t.voiceDesc,
      action: () => navigate('/voice-chat'),
      buttonText: t.startVoice,
      color: 'info'
    },
    {
      icon: <Restaurant sx={{ fontSize: 40, color: 'warning.main' }} />,
      title: t.dietTitle,
      description: t.dietDesc,
      action: () => navigate('/diet-plan'),
      buttonText: t.getDiet,
      color: 'warning'
    },
    {
      icon: <Warning sx={{ fontSize: 40, color: 'error.main' }} />,
      title: t.emergencyTitle,
      description: t.emergencyDesc,
      action: () => navigate('/emergency'),
      buttonText: t.checkSymptoms,
      color: 'error'
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box textAlign="center" mb={6}>
        <HealthAndSafety sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
        <Typography variant="h3" component="h1" gutterBottom color="primary">
          {t.welcome}
        </Typography>
        <Typography variant="h5" component="h2" color="text.secondary" gutterBottom>
          {t.tagline}
        </Typography>
      </Box>

      <Typography variant="h4" component="h2" gutterBottom textAlign="center" mb={4}>
        {t.features}
      </Typography>

      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'transform 0.2s, box-shadow 0.2s',
                border: `2px solid`,
                borderColor: `${feature.color}.light`,
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 8,
                  borderColor: `${feature.color}.main`,
                }
              }}
            >
              <CardContent sx={{ 
                flexGrow: 1, 
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between'
              }}>
                <Box>
                  <Box mb={2}>
                    {feature.icon}
                  </Box>
                  <Typography variant="h5" component="h3" gutterBottom>
                    {feature.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" paragraph>
                    {feature.description}
                  </Typography>
                </Box>
                <Button 
                  variant="contained" 
                  color={feature.color}
                  onClick={feature.action}
                  fullWidth
                  size="large"
                  sx={{ 
                    mt: 2,
                    fontWeight: 'bold'
                  }}
                >
                  {feature.buttonText}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Additional Info Section */}
      <Box sx={{ mt: 8, p: 4, bgcolor: 'background.default', borderRadius: 2 }}>
        <Typography variant="h5" gutterBottom textAlign="center" color="primary">
          🚀 Powered by Advanced AI Technology
        </Typography>
        <Typography variant="body1" textAlign="center" color="text.secondary">
          Our platform combines machine learning models with real-time AI assistants 
          to provide comprehensive diabetes care and prevention strategies.
        </Typography>
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4} textAlign="center">
            <Typography variant="h6" color="success.main">
              🤖 Groq LLM Integration
            </Typography>
            <Typography variant="body2">
              Real-time AI conversations powered by state-of-the-art language models
            </Typography>
          </Grid>
          <Grid item xs={12} md={4} textAlign="center">
            <Typography variant="h6" color="info.main">
              📊 XGBoost Analytics
            </Typography>
            <Typography variant="body2">
              Advanced machine learning for accurate diabetes risk prediction
            </Typography>
          </Grid>
          <Grid item xs={12} md={4} textAlign="center">
            <Typography variant="h6" color="warning.main">
              🌍 Multi-language Support
            </Typography>
            <Typography variant="body2">
              Accessible healthcare assistance in multiple languages
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default Home;
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
  Warning, // Changed from Emergency
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
      testDesc: 'Get your personalized diabetes risk prediction',
      chatTitle: 'AI Health Assistant',
      chatDesc: 'Chat about diabetes prevention and management',
      voiceTitle: 'Voice Chat',
      voiceDesc: 'Talk to our AI assistant in your preferred language',
      dietTitle: 'Personalized Diet Plans',
      dietDesc: 'Get customized meal plans for diabetes prevention',
      emergencyTitle: 'Symptom Checker',
      emergencyDesc: 'Assess emergency symptoms and get guidance',
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
      testDesc: 'Pata utabiri wa hatari yako ya kisukari',
      chatTitle: 'Msaidizi wa Afya wa AI',
      chatDesc: 'Zungumza kuhusu kuzuia na kudhibiti kisukari',
      voiceTitle: 'Mazungumzo kwa Sauti',
      voiceDesc: 'Zungumza na msaidizi wetu wa AI kwa lugha unayopendelea',
      dietTitle: 'Mipango Binafsi ya Lishe',
      dietDesc: 'Pata mipango ya vyakula iliyobinafsishwa kwa kuzuia kisukari',
      emergencyTitle: 'Kukagua Dalili',
      emergencyDesc: 'Kagua dalili za dharura na upate mwongozo',
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
      testDesc: 'Get your personalized diabetes risk prediction',
      chatTitle: 'AI Health Assistant',
      chatDesc: 'Chat about diabetes prevention and management',
      voiceTitle: 'Voice Chat',
      voiceDesc: 'Talk to our AI assistant in your preferred language',
      dietTitle: 'Personalized Diet Plans',
      dietDesc: 'Get customized meal plans for diabetes prevention',
      emergencyTitle: 'Symptom Checker',
      emergencyDesc: 'Assess emergency symptoms and get guidance',
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
    },
    {
      icon: <Chat sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t.chatTitle,
      description: t.testDesc,
      action: () => navigate('/chat'),
      buttonText: t.startChat,
    },
    {
      icon: <RecordVoiceOver sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t.voiceTitle,
      description: t.voiceDesc,
      action: () => navigate('/voice-chat'),
      buttonText: t.startVoice,
    },
    {
      icon: <Restaurant sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: t.dietTitle,
      description: t.dietDesc,
      action: () => navigate('/diet-plan'),
      buttonText: t.getDiet,
    },
    {
      icon: <Warning sx={{ fontSize: 40, color: 'secondary.main' }} />, // Changed from Emergency
      title: t.emergencyTitle,
      description: t.emergencyDesc,
      action: () => navigate('/emergency'),
      buttonText: t.checkSymptoms,
    },
  ];

  return (
    <Container maxWidth="lg">
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
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 6
                }
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Box mb={2}>
                  {feature.icon}
                </Box>
                <Typography variant="h5" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                  {feature.description}
                </Typography>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={feature.action}
                  fullWidth
                >
                  {feature.buttonText}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default Home;
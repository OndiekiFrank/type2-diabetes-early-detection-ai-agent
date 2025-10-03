import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Select,
  MenuItem,
  Box,
} from '@mui/material';
import { MedicalInformation } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Header = ({ language, setLanguage }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const translations = {
    english: {
      home: 'Home',
      test: 'Diabetes Test',
      chat: 'AI Chat',
      voiceChat: 'Voice Chat',
      diet: 'Diet Plan',
      emergency: 'Emergency Check',
    },
    swahili: {
      home: 'Nyumbani',
      test: 'Kipimo cha Kisukari',
      chat: 'Mazungumzo ya AI',
      voiceChat: 'Mazungumzo kwa Sauti',
      diet: 'Mpango wa Lishe',
      emergency: 'Kukagua Dharura',
    },
    sheng: {
      home: 'Home',
      test: 'Sugar Test',
      chat: 'AI Chat',
      voiceChat: 'Voice Chat',
      diet: 'Diet Plan',
      emergency: 'Emergency Check',
    }
  };

  const t = translations[language] || translations.english;

  return (
    <AppBar position="static" sx={{ bgcolor: 'primary.main' }}>
      <Toolbar>
        <MedicalInformation sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Insulyn AI
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, mr: 2, flexWrap: 'wrap' }}>
          <Button 
            color="inherit" 
            onClick={() => navigate('/')}
            variant={location.pathname === '/' ? 'outlined' : 'text'}
            size="small"
          >
            {t.home}
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/test')}
            variant={location.pathname === '/test' ? 'outlined' : 'text'}
            size="small"
          >
            {t.test}
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/chat')}
            variant={location.pathname === '/chat' ? 'outlined' : 'text'}
            size="small"
          >
            {t.chat}
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/voice-chat')}
            variant={location.pathname === '/voice-chat' ? 'outlined' : 'text'}
            size="small"
          >
            {t.voiceChat}
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/diet-plan')}
            variant={location.pathname === '/diet-plan' ? 'outlined' : 'text'}
            size="small"
          >
            {t.diet}
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/emergency')}
            variant={location.pathname === '/emergency' ? 'outlined' : 'text'}
            size="small"
          >
            {t.emergency}
          </Button>
        </Box>

        <Select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          sx={{ color: 'white', minWidth: 120 }}
          variant="outlined"
          size="small"
        >
          <MenuItem value="english">English</MenuItem>
          <MenuItem value="swahili">Swahili</MenuItem>
          <MenuItem value="sheng">Sheng</MenuItem>
        </Select>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
import React from 'react';
import { Box, Typography, Container } from '@mui/material';
import { Favorite, MedicalInformation } from '@mui/icons-material';

const Footer = ({ language }) => {
  const translations = {
    english: {
      tagline: 'Empowering Health Through AI',
      disclaimer: 'This is not a substitute for professional medical advice. Always consult healthcare professionals.',
      madeWith: 'Made with',
      for: 'for better health'
    },
    swahili: {
      tagline: 'Kuimarisha Afya Kupitia AI',
      disclaimer: 'Huu si uingizwaji wa ushauri wa matibabu. Shauriana daima na wataalamu wa afya.',
      madeWith: 'Imetengenezwa kwa',
      for: 'kwa afya bora'
    },
    sheng: {
      tagline: 'Empowering Health Through AI',
      disclaimer: 'This is not a substitute for professional medical advice. Always consult healthcare professionals.',
      madeWith: 'Made with',
      for: 'for better health'
    }
  };

  const t = translations[language] || translations.english;

  return (
    <Box
      component="footer"
      sx={{
        bgcolor: 'primary.main',
        color: 'white',
        py: 3,
        mt: 'auto'
      }}
    >
      <Container maxWidth="lg">
        <Typography variant="h6" gutterBottom align="center">
          <MedicalInformation sx={{ mr: 1, verticalAlign: 'middle' }} />
          Insulyn AI
        </Typography>
        <Typography variant="subtitle1" align="center" gutterBottom>
          {t.tagline}
        </Typography>
        <Typography variant="body2" align="center" sx={{ mt: 1 }}>
          {t.madeWith} <Favorite sx={{ color: 'red', fontSize: 16, verticalAlign: 'middle' }} /> {t.for}
        </Typography>
        <Typography variant="caption" display="block" align="center" sx={{ mt: 2, opacity: 0.8 }}>
          {t.disclaimer}
        </Typography>
      </Container>
    </Box>
  );
};

export default Footer;
import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Box,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Avatar,
  LinearProgress,
  Fade
} from '@mui/material';
import {
  Warning,
  CheckCircle,
  LocalHospital,
  ExpandMore,
  Favorite,
  FitnessCenter,
  Restaurant,
  MonitorHeart,
  Timeline,
  Group,
  School,
  Psychology,
  SelfImprovement,
  WaterDrop,
  Nightlight,
  Spa
} from '@mui/icons-material';

const PremiumDiabetesTest = ({ language = 'english' }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    blood_pressure: '',
    skin_thickness: '',
    insulin: '',
    weight: '',
    height: '',
    diabetes_pedigree_function: '',
    age: '',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Use environment variable or fallback to production URL
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://insulyn-ai-backend.onrender.com';

  const translations = {
    english: {
      title: 'Premium Diabetes Risk Assessment',
      steps: ['Personal Information', 'Health Metrics', 'Comprehensive Results'],
      next: 'Next',
      back: 'Back',
      submit: 'Get Premium Assessment',
      newTest: 'Start New Assessment',
      pregnancies: 'Number of Pregnancies',
      glucose: 'Glucose Level (mg/dL) *',
      bloodPressure: 'Blood Pressure (mmHg) *',
      skinThickness: 'Skin Thickness (mm)',
      insulin: 'Insulin Level (mu U/ml)',
      weight: 'Weight (kg) *',
      height: 'Height (cm) *',
      pedigree: 'Diabetes Pedigree Function',
      age: 'Age *',
      loading: 'Analyzing your comprehensive health profile...',
      riskHigh: 'High Risk',
      riskMedium: 'Moderate Risk',
      riskLow: 'Low Risk',
      riskVeryLow: 'Very Low Risk',
      probability: 'Probability',
      bmi: 'BMI',
      category: 'Category',
      confidence: 'Confidence',
      riskFactors: 'Key Risk Factors',
      recommendations: 'Premium Recommendations',
      nextSteps: 'Success Plan',
      requiredField: 'This field is required',
      executiveSummary: 'Executive Summary',
      healthMetrics: 'Health Metrics',
      vitalStatistics: 'Vital Statistics',
      metabolicAge: 'Metabolic Age',
      healthScore: 'Health Score',
      improvementAreas: 'Improvement Opportunities',
      lifestyleOptimization: 'Lifestyle Optimization',
      nutritionPlan: 'Nutrition Plan',
      fitnessProgram: 'Fitness Program',
      wellnessStrategies: 'Wellness Strategies',
      monitoringSchedule: 'Monitoring Schedule',
      supportResources: 'Support Resources',
      actionTimeline: 'Action Timeline',
      progressTracking: 'Progress Tracking',
      immediate: 'Immediate',
      days30: '30 Days',
      days90: '90 Days',
      keyMetrics: 'Key Metrics',
      tools: 'Tools',
      professional: 'Professional',
      educational: 'Educational',
      community: 'Community'
    },
    swahili: {
      title: 'Tathmini ya Premium ya Hatari ya Kisukari',
      steps: ['Taarifa Binafsi', 'Vipimo vya Afya', 'Matokeo ya kina'],
      next: 'Inayofuata',
      back: 'Nyuma',
      submit: 'Pata Tathmini ya Premium',
      newTest: 'Anza Tathmini Mpya',
      pregnancies: 'Idadi ya Mimba',
      glucose: 'Kiwango cha Glukosi (mg/dL) *',
      bloodPressure: 'Shinikizo la Damu (mmHg) *',
      skinThickness: 'Unene wa Ngozi (mm)',
      insulin: 'Kiwango cha Insulini (mu U/ml)',
      weight: 'Uzito (kg) *',
      height: 'Urefu (cm) *',
      pedigree: 'Utendaji wa Ukoo wa Kisukari',
      age: 'Umri *',
      loading: 'Inachambua wasifu wako wa afya kwa kina...',
      riskHigh: 'Hatari Kubwa',
      riskMedium: 'Hatari ya Wastani',
      riskLow: 'Hatari Ndogo',
      riskVeryLow: 'Hatari Ndogo Sana',
      probability: 'Uwezekano',
      bmi: 'BMI',
      category: 'Kategoria',
      confidence: 'Uthabiti',
      riskFactors: 'Sababu Kuu za Hatari',
      recommendations: 'Mapendekezo ya Premium',
      nextSteps: 'Mpango wa Mafanikio',
      requiredField: 'Sehemu hii inahitajika',
      executiveSummary: 'Muhtasari Mtendaji',
      healthMetrics: 'Vipimo vya Afya',
      vitalStatistics: 'Takwimu Muhimu',
      metabolicAge: 'Umri wa Metaboliki',
      healthScore: 'Alama ya Afya',
      improvementAreas: 'Fursa za Uboreshaji',
      lifestyleOptimization: 'Uboreshaji wa Mtindo wa Maisha',
      nutritionPlan: 'Mpango wa Lishe',
      fitnessProgram: 'Programu ya Mazoezi',
      wellnessStrategies: 'Mikakati ya Ustawi',
      monitoringSchedule: 'Ratiba ya Ufuatiliaji',
      supportResources: 'Rasilimali za Usaidizi',
      actionTimeline: 'Mpango wa Wakati wa Vitendo',
      progressTracking: 'Ufuatiliaji wa Maendeleo',
      immediate: 'Haraka',
      days30: 'Siku 30',
      days90: 'Siku 90',
      keyMetrics: 'Vipimo Muhimu',
      tools: 'Zana',
      professional: 'Kitaaluma',
      educational: 'Kielimu',
      community: 'Jumuiya'
    }
  };

  const t = translations[language] || translations.english;

  const handleInputChange = (field) => (event) => {
    setFormData({
      ...formData,
      [field]: event.target.value,
    });
  };

  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    
    try {
      // Prepare the request data
      const requestData = {
        pregnancies: parseInt(formData.pregnancies) || 0,
        glucose: parseFloat(formData.glucose),
        blood_pressure: parseFloat(formData.blood_pressure),
        skin_thickness: parseFloat(formData.skin_thickness) || 20,
        insulin: parseFloat(formData.insulin) || 80,
        weight: parseFloat(formData.weight),
        height: parseFloat(formData.height),
        diabetes_pedigree_function: parseFloat(formData.diabetes_pedigree_function) || 0.5,
        age: parseInt(formData.age),
        language: language
      };

      // Validate required fields
      if (!formData.glucose || !formData.blood_pressure || !formData.weight || !formData.height || !formData.age) {
        throw new Error('Please fill in all required fields');
      }

      // UPDATED: Use production backend URL
      const response = await fetch(`${API_BASE_URL}/api/v1/diabetes-assessment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Assessment failed');
      }

      const resultData = await response.json();
      setResult(resultData);
      setActiveStep(2);
      
    } catch (err) {
      setError(err.message || 'Failed to get assessment. Please try again.');
      console.error('Assessment error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNewTest = () => {
    setActiveStep(0);
    setFormData({
      pregnancies: '',
      glucose: '',
      blood_pressure: '',
      skin_thickness: '',
      insulin: '',
      weight: '',
      height: '',
      diabetes_pedigree_function: '',
      age: '',
    });
    setResult(null);
    setError('');
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high risk': return 'error';
      case 'moderate risk': return 'warning';
      case 'low risk': return 'success';
      case 'very low risk': return 'info';
      default: return 'info';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high risk': return <Warning />;
      default: return <CheckCircle />;
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high': return 'error';
      case 'moderate': return 'warning';
      case 'low': return 'success';
      default: return 'info';
    }
  };

  const getHealthScoreColor = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  // Validation for step progression
  const canProceedToStep1 = formData.age;
  const canProceedToStep2 = formData.glucose && formData.blood_pressure && formData.weight && formData.height;

  const steps = [
    // Step 1: Personal Information
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom color="primary">
          📋 Personal Health Profile
        </Typography>
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label={t.pregnancies}
          type="number"
          value={formData.pregnancies}
          onChange={handleInputChange('pregnancies')}
          helperText="Enter 0 if not applicable"
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          required
          label={t.age}
          type="number"
          value={formData.age}
          onChange={handleInputChange('age')}
          error={!formData.age}
          helperText={!formData.age ? t.requiredField : ''}
          variant="outlined"
        />
      </Grid>
    </Grid>,

    // Step 2: Health Metrics
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom color="primary">
          🩺 Health Metrics & Vital Signs
        </Typography>
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          required
          label={t.glucose}
          type="number"
          value={formData.glucose}
          onChange={handleInputChange('glucose')}
          error={!formData.glucose}
          helperText={!formData.glucose ? t.requiredField : 'Fasting blood sugar level'}
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          required
          label={t.bloodPressure}
          type="number"
          value={formData.blood_pressure}
          onChange={handleInputChange('blood_pressure')}
          error={!formData.blood_pressure}
          helperText={!formData.blood_pressure ? t.requiredField : 'Systolic blood pressure'}
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label={t.skinThickness}
          type="number"
          value={formData.skin_thickness}
          onChange={handleInputChange('skin_thickness')}
          helperText="Triceps skinfold thickness (mm)"
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label={t.insulin}
          type="number"
          value={formData.insulin}
          onChange={handleInputChange('insulin')}
          helperText="2-Hour serum insulin (mu U/ml)"
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          required
          label={t.weight}
          type="number"
          value={formData.weight}
          onChange={handleInputChange('weight')}
          error={!formData.weight}
          helperText={!formData.weight ? t.requiredField : ''}
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          required
          label={t.height}
          type="number"
          value={formData.height}
          onChange={handleInputChange('height')}
          error={!formData.height}
          helperText={!formData.height ? t.requiredField : 'Enter height in centimeters'}
          variant="outlined"
        />
      </Grid>
      <Grid item xs={12}>
        <TextField
          fullWidth
          label={t.pedigree}
          type="number"
          step="0.01"
          value={formData.diabetes_pedigree_function}
          onChange={handleInputChange('diabetes_pedigree_function')}
          helperText="Family history of diabetes (0.0 - 2.5)"
          variant="outlined"
        />
      </Grid>
    </Grid>,

    // Step 3: Comprehensive Results
    <Fade in={true} timeout={1000}>
      <Box>
        {loading ? (
          <Box textAlign="center" py={4}>
            <CircularProgress size={60} />
            <Typography variant="h6" sx={{ mt: 2 }}>
              {t.loading}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Analyzing your comprehensive health profile...
            </Typography>
          </Box>
        ) : result && (
          <Grid container spacing={3}>
            {/* Executive Summary */}
            <Grid item xs={12}>
              <Card elevation={3}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                      <LocalHospital />
                    </Avatar>
                    <Typography variant="h5" component="h2">
                      {t.executiveSummary}
                    </Typography>
                  </Box>
                  <Typography variant="body1" paragraph>
                    {result.executive_summary}
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    <Chip 
                      label={`Assessment ID: ${result.assessment_id}`} 
                      variant="outlined" 
                      size="small" 
                    />
                    <Chip 
                      label={new Date(result.timestamp).toLocaleDateString()} 
                      variant="outlined" 
                      size="small" 
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Risk Analysis */}
            <Grid item xs={12} md={6}>
              <Card elevation={2}>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    🎯 Diabetes Risk Assessment
                  </Typography>
                  <Alert 
                    severity={getRiskColor(result.risk_level)} 
                    icon={getRiskIcon(result.risk_level)}
                    sx={{ mb: 2 }}
                  >
                    <Typography variant="h6">
                      {result.risk_level}
                    </Typography>
                    <Typography variant="body2">
                      Probability: {((result.probability || result.risk_analysis?.diabetes_risk?.probability || 0) * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">
                      Confidence: {result.risk_analysis?.diabetes_risk?.confidence || 'moderate'}
                    </Typography>
                  </Alert>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Prevention Strategy:
                  </Typography>
                  <Typography variant="body2" paragraph>
                    {result.recommendations?.medical_followup || "Proactive lifestyle management"}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Key Risk Factors */}
            <Grid item xs={12} md={6}>
              <Card elevation={2}>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    ⚠️ {t.riskFactors}
                  </Typography>
                  <List dense>
                    {result.key_factors?.map((factor, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <Chip 
                            label={factor.severity} 
                            size="small" 
                            color={getSeverityColor(factor.severity)}
                          />
                        </ListItemIcon>
                        <ListItemText 
                          primary={factor.factor}
                          secondary={factor.severity === 'moderate' ? 'Consider lifestyle changes' : 'Monitor regularly'}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Health Metrics */}
            <Grid item xs={12}>
              <Card elevation={2}>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    📊 {t.healthMetrics}
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        {t.vitalStatistics}
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableBody>
                            <TableRow>
                              <TableCell><strong>BMI</strong></TableCell>
                              <TableCell>{result.health_metrics?.bmi}</TableCell>
                              <TableCell>
                                <Chip 
                                  label={result.health_metrics?.bmi_category} 
                                  size="small" 
                                  color="primary"
                                />
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Metabolic Age</strong></TableCell>
                              <TableCell>{result.health_metrics?.metabolic_age}</TableCell>
                              <TableCell>years</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Health Score</strong></TableCell>
                              <TableCell>
                                <Box display="flex" alignItems="center" gap={1}>
                                  {result.health_metrics?.health_score}/100
                                  <LinearProgress 
                                    variant="determinate" 
                                    value={result.health_metrics?.health_score} 
                                    sx={{ flexGrow: 1 }}
                                    color={getHealthScoreColor(result.health_metrics?.health_score)}
                                  />
                                </Box>
                              </TableCell>
                              <TableCell>
                                <Chip 
                                  label={`${result.health_metrics?.health_score}%`} 
                                  size="small" 
                                  color={getHealthScoreColor(result.health_metrics?.health_score)}
                                />
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        Health Insights
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemIcon>
                            <Favorite color="success" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={`Metabolic Health: ${result.risk_level === 'Low Risk' ? 'Good' : 'Needs Attention'}`}
                            secondary={`Risk Level: ${result.risk_level}`}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <MonitorHeart color="info" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={`Cardiovascular Status: ${result.health_metrics?.bmi_category === 'Normal' ? 'Healthy' : 'Monitor'}`}
                            secondary="Based on BMI and risk factors"
                          />
                        </ListItem>
                      </List>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Lifestyle Recommendations Accordion */}
            <Grid item xs={12}>
              <Accordion defaultExpanded elevation={2}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box display="flex" alignItems="center">
                    <Psychology color="primary" sx={{ mr: 2 }} />
                    <Typography variant="h6">🌟 Health Insights & Recommendations</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    {/* Nutrition Plan */}
                    <Grid item xs={12} md={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box display="flex" alignItems="center" mb={2}>
                            <Restaurant color="primary" sx={{ mr: 1 }} />
                            <Typography variant="h6">🍎 {t.nutritionPlan}</Typography>
                          </Box>
                          <Typography variant="body2" paragraph>
                            <strong>Focus on:</strong> Low-glycemic foods, portion control, and balanced meals
                          </Typography>
                          <Typography variant="body2">
                            <strong>Key Advice:</strong> Emphasize whole grains, limit added sugars, maintain hydration
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>

                    {/* Fitness Program */}
                    <Grid item xs={12} md={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box display="flex" alignItems="center" mb={2}>
                            <FitnessCenter color="primary" sx={{ mr: 1 }} />
                            <Typography variant="h6">💪 {t.fitnessProgram}</Typography>
                          </Box>
                          <Typography variant="body2" paragraph>
                            <strong>Cardio:</strong> 150 min/week moderate activity
                          </Typography>
                          <Typography variant="body2">
                            <strong>Strength:</strong> 2-3 times per week
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>

                    {/* Lifestyle Changes */}
                    <Grid item xs={12}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box display="flex" alignItems="center" mb={2}>
                            <Spa color="primary" sx={{ mr: 1 }} />
                            <Typography variant="h6">🌿 Lifestyle Changes</Typography>
                          </Box>
                          <List dense>
                            {result.recommendations?.lifestyle_changes?.map((change, index) => (
                              <ListItem key={index}>
                                <ListItemIcon>
                                  <SelfImprovement color="primary" />
                                </ListItemIcon>
                                <ListItemText primary={change} />
                              </ListItem>
                            ))}
                          </List>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Action Timeline */}
            <Grid item xs={12} md={6}>
              <Card elevation={2}>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    📅 {t.actionTimeline}
                  </Typography>
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      {t.immediate}:
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircle color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Consult healthcare provider" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircle color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Start monitoring glucose levels" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <CheckCircle color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Begin daily walking routine" />
                      </ListItem>
                    </List>
                    
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      {t.days30}:
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <Timeline color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Implement dietary changes" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <Timeline color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Establish exercise routine" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <Timeline color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Monitor progress weekly" />
                      </ListItem>
                    </List>
                    
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      {t.days90}:
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <WaterDrop color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Reassess health metrics" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <WaterDrop color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Adjust plan as needed" />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <WaterDrop color="primary" />
                        </ListItemIcon>
                        <ListItemText primary="Schedule follow-up appointment" />
                      </ListItem>
                    </List>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Support Resources */}
            <Grid item xs={12} md={6}>
              <Card elevation={2}>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    🤝 {t.supportResources}
                  </Typography>
                  <Box>
                    <Box display="flex" alignItems="center" mb={1}>
                      <LocalHospital color="primary" sx={{ mr: 1 }} />
                      <Typography variant="subtitle2">{t.professional}:</Typography>
                    </Box>
                    <Typography variant="body2" paragraph>
                      {result.recommendations?.medical_followup || "Schedule appointment with healthcare provider for comprehensive evaluation"}
                    </Typography>
                    
                    <Box display="flex" alignItems="center" mb={1}>
                      <School color="primary" sx={{ mr: 1 }} />
                      <Typography variant="subtitle2">{t.educational}:</Typography>
                    </Box>
                    <Typography variant="body2" paragraph>
                      Diabetes prevention resources and lifestyle guidance
                    </Typography>
                    
                    <Box display="flex" alignItems="center" mb={1}>
                      <Group color="primary" sx={{ mr: 1 }} />
                      <Typography variant="subtitle2">{t.community}:</Typography>
                    </Box>
                    <Typography variant="body2">
                      Health support groups and wellness communities
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* New Test Button */}
            <Grid item xs={12}>
              <Box textAlign="center" mt={4}>
                <Button 
                  variant="contained" 
                  onClick={handleNewTest}
                  startIcon={<LocalHospital />}
                  size="large"
                  sx={{ px: 4, py: 1.5 }}
                >
                  {t.newTest}
                </Button>
              </Box>
            </Grid>
          </Grid>
        )}
      </Box>
    </Fade>
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
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
          sx={{ color: 'white', mb: 4, opacity: 0.9 }}
        >
          Comprehensive AI-Powered Health Intelligence Platform
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 4, '& .MuiStepLabel-root': { color: 'white' } }}>
          {t.steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper elevation={6} sx={{ p: 4, minHeight: '400px' }}>
          {steps[activeStep]}
        </Paper>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
          <Button
            disabled={activeStep === 0 || loading}
            onClick={handleBack}
            variant="outlined"
            sx={{ color: 'white', borderColor: 'white' }}
          >
            {t.back}
          </Button>
          
          <Button
            variant="contained"
            onClick={activeStep === 1 ? handleSubmit : handleNext}
            disabled={
              loading || 
              (activeStep === 0 && !canProceedToStep1) ||
              (activeStep === 1 && !canProceedToStep2) ||
              activeStep === 2
            }
            startIcon={activeStep === 1 && loading ? <CircularProgress size={20} /> : null}
            sx={{ 
              background: 'white',
              color: '#667eea',
              fontWeight: 'bold',
              '&:hover': {
                background: '#f8f9fa'
              }
            }}
          >
            {activeStep === 1 ? (loading ? t.loading : t.submit) : t.next}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default PremiumDiabetesTest;
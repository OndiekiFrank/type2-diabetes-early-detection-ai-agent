import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Alert,
  Grid,
  CircularProgress,
  TextField,
  Chip,
  InputAdornment
} from '@mui/material';
import { Warning, CheckCircle, Cancel, LocalHospital, Person, Height, FitnessCenter } from '@mui/icons-material';

const EmergencyCheck = ({ language = 'english' }) => {
  const [selectedSymptomIds, setSelectedSymptomIds] = useState([]);
  const [assessment, setAssessment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPersonalData, setShowPersonalData] = useState(false);

  // Personal data state
  const [personalData, setPersonalData] = useState({
    age: '',
    weight: '',
    height: '',
    existingConditions: [],
    currentMedications: [],
    newCondition: '',
    newMedication: ''
  });

  // Use environment variable or fallback to production URL
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://insulyn-ai-backend.onrender.com';

  const symptomList = [
    { id: 'extreme_thirst', label: 'Extreme thirst' },
    { id: 'frequent_urination', label: 'Frequent urination' },
    { id: 'blurred_vision', label: 'Blurred vision' },
    { id: 'fatigue', label: 'Extreme fatigue' },
    { id: 'weight_loss', label: 'Unexplained weight loss' },
    { id: 'nausea', label: 'Nausea or vomiting' },
    { id: 'confusion', label: 'Confusion or difficulty concentrating' },
    { id: 'breathing', label: 'Difficulty breathing' },
    { id: 'abdominal_pain', label: 'Abdominal pain' },
    { id: 'fruity_breath', label: 'Fruity-smelling breath' },
    { id: 'dizziness', label: 'Dizziness or lightheadedness' },
    { id: 'rapid_heartbeat', label: 'Rapid heartbeat' }
  ];

  const handleSymptomChange = (symptomId) => (event) => {
    setSelectedSymptomIds(prev => {
      if (event.target.checked) {
        return [...prev, symptomId];
      } else {
        return prev.filter(id => id !== symptomId);
      }
    });
  };

  const handlePersonalDataChange = (field) => (event) => {
    setPersonalData(prev => ({
      ...prev,
      [field]: event.target.value
    }));
  };

  const addCondition = () => {
    if (personalData.newCondition.trim()) {
      setPersonalData(prev => ({
        ...prev,
        existingConditions: [...prev.existingConditions, prev.newCondition.trim()],
        newCondition: ''
      }));
    }
  };

  const removeCondition = (conditionToRemove) => {
    setPersonalData(prev => ({
      ...prev,
      existingConditions: prev.existingConditions.filter(condition => condition !== conditionToRemove)
    }));
  };

  const addMedication = () => {
    if (personalData.newMedication.trim()) {
      setPersonalData(prev => ({
        ...prev,
        currentMedications: [...prev.currentMedications, prev.newMedication.trim()],
        newMedication: ''
      }));
    }
  };

  const removeMedication = (medicationToRemove) => {
    setPersonalData(prev => ({
      ...prev,
      currentMedications: prev.currentMedications.filter(medication => medication !== medicationToRemove)
    }));
  };

  const getSymptomLabels = (symptomIds) => {
    return symptomIds.map(id => {
      const symptom = symptomList.find(s => s.id === id);
      return symptom ? symptom.label : id;
    });
  };

  const assessSymptoms = async () => {
    setLoading(true);
    setAssessment(null);
    setError(null);
    
    try {
      const symptomLabels = getSymptomLabels(selectedSymptomIds);
      
      const requestBody = {
        symptoms: symptomLabels,
        language: language,
        age: personalData.age ? parseInt(personalData.age) : null,
        weight: personalData.weight ? parseFloat(personalData.weight) : null,
        height: personalData.height ? parseFloat(personalData.height) : null,
        existing_conditions: personalData.existingConditions,
        current_medications: personalData.currentMedications
      };

      console.log('Sending request:', requestBody);

      // UPDATED: Use production backend URL
      const response = await fetch(`${API_BASE_URL}/api/v1/emergency-assessment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAssessment(data);
      
    } catch (error) {
      console.error('Error assessing symptoms:', error);
      setError('Failed to assess symptoms. Please try again.');
      
      // Enhanced fallback assessment with personal data
      const symptomLabels = getSymptomLabels(selectedSymptomIds);
      const hasCritical = selectedSymptomIds.some(symptom => 
        ['breathing', 'confusion', 'rapid_heartbeat'].includes(symptom)
      );
      
      // Use personal data in fallback
      const riskFactors = [];
      if (personalData.age && personalData.age > 60) riskFactors.push('Senior age group');
      if (personalData.existingConditions.length > 0) riskFactors.push('Existing health conditions');
      if (personalData.currentMedications.length > 0) riskFactors.push('Current medications');
      
      setAssessment({
        assessment: hasCritical ? 
          "URGENT: You are experiencing symptoms that may require immediate medical attention." :
          `Based on your ${symptomLabels.length} symptoms${riskFactors.length > 0 ? ' and health profile' : ''}, monitor closely.`,
        personalized_analysis: riskFactors.length > 0 ? 
          `Personal factors considered: ${riskFactors.join(', ')}` : 
          "Basic assessment completed",
        urgency_level: hasCritical ? 'critical' : 'medium',
        recommendations: hasCritical ? [
          "Seek emergency medical care immediately",
          "Do not drive yourself to the hospital",
          "Contact emergency services or go to nearest ER",
          "Inform medical staff about your health conditions"
        ] : [
          "Schedule an appointment with your healthcare provider",
          "Monitor your symptoms regularly",
          "Stay hydrated and rest",
          "Avoid strenuous activities"
        ],
        risk_factors: riskFactors.length > 0 ? riskFactors : ["Multiple symptoms present"],
        next_steps: hasCritical ? [
          "Call emergency services now",
          "Prepare medical information and medications list",
          "Have someone stay with you"
        ] : [
          "Continue monitoring symptoms",
          "Keep hydration up",
          "Follow up with doctor within 24 hours"
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (urgencyLevel) => {
    switch (urgencyLevel) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'info';
    }
  };

  const getSeverityIcon = (urgencyLevel) => {
    switch (urgencyLevel) {
      case 'critical':
      case 'high':
        return <Warning />;
      default:
        return <CheckCircle />;
    }
  };

  const getSeverityTitle = (urgencyLevel) => {
    switch (urgencyLevel) {
      case 'critical':
        return 'CRITICAL - Emergency Attention Required';
      case 'high':
        return 'HIGH - Urgent Medical Attention Needed';
      case 'medium':
        return 'Assessment Complete - Monitor Closely';
      case 'low':
        return 'Assessment Complete - Continue Monitoring';
      default:
        return 'Assessment Complete';
    }
  };

  const hasPersonalData = personalData.age || personalData.weight || personalData.height || 
                          personalData.existingConditions.length > 0 || personalData.currentMedications.length > 0;

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Symptom Checker
        </Typography>
        
        <Typography variant="body1" color="text.secondary" align="center" gutterBottom>
          Assess emergency symptoms and get personalized guidance
        </Typography>

        <Alert severity="warning" sx={{ mb: 3 }}>
          <strong>Important:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice. In case of emergency, seek immediate medical attention.
        </Alert>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Symptoms Selection */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Select your symptoms:
            </Typography>
            <FormGroup>
              <Grid container spacing={1}>
                {symptomList.map((symptom) => (
                  <Grid item xs={12} sm={6} key={symptom.id}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={selectedSymptomIds.includes(symptom.id)}
                          onChange={handleSymptomChange(symptom.id)}
                          color="primary"
                        />
                      }
                      label={symptom.label}
                    />
                  </Grid>
                ))}
              </Grid>
            </FormGroup>
            
            {selectedSymptomIds.length > 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Selected {selectedSymptomIds.length} symptom(s)
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Personal Data Section */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="h6">
                Personal Health Information (Optional)
              </Typography>
              <Button 
                variant="outlined" 
                size="small"
                onClick={() => setShowPersonalData(!showPersonalData)}
              >
                {showPersonalData ? 'Hide' : 'Add Personal Info'}
              </Button>
            </Box>
            
            {showPersonalData && (
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Age"
                    type="number"
                    value={personalData.age}
                    onChange={handlePersonalDataChange('age')}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Person />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Weight (kg)"
                    type="number"
                    value={personalData.weight}
                    onChange={handlePersonalDataChange('weight')}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <FitnessCenter />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Height (cm)"
                    type="number"
                    value={personalData.height}
                    onChange={handlePersonalDataChange('height')}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Height />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>

                {/* Existing Conditions */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Existing Health Conditions
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap" mb={1}>
                    {personalData.existingConditions.map((condition, index) => (
                      <Chip
                        key={index}
                        label={condition}
                        onDelete={() => removeCondition(condition)}
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                  <Box display="flex" gap={1}>
                    <TextField
                      size="small"
                      placeholder="Add condition"
                      value={personalData.newCondition}
                      onChange={handlePersonalDataChange('newCondition')}
                      onKeyPress={(e) => e.key === 'Enter' && addCondition()}
                    />
                    <Button variant="outlined" onClick={addCondition}>
                      Add
                    </Button>
                  </Box>
                </Grid>

                {/* Current Medications */}
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Current Medications
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap" mb={1}>
                    {personalData.currentMedications.map((medication, index) => (
                      <Chip
                        key={index}
                        label={medication}
                        onDelete={() => removeMedication(medication)}
                        color="secondary"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                  <Box display="flex" gap={1}>
                    <TextField
                      size="small"
                      placeholder="Add medication"
                      value={personalData.newMedication}
                      onChange={handlePersonalDataChange('newMedication')}
                      onKeyPress={(e) => e.key === 'Enter' && addMedication()}
                    />
                    <Button variant="outlined" onClick={addMedication}>
                      Add
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            )}

            {hasPersonalData && !showPersonalData && (
              <Alert severity="success">
                Personal health information included for personalized assessment
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Assessment Button */}
        <Box textAlign="center">
          <Button
            variant="contained"
            color="secondary"
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <LocalHospital />}
            onClick={assessSymptoms}
            disabled={loading || selectedSymptomIds.length === 0}
            sx={{ px: 4, py: 1.5, minWidth: 200 }}
          >
            {loading ? 'Assessing...' : 'Get Personalized Assessment'}
          </Button>
        </Box>

        {/* Assessment Results */}
        {assessment && (
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Alert 
                severity={getSeverityColor(assessment.urgency_level)} 
                sx={{ mb: 2 }}
                icon={getSeverityIcon(assessment.urgency_level)}
              >
                <Typography variant="h6">
                  {getSeverityTitle(assessment.urgency_level)}
                </Typography>
              </Alert>
              
              <Typography variant="body1" paragraph sx={{ lineHeight: 1.6, mb: 2 }}>
                {assessment.assessment}
              </Typography>

              {assessment.personalized_analysis && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    {assessment.personalized_analysis}
                  </Typography>
                </Alert>
              )}
              
              {assessment.risk_factors && assessment.risk_factors.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom>
                    Risk Factors:
                  </Typography>
                  <List dense sx={{ mb: 2 }}>
                    {assessment.risk_factors.map((factor, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <Warning color="warning" />
                        </ListItemIcon>
                        <ListItemText primary={factor} />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
              
              <Typography variant="h6" gutterBottom>
                Recommendations:
              </Typography>
              <List>
                {(assessment.recommendations || []).map((rec, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {rec.includes('emergency') || rec.includes('Urgent') || rec.includes('911') ? (
                        <Cancel color="error" />
                      ) : (
                        <CheckCircle color="success" />
                      )}
                    </ListItemIcon>
                    <ListItemText primary={rec} />
                  </ListItem>
                ))}
              </List>
              
              {assessment.next_steps && assessment.next_steps.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                    Next Steps:
                  </Typography>
                  <List dense>
                    {assessment.next_steps.map((step, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckCircle color="primary" />
                        </ListItemIcon>
                        <ListItemText primary={step} />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
              
              {(assessment.urgency_level === 'critical' || assessment.urgency_level === 'high') && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  <strong>Emergency Contact:</strong> Call your local emergency number (911/112) or go to the nearest hospital immediately.
                </Alert>
              )}
            </CardContent>
          </Card>
        )}
      </Paper>
    </Container>
  );
};

export default EmergencyCheck;
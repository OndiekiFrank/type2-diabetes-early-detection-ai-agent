import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  Snackbar
} from '@mui/material';
import { Restaurant, Download, Calculate, LocalDining } from '@mui/icons-material';

const DietPlan = ({ language = 'english' }) => {
  const [formData, setFormData] = useState({
    age: '',
    weight: '',
    height: '',
    gender: '',
    dietaryPreference: 'balanced',
    healthConditions: '',
    activityLevel: 'moderate',
    goals: 'diabetes_prevention',
    allergies: '',
    typicalDay: ''
  });
  const [dietPlan, setDietPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const generateDietPlan = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    
    try {
      // Remove fields that backend doesn't expect
      const requestData = { ...formData };
      delete requestData.bmi;
      delete requestData.bmi_category;
      
      const response = await fetch('http://localhost:8000/api/v1/diet-plan/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate diet plan');
      }

      const data = await response.json();
      setDietPlan(data);
      setSuccess('Diet plan generated successfully!');
      
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to generate personalized diet plan. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const calculateBMI = () => {
    if (formData.weight && formData.height) {
      const heightInMeters = formData.height / 100;
      return (formData.weight / (heightInMeters * heightInMeters)).toFixed(1);
    }
    return null;
  };

  const getBMICategory = (bmi) => {
    if (!bmi) return '';
    if (bmi < 18.5) return 'underweight';
    if (bmi <= 24.9) return 'normal';
    if (bmi <= 29.9) return 'overweight';
    return 'obese';
  };

  const calculateDailyCalories = () => {
    if (formData.weight && formData.height && formData.age && formData.gender && formData.activityLevel) {
      let bmr;
      if (formData.gender === 'male') {
        bmr = 88.362 + (13.397 * formData.weight) + (4.799 * formData.height) - (5.677 * formData.age);
      } else {
        bmr = 447.593 + (9.247 * formData.weight) + (3.098 * formData.height) - (4.330 * formData.age);
      }

      const activityMultipliers = {
        sedentary: 1.2,
        light: 1.375,
        moderate: 1.55,
        active: 1.725,
        very_active: 1.9
      };

      return Math.round(bmr * (activityMultipliers[formData.activityLevel] || 1.55));
    }
    return null;
  };

  const bmi = calculateBMI();
  const dailyCalories = calculateDailyCalories();
  const isFormValid = formData.age && formData.weight && formData.height && formData.gender;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center" color="primary">
          🥗 Personalized Diabetes Diet Plan
        </Typography>
        
        <Typography variant="body1" color="text.secondary" align="center" gutterBottom>
          Get AI-powered customized meal plans tailored to your unique profile
        </Typography>

        {/* Notifications */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        <Snackbar 
          open={!!success} 
          autoHideDuration={6000} 
          onClose={() => setSuccess('')}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert severity="success" onClose={() => setSuccess('')}>
            {success}
          </Alert>
        </Snackbar>

        <Grid container spacing={3} sx={{ mt: 1 }}>
          {/* Basic Information */}
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Age"
              type="number"
              value={formData.age}
              onChange={(e) => handleInputChange('age', e.target.value)}
              helperText="Your current age"
              required
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Weight (kg)"
              type="number"
              value={formData.weight}
              onChange={(e) => handleInputChange('weight', e.target.value)}
              helperText="Current weight"
              required
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Height (cm)"
              type="number"
              value={formData.height}
              onChange={(e) => handleInputChange('height', e.target.value)}
              helperText="Height in centimeters"
              required
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth required>
              <InputLabel>Gender</InputLabel>
              <Select
                value={formData.gender}
                label="Gender"
                onChange={(e) => handleInputChange('gender', e.target.value)}
              >
                <MenuItem value="male">Male</MenuItem>
                <MenuItem value="female">Female</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth>
              <InputLabel>Dietary Preference</InputLabel>
              <Select
                value={formData.dietaryPreference}
                label="Dietary Preference"
                onChange={(e) => handleInputChange('dietaryPreference', e.target.value)}
              >
                <MenuItem value="vegetarian">Vegetarian</MenuItem>
                <MenuItem value="non-vegetarian">Non-Vegetarian</MenuItem>
                <MenuItem value="vegan">Vegan</MenuItem>
                <MenuItem value="balanced">Balanced</MenuItem>
                <MenuItem value="mediterranean">Mediterranean</MenuItem>
                <MenuItem value="low_carb">Low Carb</MenuItem>
                <MenuItem value="diabetic_friendly">Diabetic Friendly</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth>
              <InputLabel>Activity Level</InputLabel>
              <Select
                value={formData.activityLevel}
                label="Activity Level"
                onChange={(e) => handleInputChange('activityLevel', e.target.value)}
              >
                <MenuItem value="sedentary">Sedentary (little exercise)</MenuItem>
                <MenuItem value="light">Light (1-3 days/week)</MenuItem>
                <MenuItem value="moderate">Moderate (3-5 days/week)</MenuItem>
                <MenuItem value="active">Active (6-7 days/week)</MenuItem>
                <MenuItem value="very_active">Very Active (intense daily)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Primary Goal</InputLabel>
              <Select
                value={formData.goals}
                label="Primary Goal"
                onChange={(e) => handleInputChange('goals', e.target.value)}
              >
                <MenuItem value="diabetes_prevention">Diabetes Prevention</MenuItem>
                <MenuItem value="blood_sugar_control">Blood Sugar Control</MenuItem>
                <MenuItem value="weight_loss">Weight Loss</MenuItem>
                <MenuItem value="weight_gain">Healthy Weight Gain</MenuItem>
                <MenuItem value="maintenance">Maintenance</MenuItem>
                <MenuItem value="gestational_diabetes">Gestational Diabetes</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Health Conditions"
              value={formData.healthConditions}
              onChange={(e) => handleInputChange('healthConditions', e.target.value)}
              placeholder="e.g., high BP, cholesterol, PCOS, kidney issues"
              helperText="Any existing health conditions"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Food Allergies/Intolerances"
              value={formData.allergies}
              onChange={(e) => handleInputChange('allergies', e.target.value)}
              placeholder="e.g., gluten, dairy, nuts, shellfish"
              helperText="List any food allergies or intolerances"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Typical Daily Routine"
              value={formData.typicalDay}
              onChange={(e) => handleInputChange('typicalDay', e.target.value)}
              placeholder="e.g., wake up 7AM, work 9-5, exercise evening"
              helperText="Help us time your meals optimally"
            />
          </Grid>
        </Grid>

        {/* Health Metrics Display */}
        {(bmi || dailyCalories) && (
          <Card sx={{ mt: 3, bgcolor: 'background.default' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Calculate sx={{ mr: 1, verticalAlign: 'middle' }} />
                Your Health Metrics
              </Typography>
              <Grid container spacing={2}>
                {bmi && (
                  <Grid item xs={12} sm={4}>
                    <Chip 
                      label={`BMI: ${bmi} (${getBMICategory(bmi)})`}
                      color={
                        getBMICategory(bmi) === 'normal' ? 'success' : 
                        getBMICategory(bmi) === 'overweight' ? 'warning' : 'error'
                      }
                      variant="outlined"
                    />
                  </Grid>
                )}
                {dailyCalories && (
                  <Grid item xs={12} sm={4}>
                    <Chip 
                      label={`Daily Calories: ${dailyCalories}`}
                      color="primary"
                      variant="outlined"
                    />
                  </Grid>
                )}
                {formData.goals && (
                  <Grid item xs={12} sm={4}>
                    <Chip 
                      label={`Goal: ${formData.goals.replace(/_/g, ' ')}`}
                      color="secondary"
                      variant="outlined"
                    />
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        )}

        <Box textAlign="center" sx={{ mt: 3 }}>
          <Button
            variant="contained"
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <LocalDining />}
            onClick={generateDietPlan}
            disabled={loading || !isFormValid}
            sx={{ px: 4, py: 1.5, fontSize: '1.1rem' }}
          >
            {loading ? 'Generating Your Plan...' : 'Generate Personalized Diet Plan'}
          </Button>
        </Box>

        {dietPlan && (
          <Card sx={{ mt: 4, border: 2, borderColor: 'primary.main' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5" color="primary">
                  🥗 Your Personalized Diabetes Diet Plan
                </Typography>
                <Button startIcon={<Download />} variant="outlined" color="primary">
                  Download PDF
                </Button>
              </Box>
              
              <Divider sx={{ mb: 3 }} />
              
              {/* Plan Overview */}
              {dietPlan.overview && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom color="secondary">
                    📊 Plan Overview
                  </Typography>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.6 }}>
                    {dietPlan.overview}
                  </Typography>
                </Box>
              )}

              {/* Daily Meal Plan */}
              {dietPlan.daily_plan && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom color="secondary">
                    🍽️ Daily Meal Plan
                  </Typography>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.8 }}>
                    {dietPlan.daily_plan}
                  </Typography>
                </Box>
              )}

              {/* Grocery List */}
              {dietPlan.grocery_list && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom color="secondary">
                    🛒 Recommended Groceries
                  </Typography>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.8 }}>
                    {dietPlan.grocery_list}
                  </Typography>
                </Box>
              )}

              {/* Important Notes */}
              {dietPlan.important_notes && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    💡 Important Notes:
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                    {dietPlan.important_notes}
                  </Typography>
                </Alert>
              )}

              {/* Nutritional Info */}
              {dietPlan.nutritional_info && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="text.secondary">
                    📈 Nutritional Targets:
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {dietPlan.nutritional_info.daily_calories} calories • {dietPlan.nutritional_info.protein_grams}g protein • {dietPlan.nutritional_info.carbs_grams}g carbs • {dietPlan.nutritional_info.fat_grams}g fat
                  </Typography>
                </Box>
              )}

              {/* Timestamp */}
              {dietPlan.timestamp && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Generated on: {new Date(dietPlan.timestamp).toLocaleString()} • 
                  Processing time: {dietPlan.generation_time}s
                </Typography>
              )}
            </CardContent>
          </Card>
        )}
      </Paper>
    </Container>
  );
};

export default DietPlan;
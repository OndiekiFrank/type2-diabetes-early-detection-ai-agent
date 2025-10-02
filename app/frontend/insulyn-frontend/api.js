// API Service for Insulyn AI Frontend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Generic API request function
async function apiRequest(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API request failed for ${endpoint}:`, error);
    throw error;
  }
}

// Diabetes Risk Assessment
export async function predictDiabetesRisk(formData) {
  return apiRequest('/api/v1/predict', {
    method: 'POST',
    body: JSON.stringify(formData),
  });
}

// Chat with AI Assistant
export async function chatWithAI(message, language = 'english') {
  return apiRequest('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({ message, language }),
  });
}

// Voice Chat
export async function voiceChat(audioData, language = 'english') {
  return apiRequest('/api/v1/chat/voice', {
    method: 'POST',
    body: JSON.stringify({ audio_data: audioData, language }),
  });
}

// Generate Diet Plan
export async function generateDietPlan(userData, language = 'english') {
  return apiRequest('/api/v1/diet-plan', {
    method: 'POST',
    body: JSON.stringify({ ...userData, language }),
  });
}

// Emergency Symptom Assessment
export async function assessSymptoms(symptoms, language = 'english') {
  return apiRequest('/api/v1/emergency-assessment', {
    method: 'POST',
    body: JSON.stringify({ symptoms, language }),
  });
}

// Get Chat Topics
export async function getChatTopics() {
  return apiRequest('/api/v1/chat/topics');
}

// Clear Chat History
export async function clearChatHistory() {
  return apiRequest('/api/v1/chat/clear', {
    method: 'POST',
  });
}

// Health Check
export async function healthCheck() {
  return apiRequest('/health');
}

// Default export with all functions
export default {
  predictDiabetesRisk,
  chatWithAI,
  voiceChat,
  generateDietPlan,
  assessSymptoms,
  getChatTopics,
  clearChatHistory,
  healthCheck,
};

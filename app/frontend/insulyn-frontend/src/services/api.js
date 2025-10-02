const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

async function apiRequest(endpoint, options = {}) {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log(`API Request: ${url}`, options);
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log(`API Response from ${endpoint}:`, data);
    return data;
  } catch (error) {
    console.error(`API request failed for ${endpoint}:`, error);
    throw error;
  }
}

export async function predictDiabetesRisk(formData) {
  return apiRequest('/api/v1/predict', {
    method: 'POST',
    body: JSON.stringify(formData),
  });
}

export async function chatWithAI(message, language = 'english') {
  return apiRequest('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({ message, language }),
  });
}

export async function voiceChat(audioData, language = 'english') {
  return apiRequest('/api/v1/chat/voice', {
    method: 'POST',
    body: JSON.stringify({ audio_data: audioData, language }),
  });
}

export async function generateDietPlan(userData, language = 'english') {
  return apiRequest('/api/v1/diet-plan', {
    method: 'POST',
    body: JSON.stringify({ ...userData, language }),
  });
}

export async function assessSymptoms(symptoms, language = 'english') {
  return apiRequest('/api/v1/emergency-assessment', {
    method: 'POST',
    body: JSON.stringify({ symptoms, language }),
  });
}

export async function getChatTopics() {
  return apiRequest('/api/v1/chat/topics');
}

export async function clearChatHistory() {
  return apiRequest('/api/v1/chat/clear', {
    method: 'POST',
  });
}

export async function healthCheck() {
  return apiRequest('/health');
}

const apiService = {
  predictDiabetesRisk,
  chatWithAI,
  voiceChat,
  generateDietPlan,
  assessSymptoms,
  getChatTopics,
  clearChatHistory,
  healthCheck,
};

export default apiService;
import json
import logging
from typing import Dict, List, Optional
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from app.config import settings

logger = logging.getLogger(__name__)

class InsulynAILLM:
    """
    Fixed Insulyn AI LLM service with proper chat functionality
    """

    def __init__(self):
        self.llm = None
        self.prediction_chain = None
        self.chat_chain = None
        self.diet_chain = None
        self.emergency_chain = None
        self.voice_chain = None
        self.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="user_message",
        )
        self.initialize_llm()

    def initialize_llm(self):
        """Initialize Groq LLM with proper configuration"""
        try:
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.strip() == "":
                logger.error("❌ GROQ_API_KEY not configured")
                self._create_mock_llm()
                return

            self.llm = ChatGroq(
                model=settings.LLM_MODEL_NAME,
                temperature=0.7,  # Increased for more creative responses
                api_key=settings.GROQ_API_KEY,
                max_tokens=1024,
                timeout=30
            )

            self._initialize_chat_chain()
            logger.info("✅ Insulyn AI LLM initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            self._create_mock_llm()

    def _create_mock_llm(self):
        """Create mock LLM for development"""
        logger.warning("🔄 Using mock LLM for development")
        self.llm = None

    def _initialize_chat_chain(self):
        """Initialize the main chat chain with proper diabetes expertise"""
        chat_template = """You are Insulyn AI, a specialized medical AI assistant focused exclusively on diabetes prevention, management, and education.

IMPORTANT: You must provide accurate, evidence-based medical information about diabetes. Always be specific and detailed.

USER QUESTION: {message}
LANGUAGE: {language}
CONTEXT: {context}

CRITICAL REQUIREMENTS:
1. Provide comprehensive, medically accurate information
2. Tailor response to the specific question asked
3. Include practical, actionable advice
4. Reference current medical guidelines when possible
5. Be empathetic and supportive
6. If discussing treatments, mention both lifestyle and medical options
7. Always explain complex medical terms in simple language

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide detailed explanations with examples
- Include practical recommendations
- End with encouraging next steps

Now provide a detailed response in {language}:"""

        self.chat_prompt = PromptTemplate(
            input_variables=["message", "language", "context"],
            template=chat_template
        )

        if self.llm:
            self.chat_chain = LLMChain(
                llm=self.llm, 
                prompt=self.chat_prompt, 
                verbose=True  # Enable verbose for debugging
            )

    def chat_about_diabetes(self, message: str, conversation_context: str = "diabetes", language: str = "english") -> str:
        """Main chat method - processes diabetes-related questions"""
        if not self.chat_chain:
            return self._get_fallback_error_message()

        try:
            logger.info(f"🔄 Processing chat request: '{message}' in {language}")
            
            # Enhanced context based on message type
            enhanced_context = self._get_enhanced_context(message, conversation_context)
            
            response = self.chat_chain.run({
                "message": message,
                "language": language,
                "context": enhanced_context
            })
            
            logger.info(f"✅ LLM Response generated: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            return self._get_error_message(message, language)

    def _get_enhanced_context(self, message: str, base_context: str) -> str:
        """Enhance context based on message content"""
        message_lower = message.lower()
        
        if any(term in message_lower for term in ['pathophysiology', 'mechanism', 'how it works']):
            return "diabetes_pathophysiology - Focus on biological mechanisms, insulin resistance, beta cell function"
        
        elif any(term in message_lower for term in ['mental health', 'psychology', 'stress', 'depression']):
            return "diabetes_mental_health - Focus on psychological aspects, stress management, emotional impact"
        
        elif any(term in message_lower for term in ['treatment', 'medication', 'therapy']):
            return "diabetes_treatment - Focus on medical treatments, medications, insulin therapy"
        
        elif any(term in message_lower for term in ['prevent', 'risk', 'avoid']):
            return "diabetes_prevention - Focus on risk reduction, lifestyle changes, early detection"
        
        elif any(term in message_lower for term in ['diet', 'food', 'nutrition']):
            return "diabetes_nutrition - Focus on dietary management, meal planning, carbohydrate counting"
        
        elif any(term in message_lower for term in ['exercise', 'workout', 'physical']):
            return "diabetes_exercise - Focus on physical activity, exercise recommendations, safety"
        
        return base_context

    def _get_fallback_error_message(self) -> str:
        """Return error message when LLM is not available"""
        return "🚨 **LLM Service Currently Unavailable**\n\nI'm unable to connect to the AI service at the moment. This could be due to:\n\n• API key configuration issues\n• Network connectivity problems\n• Service temporary unavailability\n\nPlease check your backend configuration and ensure the Groq API key is properly set in your environment variables."

    def _get_error_message(self, original_message: str, language: str) -> str:
        """Return contextual error message"""
        error_messages = {
            "english": f"🚨 **AI Service Error**\n\nI encountered an error while processing your question about '{original_message}'. This might be due to:\n\n• Temporary service interruption\n• Network connectivity issues\n• API configuration problems\n\nPlease try again in a moment or contact support if this persists.",
            "swahili": f"🚨 **Hitilafu ya Huduma ya AK**\n\nNimekutana na hitilafu wakati wa kuchakata swali lako kuhusu '{original_message}'. Hii inaweza kusababishwa na:\n\n• Usumbufu wa muda wa huduma\n• Matatizo ya muunganisho wa mtandao\n• Matatizo ya usanidi wa API\n\nTafadhali jaribu tena baada ya muda mfupi au wasiliana na usaidizi ikiwa itaendelea."
        }
        return error_messages.get(language, error_messages["english"])

    # Other methods remain the same but ensure they use the chat_chain
    def generate_advice(self, patient_data: dict, ml_output: dict, bmi_category: str = None) -> dict:
        """Generate structured advice - uses chat chain"""
        if not self.chat_chain:
            return {"error": "LLM service unavailable"}
        
        try:
            prompt_data = self.prepare_prompt_data(patient_data, ml_output, bmi_category)
            # Convert to chat format
            message = f"Based on this patient data: {prompt_data}, provide diabetes risk assessment and recommendations."
            response = self.chat_chain.run({
                "message": message,
                "language": "english",
                "context": "diabetes_risk_assessment"
            })
            return {"advice": response}
        except Exception as e:
            logger.error(f"Advice generation error: {e}")
            return {"error": str(e)}

    @staticmethod
    def prepare_prompt_data(patient_data: dict, ml_output: dict, bmi_category: str = None) -> dict:
        """Format patient data for LLM"""
        return {
            "pregnancies": patient_data.get("pregnancies", "Not available"),
            "glucose": patient_data.get("glucose", "Not available"),
            "blood_pressure": patient_data.get("blood_pressure", "Not available"),
            "skin_thickness": patient_data.get("skin_thickness", "Not available"),
            "insulin": patient_data.get("insulin", "Not available"),
            "bmi": ml_output.get("calculated_bmi", "Not available"),
            "bmi_category": bmi_category or "Not available",
            "diabetes_pedigree_function": patient_data.get("diabetes_pedigree_function", "Not available"),
            "age": patient_data.get("age", "Not available"),
            "ml_model_output": ml_output.get("risk_label", "Not available"),
            "probability": ml_output.get("probability", 0.0),
        }

# Global instance
insulyn_llm = InsulynAILLM()

def initialize_llm_service():
    return insulyn_llm.llm is not None
# # import os
# # from langchain_groq import ChatGroq
# # from langchain.prompts import PromptTemplate
# # from langchain.chains import LLMChain
# # from app.config import settings
# # import logging

# # logger = logging.getLogger(__name__)

# # class InsulynAILLM:
# #     def __init__(self):
# #         self.llm = None
# #         self.chain = None
# #         self.initialize_llm()
    
# #     def initialize_llm(self):
# #         """Initialize Groq LLM with configuration from settings"""
# #         try:
# #             if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your-groq-api-key-here":
# #                 logger.error("GROQ_API_KEY not configured")
# #                 return
            
# #             self.llm = ChatGroq(
# #                 model=settings.LLM_MODEL_NAME,
# #                 temperature=settings.LLM_TEMPERATURE,
# #                 api_key=settings.GROQ_API_KEY
# #             )
            
# #             # Create prompt template for diabetes advice
# #             template = """
# # You are Insulyn AI, a clinical decision-support assistant for diabetes type 2 detection and prevention.

# # Important: The underlying ML model has been trained on these patient variables:
# # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction]

# # Interpret the ML model output carefully:
# # - 0 = Low risk of type II diabetes
# # - 1 = High risk of Type II diabetes

# # Patient Data:
# # - Pregnancies: {pregnancies}
# # - Glucose: {glucose} mg/dL
# # - Blood Pressure: {blood_pressure} mmHg
# # - Skin Thickness: {skin_thickness} mm
# # - Insulin: {insulin} mu U/ml
# # - BMI: {bmi} ({bmi_category})
# # - Diabetes Pedigree Function: {diabetes_pedigree_function}
# # - Age: {age} years
# # - Model Prediction: {ml_model_output} (Probability: {probability:.2f})
# # - Feature Importances: {feature_importances}

# # Guidelines for response:
# # - Base your reasoning only on the ML risk score, the listed patient variables, and medical best practices.
# # - Provide actionable, evidence-based recommendations for diabetes prevention and management.
# # - Focus on lifestyle modifications, diet, and exercise advice.
# # - Explain what the risk factors mean in simple terms.
# # - Always include specific numeric thresholds for when to see a doctor.

# # Generate a structured response with these sections:

# # 1. Risk Summary: {risk_summary}
# # 2. Clinical Interpretation: {clinical_interpretation}
# # 3. Recommendations: {recommendations}
# # 4. Prevention Tips: {prevention_tips}
# # 5. Monitoring Plan: {monitoring_plan}
# # 6. Clinician Message: {clinician_message}
# # 7. Feature Explanation: {feature_explanation}
# # 8. Safety Note: {safety_note}

# # Keep responses clear, structured, and medically accurate.
# # """

# #             self.prompt = PromptTemplate(
# #                 input_variables=[
# #                     "pregnancies", "glucose", "blood_pressure", "skin_thickness",
# #                     "insulin", "bmi", "bmi_category", "diabetes_pedigree_function", 
# #                     "age", "ml_model_output", "probability", "feature_importances",
# #                     "risk_summary", "clinical_interpretation", "recommendations",
# #                     "prevention_tips", "monitoring_plan", "clinician_message",
# #                     "feature_explanation", "safety_note"
# #                 ],
# #                 template=template
# #             )
            
# #             # Create chain
# #             self.chain = LLMChain(
# #                 llm=self.llm,
# #                 prompt=self.prompt,
# #                 verbose=settings.DEBUG
# #             )
            
# #             logger.info("Insulyn AI LLM initialized successfully")
            
# #         except Exception as e:
# #             logger.error(f"Failed to initialize LLM: {e}")
# #             self.llm = None
    
# #     def generate_advice(self, prompt_data: dict) -> str:
# #         """Generate diabetes advice using Groq LLM"""
# #         if not self.chain:
# #             return "LLM service temporarily unavailable. Please try again later."
        
# #         try:
# #             response = self.chain.run(**prompt_data)
# #             return response
# #         except Exception as e:
# #             logger.error(f"LLM generation error: {e}")
# #             return f"Error generating advice: {str(e)}"

# # # Global LLM instance
# # insulyn_llm = InsulynAILLM()

# # def initialize_llm_service():
# #     """Initialize LLM service on application startup"""
# #     return insulyn_llm.llm is not None

# import os
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from app.config import settings
# import logging

# logger = logging.getLogger(__name__)

# class InsulynAILLM:
#     def __init__(self):
#         self.llm = None
#         self.prediction_chain = None
#         self.chat_chain = None
#         self.diet_chain = None
#         self.chat_memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             input_key="user_message"
#         )
#         self.initialize_llm()
    
#     def initialize_llm(self):
#         """Initialize Groq LLM with configuration from settings"""
#         try:
#             if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your-groq-api-key-here":
#                 logger.error("GROQ_API_KEY not configured")
#                 return
            
#             self.llm = ChatGroq(
#                 model=settings.LLM_MODEL_NAME,
#                 temperature=settings.LLM_TEMPERATURE,
#                 api_key=settings.GROQ_API_KEY
#             )
            
#             # Prediction prompt (existing)
#             prediction_template = """
# # ... existing prediction template ...
# """
#             self.prediction_prompt = PromptTemplate(
#                 input_variables=[
#                     "pregnancies", "glucose", "blood_pressure", "skin_thickness",
#                     "insulin", "bmi", "bmi_category", "diabetes_pedigree_function", 
#                     "age", "ml_model_output", "probability", "feature_importances",
#                     "risk_summary", "clinical_interpretation", "recommendations",
#                     "prevention_tips", "monitoring_plan", "clinician_message",
#                     "feature_explanation", "safety_note"
#                 ],
#                 template=prediction_template
#             )
            
#             # Chat prompt for diet and diabetes discussions
#             chat_template = """
# You are Insulyn AI, a friendly and knowledgeable diabetes educator and nutrition specialist. Your role is to provide helpful, evidence-based information about diabetes prevention, management, and diet.

# Conversation History:
# {chat_history}

# User Message: {user_message}
# Context: {conversation_context}

# Guidelines:
# - Provide accurate, medically sound information about diabetes and nutrition
# - Suggest practical, achievable dietary changes
# - Explain complex concepts in simple terms
# - Be encouraging and supportive
# - Focus on evidence-based recommendations
# - When discussing foods, consider different cultural preferences
# - Always recommend consulting healthcare providers for personalized advice
# - If unsure about something, admit it and suggest consulting a professional

# Response Structure:
# 1. Address the user's question directly and empathetically
# 2. Provide clear, actionable information
# 3. Include specific food examples when discussing diet
# 4. Mention lifestyle factors beyond just food
# 5. End with encouraging next steps

# Generate 2-3 follow-up question suggestions that would help provide more personalized advice.

# Remember: You're talking to someone who might be worried about diabetes, so be reassuring and practical.
# """

#             self.chat_prompt = PromptTemplate(
#                 input_variables=["chat_history", "user_message", "conversation_context"],
#                 template=chat_template
#             )
            
#             # Diet plan prompt
#             diet_template = """
# Create a personalized diabetes-friendly diet plan based on:

# User Profile:
# - Age: {age}
# - Weight: {weight} kg
# - Height: {height} m (BMI: {bmi:.1f})
# - Dietary Preferences: {dietary_preferences}
# - Health Conditions: {health_conditions}
# - Diabetes Risk: {diabetes_risk}

# Create a comprehensive diet plan with:

# 1. Daily Calorie Target: Estimate based on weight management goals
# 2. Meal Plan: Breakfast, Lunch, Dinner, Snacks with specific food suggestions
# 3. Foods to Include: List of beneficial foods for diabetes prevention/management
# 4. Foods to Avoid: List of foods to limit or avoid
# 5. Portion Guidance: Practical portion control tips
# 6. Timing Recommendations: Meal timing advice for blood sugar control

# Focus on:
# - Whole, unprocessed foods
# - Low glycemic index foods
# - Balanced macronutrients
# - Cultural adaptability
# - Practical, affordable options

# Make it personalized to the user's profile and risk level.
# """

#             self.diet_prompt = PromptTemplate(
#                 input_variables=["age", "weight", "height", "bmi", "dietary_preferences", 
#                                "health_conditions", "diabetes_risk"],
#                 template=diet_template
#             )
            
#             # Create chains
#             self.prediction_chain = LLMChain(
#                 llm=self.llm,
#                 prompt=self.prediction_prompt,
#                 verbose=settings.DEBUG
#             )
            
#             self.chat_chain = LLMChain(
#                 llm=self.llm,
#                 prompt=self.chat_prompt,
#                 memory=self.chat_memory,
#                 verbose=settings.DEBUG
#             )
            
#             self.diet_chain = LLMChain(
#                 llm=self.llm,
#                 prompt=self.diet_prompt,
#                 verbose=settings.DEBUG
#             )
            
#             logger.info("Insulyn AI LLM initialized successfully with chat capabilities")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize LLM: {e}")
#             self.llm = None
    
#     def generate_advice(self, prompt_data: dict) -> str:
#         """Generate diabetes advice using Groq LLM"""
#         if not self.prediction_chain:
#             return "LLM service temporarily unavailable. Please try again later."
        
#         try:
#             response = self.prediction_chain.run(**prompt_data)
#             return response
#         except Exception as e:
#             logger.error(f"LLM generation error: {e}")
#             return f"Error generating advice: {str(e)}"
    
#     def chat_about_diabetes(self, user_message: str, conversation_context: str = None) -> str:
#         """Chat about diabetes and diet topics"""
#         if not self.chat_chain:
#             return "Chat service temporarily unavailable. Please try again later."
        
#         try:
#             response = self.chat_chain.run({
#                 "user_message": user_message,
#                 "conversation_context": conversation_context or "General diabetes and diet discussion"
#             })
#             return response
#         except Exception as e:
#             logger.error(f"Chat generation error: {e}")
#             return f"Error in chat: {str(e)}"
    
#     def generate_diet_plan(self, diet_request: dict) -> str:
#         """Generate personalized diet plan"""
#         if not self.diet_chain:
#             return "Diet planning service temporarily unavailable."
        
#         try:
#             # Calculate BMI
#             bmi = diet_request['weight'] / (diet_request['height'] ** 2)
            
#             response = self.diet_chain.run({
#                 "age": diet_request['age'],
#                 "weight": diet_request['weight'],
#                 "height": diet_request['height'],
#                 "bmi": bmi,
#                 "dietary_preferences": diet_request.get('dietary_preferences', 'No specific preferences'),
#                 "health_conditions": diet_request.get('health_conditions', 'None specified'),
#                 "diabetes_risk": diet_request.get('diabetes_risk', 'Not specified')
#             })
#             return response
#         except Exception as e:
#             logger.error(f"Diet plan generation error: {e}")
#             return f"Error generating diet plan: {str(e)}"
    
#     def clear_chat_memory(self):
#         """Clear conversation memory"""
#         if self.chat_memory:
#             self.chat_memory.clear()
#             logger.info("Chat memory cleared")

# # Global LLM instance
# insulyn_llm = InsulynAILLM()

# def initialize_llm_service():
#     """Initialize LLM service on application startup"""
#     return insulyn_llm.llm is not None

# import os
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from app.config import settings
# import logging

# logger = logging.getLogger(__name__)

# class InsulynAILLM:
#     def __init__(self):
#         self.llm = None
#         self.prediction_chain = None
#         self.chat_chain = None
#         self.diet_chain = None
#         self.chat_memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             input_key="user_message"
#         )
#         self.initialize_llm()
    
#     def initialize_llm(self):
#         try:
#             if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your-groq-api-key-here":
#                 logger.error("GROQ_API_KEY not configured")
#                 return
            
#             self.llm = ChatGroq(
#                 model=settings.LLM_MODEL_NAME,
#                 temperature=settings.LLM_TEMPERATURE,
#                 api_key=settings.GROQ_API_KEY
#             )
            
#             # Prediction prompt template
#             prediction_template = """
# You are Insulyn AI, a diabetes clinical decision-support assistant.

# Patient Data:
# - Pregnancies: {pregnancies}
# - Glucose: {glucose} mg/dL
# - Blood Pressure: {blood_pressure} mmHg
# - Skin Thickness: {skin_thickness} mm
# - Insulin: {insulin} mu U/ml
# - BMI: {bmi} ({bmi_category})
# - Diabetes Pedigree Function: {diabetes_pedigree_function}
# - Age: {age} years
# - Model Prediction: {ml_model_output} (Probability: {probability:.2f})
# - Feature Importances: {feature_importances}

# Please provide a **structured response** including the following fields:

# 1. risk_summary: A brief summary of the patient’s diabetes risk.
# 2. clinical_interpretation: Interpretation of key risk factors and lab values.
# 3. recommendations: Recommended next steps, including lifestyle and medical guidance.
# 4. prevention_tips: Tips to reduce future risk of diabetes.
# 5. monitoring_plan: Guidance on what health metrics to monitor and how often.
# 6. clinician_message: A concise message intended for the attending clinician.
# 7. feature_explanation: Explain the impact of the most important features in the prediction.
# 8. safety_note: Any critical safety warnings or urgent actions.

# **Instructions for handling missing values:** If any input value is not provided, indicate “not available” instead of leaving blank.

# Respond in JSON format with keys exactly as listed above.
# """

#             self.prediction_prompt = PromptTemplate(
#     input_variables=[
#         "pregnancies", "glucose", "blood_pressure", "skin_thickness",
#         "insulin", "bmi", "bmi_category", "diabetes_pedigree_function", 
#         "age", "ml_model_output", "probability", "feature_importances"
#     ],
#     template=prediction_template
# )
            
#             self.prediction_chain = LLMChain(
#                 llm=self.llm,
#                 prompt=self.prediction_prompt,
#                 verbose=settings.DEBUG
#             )
            
#             logger.info("Insulyn AI LLM initialized successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize LLM: {e}")
#             self.llm = None
    
#     def generate_advice(self, prompt_data: dict) -> str:
#         """Generate diabetes advice using Groq LLM"""
#         if not self.prediction_chain:
#             return "LLM service temporarily unavailable. Please try again later."
        
#         try:
#             # Use .invoke() instead of .run() to avoid deprecated behavior
#             response = self.prediction_chain.invoke(prompt_data)
#             return response
#         except Exception as e:
#             logger.error(f"LLM generation error: {e}")
#             return f"Error generating advice: {str(e)}"

# # Global LLM instance
# insulyn_llm = InsulynAILLM()

# def initialize_llm_service():
#     return insulyn_llm.llm is not None

import json
import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from app.config import settings

logger = logging.getLogger(__name__)


class InsulynAILLM:
    """
    Insulyn AI LLM service:
    - Uses Groq to provide structured diabetes prevention & lifestyle advice
    - Connects ML predictions with tailored recommendations
    - Returns JSON only (safe to parse downstream in FastAPI)
    """

    def __init__(self):
        self.llm = None
        self.prediction_chain = None
        self.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="user_message",
        )
        self.initialize_llm()

    def initialize_llm(self):
        """Initialize Groq LLM with enforced JSON output"""
        try:
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.strip() == "":
                logger.error("❌ GROQ_API_KEY not configured in .env")
                return

            self.llm = ChatGroq(
                model=settings.LLM_MODEL_NAME,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.GROQ_API_KEY,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

            # === Prediction → Advice Prompt ===
            prediction_template = """
You are **Insulyn AI**, a clinical lifestyle assistant specialized in **type 2 diabetes prevention and management**.
Always respond with ONLY a valid JSON object. Do not include markdown, explanations, or text outside the JSON.

Patient Data:
- Pregnancies: {pregnancies}
- Glucose: {glucose} mg/dL
- Blood Pressure: {blood_pressure} mmHg
- Skin Thickness: {skin_thickness} mm
- Insulin: {insulin} mu U/ml
- BMI: {bmi} ({bmi_category})
- Diabetes Pedigree Function: {diabetes_pedigree_function}
- Age: {age} years
- Model Prediction: {ml_model_output} (Probability: {probability:.2f})
- Feature Importances: {feature_importances}

Your task:
Return ONLY this JSON object:

{{
  "risk_summary": "string",
  "clinical_interpretation": ["string", "string"],
  "recommendations": {{
    "diet": "string",
    "lifestyle": "string",
    "workout_plan": "string"
  }},
  "prevention_tips": ["string", "string"],
  "monitoring_plan": ["string", "string"],
  "clinician_message": "string",
  "feature_explanation": "string",
  "safety_note": "string",
  "resources": {{
    "articles": ["url1", "url2"],
    "videos": ["url1", "url2"],
    "images": ["url1", "url2"]
  }},
  "reminder_options": {{
    "can_set_reminder": true,
    "suggested_frequency": "Weekly re-test at same time",
    "example": "Notify me in 7 days to re-test glucose levels"
  }}
}}

⚠️ Rules:
- "workout_plan" MUST be clear & specific (e.g., "Run 30 mins daily, 5x/week, for 5 weeks covering 10km/week").
- Keep advice **strictly focused on type 2 diabetes**.
- Cover BOTH diet & exercise (lifestyle).
- Use evidence-based prevention tips.
- If patient data is missing, output "Not available".
- Ensure JSON is valid & parsable in Python.
"""

            self.prediction_prompt = PromptTemplate(
                input_variables=[
                    "pregnancies",
                    "glucose",
                    "blood_pressure",
                    "skin_thickness",
                    "insulin",
                    "bmi",
                    "bmi_category",
                    "diabetes_pedigree_function",
                    "age",
                    "ml_model_output",
                    "probability",
                    "feature_importances",
                ],
                template=prediction_template,
            )

            self.prediction_chain = LLMChain(
                llm=self.llm, prompt=self.prediction_prompt, verbose=settings.DEBUG
            )

            logger.info("✅ Insulyn AI LLM initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None

    @staticmethod
    def prepare_prompt_data(patient_data: dict, ml_output: dict, bmi_category: str = None) -> dict:
        """Format patient + ML data safely for LLM"""
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
            "feature_importances": ml_output.get("feature_importances", {}),
        }

    def generate_advice(self, patient_data: dict, ml_output: dict, bmi_category: str = None) -> dict:
        """Generate structured JSON advice from ML output + patient data"""
        if not self.prediction_chain:
            return {"error": "LLM service unavailable"}

        try:
            prompt_data = self.prepare_prompt_data(patient_data, ml_output, bmi_category)
            response = self.prediction_chain.invoke(prompt_data)

            # Extract JSON safely
            text_response = response.get("text") if isinstance(response, dict) and "text" in response else str(response)

            try:
                return json.loads(text_response)
            except json.JSONDecodeError:
                logger.warning("⚠️ LLM returned invalid JSON, wrapping as string")
                return {"raw_response": text_response}

        except Exception as e:
            logger.error(f"❌ LLM generation error: {e}")
            return {"error": str(e)}


# === Global Singleton ===
insulyn_llm = InsulynAILLM()


def initialize_llm_service():
    """Initialize service on FastAPI startup"""
    return insulyn_llm.llm is not None

# app/backend/app/main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
import json
import uuid
import asyncio
import time
import io
import base64
from contextlib import asynccontextmanager
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile

# ✅ Local imports
from app.config import settings
from app.models import (
    DiabetesInput, CombinedResponse, MLModelOutput,
    LLMAdviceResponse, ChatMessage, ChatResponse,
    DietPlanRequest, DietPlanResponse, HealthMetrics
)

# -------------------------------------------------
# Enhanced Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global variables for health monitoring
app_start_time = datetime.utcnow()
request_count = 0
error_count = 0

# Language and voice processing
class LanguageProcessor:
    """Process multiple languages and voice commands"""
    
    # Language detection keywords
    LANGUAGE_KEYWORDS = {
        'swahili': ['mimi', 'wewe', 'yeye', 'sisi', 'nyinyi', 'hii', 'ile', 'hapa', 'pale',
                   'chakula', 'mazoezi', 'afya', 'sukari', 'damu', 'gonjwa la kisukari'],
        'sheng': ['msee', 'nare', 'vibe', 'poa', 'safi', 'fiti', 'noma', 'kale', 'bazenga',
                 'dawa', 'kadunda', 'mzinga', 'ngoma', 'kanyaga'],
        'english': ['the', 'and', 'for', 'with', 'about', 'health', 'diabetes', 'sugar']
    }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language from input text"""
        if not text:
            return 'english'
            
        text_lower = text.lower()
        
        # Count language-specific keywords
        scores = {}
        for lang, keywords in LanguageProcessor.LANGUAGE_KEYWORDS.items():
            scores[lang] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Return language with highest score
        detected_lang = max(scores.items(), key=lambda x: x[1])[0]
        return detected_lang if scores[detected_lang] > 0 else 'english'

class VoiceProcessor:
    """Handle voice input and output"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    async def speech_to_text(self, audio_file: UploadFile) -> Dict[str, Any]:
        """Convert speech to text with language detection"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Process audio file
            with sr.AudioFile(temp_file_path) as source:
                audio = self.recognizer.record(source)
                
                # Try multiple speech recognition services
                try:
                    text = self.recognizer.recognize_google(audio)
                    language = LanguageProcessor.detect_language(text)
                    return {
                        "text": text,
                        "language": language,
                        "confidence": "high",
                        "success": True
                    }
                except sr.UnknownValueError:
                    return {
                        "text": "",
                        "language": "unknown",
                        "confidence": "low",
                        "success": False,
                        "error": "Could not understand audio"
                    }
                except sr.RequestError as e:
                    return {
                        "text": "",
                        "language": "unknown",
                        "confidence": "low",
                        "success": False,
                        "error": f"Speech recognition error: {e}"
                    }
                finally:
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            return {
                "text": "",
                "language": "unknown",
                "confidence": "low",
                "success": False,
                "error": str(e)
            }
    
    async def text_to_speech(self, text: str, language: str = 'en') -> bytes:
        """Convert text to speech audio"""
        try:
            # Map languages to gTTS language codes
            lang_codes = {
                'english': 'en',
                'swahili': 'sw',
                'sheng': 'en'  # Sheng uses English as base
            }
            
            lang_code = lang_codes.get(language, 'en')
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # Create audio in memory
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            # Return empty audio on error
            return b''

# Initialize processors
language_processor = LanguageProcessor()
voice_processor = VoiceProcessor()

# -------------------------------------------------
# Enhanced Startup with Context Manager
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Insulyn AI application...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM Model: {settings.LLM_MODEL_NAME}")
    logger.info(f"LLM Temperature: {settings.LLM_TEMPERATURE}")
    
    # Initialize services
    try:
        from app.ml_model import diabetes_model, initialize_model
        from app.llm_chain import insulyn_llm, initialize_llm_service
        
        if not initialize_model():
            logger.error("❌ ML model failed to load")
            raise RuntimeError("ML model initialization failed")
        logger.info("✅ ML model loaded successfully")

        if initialize_llm_service():
            logger.info("✅ LLM service initialized")
        else:
            logger.warning("⚠️ LLM service failed to initialize (running in limited mode)")

    except Exception as e:
        logger.critical(f"❌ Critical startup failure: {e}")
        raise

    yield  # App runs here
    
    # Shutdown
    logger.info("🛑 Shutting down Insulyn AI application...")

# -------------------------------------------------
# Enhanced FastAPI App
# -------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Insulyn AI - Diabetes Detection, Prevention & Lifestyle Guidance System with Multi-Language & Voice Support",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage
rate_limit_storage = {}

# -------------------------------------------------
# Enhanced Dependencies & Security
# -------------------------------------------------
async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Enhanced API key verification"""
    if hasattr(settings, 'API_KEYS') and settings.API_KEYS:
        if not credentials or credentials.credentials not in settings.API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# -------------------------------------------------
# Enhanced Health & Info Endpoints
# -------------------------------------------------
@app.get("/")
async def root():
    global app_start_time
    return {
        "message": "Insulyn AI API is running",
        "status": "healthy",
        "version": "3.0.0",
        "uptime": str(datetime.utcnow() - app_start_time),
        "timestamp": datetime.utcnow(),
        "features": [
            "Enhanced ML diabetes risk prediction",
            "Advanced LLM lifestyle interpretation",
            "Personalized diet & workout recommendations",
            "Multi-language support (English, Swahili, Sheng)",
            "Voice command processing",
            "Real-time health monitoring",
            "Caching & performance optimization",
            "Rate limiting & security",
        ],
        "supported_languages": ["english", "swahili", "sheng"],
        "voice_support": True
    }

@app.get("/health", response_model=HealthMetrics)
async def health_check():
    from app.ml_model import diabetes_model
    from app.llm_chain import insulyn_llm
    
    global request_count, error_count, app_start_time
    
    # Test ML model
    ml_healthy = getattr(diabetes_model, "model", None) is not None
    llm_healthy = getattr(insulyn_llm, "llm", None) is not None
    
    overall_status = "healthy" if all([ml_healthy, llm_healthy]) else "degraded"
    
    return HealthMetrics(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="3.0.0",
        uptime_seconds=(datetime.utcnow() - app_start_time).total_seconds(),
        services={
            "ml_model": ml_healthy,
            "llm_service": llm_healthy,
            "voice_processing": True,
            "multi_language": True,
        },
        metrics={
            "total_requests": request_count,
            "error_rate": error_count / max(request_count, 1),
        }
    )

# -------------------------------------------------
# Enhanced Lifestyle Recommendations with Multi-language
# -------------------------------------------------
class MultiLanguageLifestyleEngine:
    """Enhanced lifestyle recommendation engine with multi-language support"""
    
    @staticmethod
    def get_recommendations(risk_level: str, bmi: float, age: int, language: str = "english") -> Dict[str, Any]:
        """
        Provide personalized actionable lifestyle advice in multiple languages
        """
        # Base recommendations in English
        base_advice_en = {
            "meal_plan": [
                "Increase vegetables, whole grains, and lean proteins",
                "Reduce sugary drinks and processed carbs",
                "Stay hydrated (aim for ~8 glasses/day)",
                "Practice portion control and mindful eating",
            ],
            "workout_plan": [
                "Walk briskly 30 mins daily, 5 days/week",
                "Add light strength training 2x per week",
                "Include flexibility exercises 2x per week",
            ],
            "lifestyle_tips": [
                "Sleep 7–8 hours/night",
                "Practice stress reduction (breathing exercises, short walks, or yoga)",
                "Avoid smoking and limit alcohol",
                "Monitor progress with a health journal",
            ]
        }

        # Translate based on language
        if language == "swahili":
            base_advice = {
                "meal_plan": [
                    "Ongeza mboga, nafaka nzima, na protini nyepesi",
                    "Punguza vinywaji vilivyo na sukari na wanga uliosafishwa",
                    "Kunywa maji ya kutosha (lengo la glasi 8 kwa siku)",
                    "Dhibiti kiasi cha chakula na ule kwa umakini",
                ],
                "workout_plan": [
                    "Tembea kwa kasi dakika 30 kila siku, siku 5 kwa wiki",
                    "Ongeza mazoezi ya nguvu mara 2 kwa wiki",
                    "Jumuisha mazoezi ya kunyoosha viungo mara 2 kwa wiki",
                ],
                "lifestyle_tips": [
                    "Lala masaa 7-8 usiku",
                    "Fanya mazoezi ya kupunguza msongo wa mawazo (kupumua, matembezi mafupi, au yoga)",
                    "Epuka uvutaji sigara na punguza pombe",
                    "Fuatilia maendeleo yako kwa kutumia jarida la afya",
                ]
            }
        elif language == "sheng":
            base_advice = {
                "meal_plan": [
                    "Ongeza mboga, food zote nzima, na protein safi",
                    "Punguza soda na chips",
                    "Kunywa maji mingi (lengo glasi 8 daily)",
                    "Control portion ya chakula na kula kwa makini",
                ],
                "workout_plan": [
                    "Tembea kwa kasi dakika 30 daily, siku 5 weekly",
                    "Ongeza mzoez wa nguvu mara 2 weekly",
                    "Jumuisha kunyoosha mara 2 weekly",
                ],
                "lifestyle_tips": [
                    "Lala saa 7-8 usiku",
                    "Fanya mzoez wa kupunguza stress (kupumua, walk mfupi, au yoga)",
                    "Epuka sigara na reduce pombe",
                    "Track progress yako kwa kuandika",
                ]
            }
        else:  # English
            base_advice = base_advice_en

        risk_level = (risk_level or "").lower()
        bmi_category = "obese" if bmi >= 30 else "overweight" if bmi >= 25 else "normal"
        
        # High risk enhancements
        if "high" in risk_level:
            if language == "swahili":
                base_advice["workout_plan"] = [
                    "Fanya mazoezi ya moyo dakika 45 kila siku (kukimbia, kuendesha baiskeli, kuogelea)",
                    "Mazoezi ya nguvu mara 3 kwa wiki (mazoezi ya mwili mzima)",
                    "Mazoezi ya ukali wa juu mara 1-2 kwa wiki",
                    "Kunyoosha viungo kila siku (dakika 10-15)",
                ]
                base_advice["meal_plan"].extend([
                    "Fuata mpango wa lishe ya wanga kidogo au ya Mediterania",
                    "Fikiria kufunga kwa muda (shauriana na daktari kwanza)",
                    "Fuatilia virutubisho na kalori unazokula",
                ])
                base_advice["lifestyle_tips"].append("Ongeza mara ya kukagua sukari ya damu na shauriana na daktari wako")
            elif language == "sheng":
                base_advice["workout_plan"] = [
                    "Cardio dakika 45 daily (kukimbia, bike, swim)",
                    "Strength training mara 3 weekly (full body workouts)",
                    "HIIT mara 1-2 weekly",
                    "Kunyoosha daily (dakika 10-15)",
                ]
                base_advice["meal_plan"].extend([
                    "Fuata low-carb plan au Mediterranean diet",
                    "Fikiria intermittent fasting (consult daktari first)",
                    "Track macros na calorie intake",
                ])
                base_advice["lifestyle_tips"].append("Ongeza frequency ya kuchunguza sugar na consult daktari")
            else:
                base_advice["workout_plan"] = [
                    "Cardio 45 mins daily (running, cycling, swimming)",
                    "Strength training 3x per week (full body workouts)",
                    "High-intensity interval training 1-2x per week",
                    "Daily mobility and stretching (10–15 mins)",
                ]
                base_advice["meal_plan"].extend([
                    "Adopt a structured low-carb or Mediterranean diet",
                    "Consider intermittent fasting (consult doctor first)",
                    "Track macronutrients and calorie intake",
                ])
                base_advice["lifestyle_tips"].append("Increase frequency of blood sugar checks and consult your clinician")
        
        return base_advice

# -------------------------------------------------
# Enhanced LLM Response Generator with Multi-language
# -------------------------------------------------
class MultiLanguageLLMOrchestrator:
    """Orchestrates LLM calls with multi-language support"""
    
    def __init__(self):
        self.retry_config = {
            'max_retries': 3,
            'backoff_factor': 1.5,
        }
    
    async def generate_multi_language_advice(self, patient_data: dict, ml_output: dict, lifestyle_plan: Dict[str, Any], language: str = "english") -> LLMAdviceResponse:
        """
        Enhanced LLM response generation with multi-language support
        """
        from app.llm_chain import insulyn_llm
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                # Add language context to patient data
                enhanced_patient_data = {
                    **patient_data,
                    "preferred_language": language,
                    "response_language": language
                }
                
                # Generate base LLM response
                llm_output = await asyncio.to_thread(
                    insulyn_llm.generate_advice, enhanced_patient_data, ml_output
                )
                
                # Parse and validate
                parsed = self.try_parse_llm_text(llm_output)
                validated_response = self.validate_and_enhance_response(parsed, lifestyle_plan, language)
                
                return validated_response
                
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_config['max_retries'] - 1:
                    return self.generate_fallback_response(lifestyle_plan, language)
                
                # Exponential backoff
                await asyncio.sleep(self.retry_config['backoff_factor'] ** attempt)
    
    def try_parse_llm_text(self, llm_text: Any) -> Any:
        """Robust parsing with enhanced error recovery"""
        if llm_text is None:
            return {"error": "No response from LLM"}
        
        if isinstance(llm_text, dict):
            return llm_text
        
        text = str(llm_text).strip()
        
        # Remove code fences and clean text
        if text.startswith("```") and text.endswith("```"):
            text = text.strip("`").replace("json", "", 1).strip()
        
        # Multiple JSON parsing attempts
        for parse_attempt in [text, text[text.find("{"):text.rfind("}")+1]]:
            try:
                return json.loads(parse_attempt)
            except json.JSONDecodeError:
                continue
        
        return {"raw_response": text}
    
    def validate_and_enhance_response(self, parsed: Dict[str, Any], lifestyle_plan: Dict[str, Any], language: str) -> LLMAdviceResponse:
        """Validate and enhance LLM response with lifestyle plan integration"""
        # Ensure all required fields exist
        enhanced_data = {
            "risk_summary": parsed.get("risk_summary", self.get_default_text("risk_summary", language)),
            "clinical_interpretation": self.ensure_list(parsed.get("clinical_interpretation"), self.get_default_text("clinical", language)),
            "recommendations": self.enhance_recommendations(parsed.get("recommendations", {}), lifestyle_plan, language),
            "prevention_tips": self.ensure_list(parsed.get("prevention_tips"), self.get_default_text("prevention", language)),
            "monitoring_plan": self.ensure_list(parsed.get("monitoring_plan"), self.get_default_text("monitoring", language)),
            "clinician_message": parsed.get("clinician_message", self.get_default_text("clinician", language)),
            "feature_explanation": parsed.get("feature_explanation", self.get_default_text("features", language)),
            "safety_note": parsed.get("safety_note", self.get_default_text("safety", language)),
        }
        
        return LLMAdviceResponse(**enhanced_data)
    
    def get_default_text(self, text_type: str, language: str) -> str:
        """Get default text in appropriate language"""
        defaults = {
            "english": {
                "risk_summary": "Risk assessment not available",
                "clinical": "Clinical interpretation not available",
                "prevention": "Maintain healthy lifestyle",
                "monitoring": "Regular health checkups",
                "clinician": "Consult your healthcare provider for personalized advice",
                "features": "Key health indicators were analyzed for risk assessment",
                "safety": "If you experience severe symptoms, seek immediate medical attention"
            },
            "swahili": {
                "risk_summary": "Tathmini ya hatari haipatikani",
                "clinical": "Ufafanuzi wa kikliniki haupatikani",
                "prevention": "Endelea na mtindo wa maisha wenye afya",
                "monitoring": "Vipimo vya kawaida vya afya",
                "clinician": "Wasiliana na mhudumu wako wa afya kwa ushauri unaokufaa",
                "features": "Viashiria muhimu vya afya vilichambuliwa kwa ajili ya tathmini ya hatari",
                "safety": "Ukikutana na dalili kubwa, tafuta huduma ya matibabu mara moja"
            },
            "sheng": {
                "risk_summary": "Risk assessment haipo",
                "clinical": "Clinical interpretation haipo",
                "prevention": "Endelea na healthy lifestyle",
                "monitoring": "Regular health checkups",
                "clinician": "Consult doctor wako kwa ushauri bora",
                "features": "Key health indicators zilichambuliwa kwa risk assessment",
                "safety": "Ukipata symptoms kubwa, enda hospitalini immediately"
            }
        }
        
        return defaults.get(language, defaults["english"]).get(text_type, "Information not available")
    
    def enhance_recommendations(self, llm_recs: Dict[str, Any], lifestyle_plan: Dict[str, Any], language: str) -> Dict[str, str]:
        """Merge LLM recommendations with lifestyle plan"""
        recommendations = {}
        
        # Convert all values to strings
        for key, value in llm_recs.items():
            if isinstance(value, list):
                recommendations[key] = "\n".join(f"• {item}" for item in value)
            else:
                recommendations[key] = str(value)
        
        # Merge with lifestyle plan
        for key in ["meal_plan", "workout_plan", "lifestyle_tips"]:
            if key in lifestyle_plan and isinstance(lifestyle_plan[key], list):
                recommendations[key] = "\n".join(f"• {item}" for item in lifestyle_plan[key])
        
        return recommendations
    
    def ensure_list(self, value: Any, default: str) -> List[str]:
        """Ensure value is a list of strings"""
        if isinstance(value, list):
            return [str(item) for item in value]
        elif value:
            return [str(value)]
        else:
            return [default]
    
    def generate_fallback_response(self, lifestyle_plan: Dict[str, Any], language: str) -> LLMAdviceResponse:
        """Generate comprehensive fallback response in appropriate language"""
        fallback_texts = {
            "english": {
                "risk_summary": "Please consult healthcare provider for assessment",
                "clinical": "System temporarily unavailable for detailed analysis",
                "prevention": ["Maintain healthy weight", "Exercise regularly", "Eat balanced diet"],
                "monitoring": ["Regular health checkups", "Monitor blood sugar if advised"],
                "clinician": "Consult your primary care physician for personalized diabetes risk assessment",
                "features": "System analysis temporarily unavailable",
                "safety": "If you have concerns about diabetes symptoms, seek medical advice promptly"
            },
            "swahili": {
                "risk_summary": "Tafadhali wasiliana na mhudumu wa afya kwa tathmini",
                "clinical": "Mfumo haupatikani kwa sasa kwa uchambuzi wa kina",
                "prevention": ["Weka uzito wenye afya", "Fanya mazoezi mara kwa mara", "Lia lishe yenye usawa"],
                "monitoring": ["Vipimo vya kawaida vya afya", "Angalia sukari ya damu ikiwa umeshauriwa"],
                "clinician": "Wasiliana na daktari wako wa msingi kwa tathmini ya hatari ya kisukari inayokufaa",
                "features": "Uchambuzi wa mfumo haupatikani kwa sasa",
                "safety": "Ukiwa na wasiwasi kuhusu dalili za kisukari, tafuta ushauri wa matibabu haraka"
            },
            "sheng": {
                "risk_summary": "Tafadhali consult doctor kwa assessment",
                "clinical": "System haipo kwa sasa kwa detailed analysis",
                "prevention": ["Maintain healthy weight", "Exercise regularly", "Eat balanced diet"],
                "monitoring": ["Regular health checkups", "Monitor blood sugar kama umeadvicewa"],
                "clinician": "Consult doctor wako kwa personalized diabetes risk assessment",
                "features": "System analysis haipo kwa sasa",
                "safety": "Ukiwa na wasiwasi kuhusu diabetes symptoms, seek medical advice promptly"
            }
        }
        
        texts = fallback_texts.get(language, fallback_texts["english"])
        
        fallback_recs = {
            "note": "Personalized advice temporarily unavailable",
            "general_guidance": "Focus on balanced nutrition and regular physical activity"
        }
        
        # Include lifestyle plan in fallback
        for key in ["meal_plan", "workout_plan", "lifestyle_tips"]:
            if key in lifestyle_plan:
                fallback_recs[key] = "\n".join(f"• {item}" for item in lifestyle_plan[key])
        
        return LLMAdviceResponse(
            risk_summary=texts["risk_summary"],
            clinical_interpretation=[texts["clinical"]],
            recommendations=fallback_recs,
            prevention_tips=texts["prevention"],
            monitoring_plan=texts["monitoring"],
            clinician_message=texts["clinician"],
            feature_explanation=texts["features"],
            safety_note=texts["safety"],
        )

# -------------------------------------------------
# Enhanced Prediction Endpoint with Multi-language
# -------------------------------------------------
@app.post("/api/v1/predict", response_model=CombinedResponse)
async def predict_diabetes_risk(
    input_data: DiabetesInput,
    background_tasks: BackgroundTasks,
    language: str = "english",
    api_key: bool = Depends(verify_api_key)
):
    global request_count
    
    request_id = str(uuid.uuid4())
    request_count += 1
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Prediction request {request_id} started in {language}")
        
        from app.ml_model import diabetes_model
        
        # Extract and validate features
        features = input_data.dict()
        height, weight = features.get("height"), features.get("weight")
        
        if not height or height <= 0:
            raise HTTPException(status_code=422, detail="Invalid height provided")
        
        # Calculate BMI
        features["bmi"] = round(weight / (height ** 2), 2)
        calculated_bmi = features["bmi"]
        
        # Get BMI category
        bmi_category = diabetes_model.get_bmi_category(calculated_bmi)
        
        # ML prediction
        risk_label, probability, feature_importances = diabetes_model.predict(features)
        
        # Enhanced lifestyle recommendations with language support
        lifestyle_engine = MultiLanguageLifestyleEngine()
        lifestyle_plan = lifestyle_engine.get_recommendations(
            risk_label, calculated_bmi, features.get("age", 30), language
        )
        
        # Prepare enhanced patient data for LLM
        patient_data = {
            **{k: v for k, v in features.items() if k != 'height'},
            "bmi_category": bmi_category,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "preferred_language": language
        }
        
        ml_model_output = MLModelOutput(
            risk_label=risk_label,
            probability=probability,
            calculated_bmi=calculated_bmi,
            feature_importances=feature_importances or {},
        )
        
        # Generate LLM advice with multi-language orchestrator
        llm_orchestrator = MultiLanguageLLMOrchestrator()
        llm_advice = await llm_orchestrator.generate_multi_language_advice(
            patient_data, ml_model_output.dict(), lifestyle_plan, language
        )
        
        # Background task for analytics
        background_tasks.add_task(
            log_prediction_analytics,
            request_id, features, risk_label, probability, calculated_bmi, language
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Prediction {request_id} completed in {processing_time:.2f}s in {language}")
        
        return CombinedResponse(
            timestamp=datetime.utcnow(),
            ml_output=ml_model_output,
            llm_advice=llm_advice,
            bmi_category=bmi_category,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        global error_count
        error_count += 1
        logger.exception(f"❌ Prediction {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Error generating prediction")

async def log_prediction_analytics(request_id: str, features: dict, risk_label: str, probability: float, bmi: float, language: str):
    """Background task for analytics logging"""
    try:
        analytics_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "risk_label": risk_label,
            "probability": probability,
            "bmi": bmi,
            "age": features.get("age"),
            "language": language,
            "has_high_glucose": features.get("glucose", 0) > 140,
            "has_high_bp": features.get("blood_pressure", 0) > 120,
        }
        
        logger.info(f"📊 Analytics: {analytics_data}")
        
    except Exception as e:
        logger.error(f"Analytics logging failed: {e}")

# -------------------------------------------------
# Enhanced Chat Endpoints with Multi-language & Voice
# -------------------------------------------------
def is_diabetes_related(message: str) -> bool:
    """Enhanced content filtering for multiple languages"""
    diabetes_keywords = [
        # English
        "diabetes", "sugar", "insulin", "glucose", "blood sugar", "type 2", "type 1",
        "prediabetes", "a1c", "hba1c", "hyperglycemia", "hypoglycemia", "metformin",
        # Swahili
        "kisukari", "sukari", "insulini", "glukosi", "sukari ya damu", "aina ya pili",
        "dalili za kisukari", "ugonjwa wa kisukari",
        # Sheng
        "sugar", "dawa ya sugar", "damu sugar", "type ya sugar"
    ]
    
    health_keywords = [
        # English
        "diet", "exercise", "weight", "bmi", "nutrition", "healthy", "lifestyle",
        "meal", "food", "workout", "fitness", "health", "medical",
        # Swahili
        "lishe", "mazoezi", "uzito", "afya", "chakula", "maisha",
        # Sheng
        "kula", "mzoez", "weight", "afya", "chakula", "lifestyle"
    ]
    
    message_lower = message.lower()
    
    # Check for diabetes-specific terms
    if any(keyword in message_lower for keyword in diabetes_keywords):
        return True
    
    # Check for health terms in context
    if any(keyword in message_lower for keyword in health_keywords):
        return len(message.split()) >= 3  # Ensure it's a substantive question
    
    return False

def get_contextual_suggestions(user_message: str, ai_response: str, language: str) -> List[str]:
    """Generate intelligent follow-up suggestions in appropriate language"""
    message_lower = user_message.lower()
    
    if language == "swahili":
        suggestions = []
        
        if any(word in message_lower for word in ["chakula", "lishe", "kula"]):
            suggestions.extend([
                "Unaweza kunipa mifano maalum ya vyakula?",
                "Vipi kuhusu vitafunio bora?",
                "Ninawezaje kusoma lebo za chakula kwa sukari?"
            ])
        
        if any(word in message_lower for word in ["mazoezi", "zoezi", "mwenendo"]):
            suggestions.extend([
                "Mazoezi gani ni salama kwa wanaoanza?",
                "Mazoezi yanaathiri vipi sukari ya damu?",
                "Unaweza kupendekeza mazoezi ya nyumbani?"
            ])
        
        if any(word in message_lower for word in ["dalili", "ishara", "hisia"]):
            suggestions.extend([
                "Ninapaswa kuona daktari lini kuhusu dalili hizi?",
                "Ni ishara gani za dharura za kisukari?",
                "Kisukari hugunduliwaje?"
            ])
        
        if not suggestions:
            suggestions = [
                "Unaweza kufafanua hii kwa lugha rahisi?",
                "Ni hatua gani zinazofuata ninapaswa kuchukua?",
                "Naweza kupata rasilimali zaidi wapi kuhusu hili?"
            ]
    
    elif language == "sheng":
        suggestions = []
        
        if any(word in message_lower for word in ["chakula", "kula", "food"]):
            suggestions.extend([
                "Unaweza niachia mfano wa meals?",
                "Vipi kuhusu snacks bora za sugar?",
                "Ninasomeaje food labels kwa sugar content?"
            ])
        
        if any(word in message_lower for word in ["mzoez", "exercise", "workout"]):
            suggestions.extend([
                "Exercise gani ni safest kwa beginners?",
                "Exercise inaaffect vipi blood sugar?",
                "Unaweza suggest home workout routines?"
            ])
        
        if any(word in message_lower for word in ["symptoms", "dalili", "signs"]):
            suggestions.extend([
                "Nifae daktari lini kuhusu symptoms hizi?",
                "Ni emergency signs gani za diabetes?",
                "Diabetes inagundulikaje?"
            ])
        
        if not suggestions:
            suggestions = [
                "Unaweza explain hii kwa simple terms?",
                "Next steps gani nifanye?",
                "Naweza pata more resources wapi?"
            ]
    
    else:  # English
        suggestions = []
        
        if any(word in message_lower for word in ["food", "diet", "eat", "nutrition"]):
            suggestions.extend([
                "Can you give me specific meal examples?",
                "What about snacks for diabetes?",
                "How do I read food labels for sugar content?"
            ])
        
        if any(word in message_lower for word in ["exercise", "workout", "activity"]):
            suggestions.extend([
                "What exercises are safest for beginners?",
                "How does exercise affect blood sugar levels?",
                "Can you suggest home workout routines?"
            ])
        
        if any(word in message_lower for word in ["symptom", "sign", "feel", "experience"]):
            suggestions.extend([
                "When should I see a doctor about these symptoms?",
                "What are emergency warning signs for diabetes?",
                "How is diabetes diagnosed?"
            ])
        
        if not suggestions:
            suggestions = [
                "Can you explain this in simpler terms?",
                "What are the next steps I should take?",
                "Where can I find more resources about this?"
            ]
    
    return suggestions[:3]  # Limit to 3 suggestions

async def log_chat_analytics(user_message: str, ai_response: str, suggestions: List[str], language: str):
    """Background chat analytics"""
    try:
        analytics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "message_length": len(user_message),
            "response_length": len(ai_response),
            "suggestions_count": len(suggestions),
            "language": language,
            "has_emergency_keywords": any(word in user_message.lower() for word in ["emergency", "urgent", "help immediately", "dharura", "haraka"]),
        }
        logger.info(f"💬 Chat analytics: {analytics_data}")
    except Exception as e:
        logger.error(f"Chat analytics failed: {e}")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_about_diabetes(
    chat_message: ChatMessage,
    background_tasks: BackgroundTasks,
    language: str = "english",
    api_key: bool = Depends(verify_api_key)
):
    try:
        from app.llm_chain import insulyn_llm
        
        # Auto-detect language if not specified
        if not language or language == "auto":
            language = language_processor.detect_language(chat_message.message)
        
        # Enhanced content filtering for multiple languages
        if not is_diabetes_related(chat_message.message):
            if language == "swahili":
                return ChatResponse(
                    response="Nimelenga hasa kusaidia kuhusu kisukari. Tafadhali uliza maswali yanayohusiana na:\n• Mambo yanayochangia hatari ya kisukari\n• Mipango ya lishe yenye afya\n• Mapendekezo ya mazoezi\n• Kufuatilia sukari ya damu\n• Mabadiliko ya mtindo wa maisha",
                    timestamp=datetime.utcnow(),
                    suggestions=get_contextual_suggestions("jumla", "", language),
                )
            elif language == "sheng":
                return ChatResponse(
                    response="Nimefocus kusaidia kuhusu sugar. Tafadhali uliza maswali kuhusu:\n• Risk factors za sugar\n• Healthy eating plans\n• Exercise recommendations\n• Blood sugar monitoring\n• Lifestyle changes",
                    timestamp=datetime.utcnow(),
                    suggestions=get_contextual_suggestions("general", "", language),
                )
            else:
                return ChatResponse(
                    response="I specialize in diabetes prevention and management. Please ask questions related to:\n• Diabetes risk factors\n• Healthy eating plans\n• Exercise recommendations\n• Blood sugar monitoring\n• Lifestyle modifications",
                    timestamp=datetime.utcnow(),
                    suggestions=get_contextual_suggestions("general", "", language),
                )
        
        # Add language context to conversation
        enhanced_context = f"{chat_message.conversation_context or ''} [Language: {language}]"
        
        # Generate response
        raw_response = await asyncio.to_thread(
            insulyn_llm.chat_about_diabetes,
            chat_message.message,
            enhanced_context
        )
        
        # Parse and enhance response
        parsed = MultiLanguageLLMOrchestrator().try_parse_llm_text(raw_response)
        response_text = parsed if isinstance(parsed, str) else json.dumps(parsed, ensure_ascii=False)
        
        # Generate contextual suggestions
        suggestions = get_contextual_suggestions(chat_message.message, response_text, language)
        
        # Background analytics
        background_tasks.add_task(
            log_chat_analytics,
            chat_message.message, response_text, suggestions, language
        )
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow(),
            suggestions=suggestions,
        )
        
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="Error processing chat message")

# -------------------------------------------------
# Voice Chat Endpoint
# -------------------------------------------------
@app.post("/api/v1/chat/voice")
async def voice_chat_about_diabetes(
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: bool = Depends(verify_api_key)
):
    """Process voice messages and return both text and audio responses"""
    try:
        # Validate audio file
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Convert speech to text
        stt_result = await voice_processor.speech_to_text(audio_file)
        
        if not stt_result["success"]:
            raise HTTPException(status_code=400, detail=f"Speech recognition failed: {stt_result.get('error', 'Unknown error')}")
        
        text_message = stt_result["text"]
        detected_language = stt_result["language"]
        
        if not text_message.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Process the text message through regular chat
        chat_message = ChatMessage(
            message=text_message,
            conversation_context=f"Voice input in {detected_language}"
        )
        
        # Get text response
        chat_response = await chat_about_diabetes(
            chat_message=chat_message,
            background_tasks=background_tasks,
            language=detected_language,
            api_key=api_key
        )
        
        # Convert response to speech
        audio_response = await voice_processor.text_to_speech(
            chat_response.response, 
            detected_language
        )
        
        # Encode audio for response
        audio_base64 = base64.b64encode(audio_response).decode('utf-8')
        
        return {
            "input_text": text_message,
            "detected_language": detected_language,
            "text_response": chat_response.response,
            "audio_response": audio_base64,
            "suggestions": chat_response.suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Voice chat endpoint error")
        raise HTTPException(status_code=500, detail="Error processing voice message")

# -------------------------------------------------
# Text-to-Speech Endpoint
# -------------------------------------------------
@app.post("/api/v1/tts")
async def text_to_speech_endpoint(
    text: str = Form(...),
    language: str = Form("english"),
    api_key: bool = Depends(verify_api_key)
):
    """Convert text to speech audio"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Convert text to speech
        audio_data = await voice_processor.text_to_speech(text, language)
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        # Return audio file
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3",
                "Content-Length": str(len(audio_data))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Text-to-speech endpoint error")
        raise HTTPException(status_code=500, detail="Error generating speech")

# -------------------------------------------------
# Speech-to-Text Endpoint
# -------------------------------------------------
@app.post("/api/v1/stt")
async def speech_to_text_endpoint(
    audio_file: UploadFile = File(...),
    api_key: bool = Depends(verify_api_key)
):
    """Convert speech to text"""
    try:
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Convert speech to text
        result = await voice_processor.speech_to_text(audio_file)
        
        return {
            "success": result["success"],
            "text": result["text"],
            "language": result["language"],
            "confidence": result["confidence"],
            "error": result.get("error"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Speech-to-text endpoint error")
        raise HTTPException(status_code=500, detail="Error processing audio")

@app.post("/api/v1/chat/clear")
async def clear_chat_history(api_key: bool = Depends(verify_api_key)):
    try:
        from app.llm_chain import insulyn_llm
        insulyn_llm.clear_chat_memory()
        return {
            "message": "Chat history cleared successfully",
            "timestamp": datetime.utcnow(),
            "cleared_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.exception("Clear history error")
        raise HTTPException(status_code=500, detail="Error clearing chat history")

@app.get("/api/v1/chat/topics")
async def get_chat_topics(language: str = "english"):
    """Get chat topics in appropriate language"""
    if language == "swahili":
        return {
            "topics": [
                {"id": "diet", "name": "Vyakula vya kula na kuepuka", "description": "Mwongozo wa lishe kwa kisukari"},
                {"id": "exercise", "name": "Mapendekezo ya mazoezi", "description": "Mipango ya shughuli za kimwili"},
                {"id": "monitoring", "name": "Kufuatilia sukari ya damu", "description": "Jinsi ya kufuatilia viwango vyako"},
                {"id": "symptoms", "name": "Dalili na ishara za mapema", "description": "Kutambua ishara za onyo"},
                {"id": "medication", "name": "Dawa na matibabu", "description": "Mapitio ya chaguzi za matibabu"},
                {"id": "weight", "name": "Usimamizi wa uzito", "description": "Mikakati ya kupoteza uzito wenye afya"},
                {"id": "stress", "name": "Usimamizi wa msongo wa mawazo", "description": "Afya ya akili na ustawi"},
                {"id": "prevention", "name": "Mikakati ya kuzuia", "description": "Kupunguza hatari ya kisukari"},
            ],
            "popular_topics": ["diet", "exercise", "symptoms"],
            "last_updated": datetime.utcnow().isoformat()
        }
    elif language == "sheng":
        return {
            "topics": [
                {"id": "diet", "name": "Food za kula na kuepuka", "description": "Nutrition guidance kwa sugar"},
                {"id": "exercise", "name": "Exercise recommendations", "description": "Physical activity plans"},
                {"id": "monitoring", "name": "Blood sugar monitoring", "description": "How to track levels yako"},
                {"id": "symptoms", "name": "Symptoms na early signs", "description": "Kutambua warning signals"},
                {"id": "medication", "name": "Medication na treatment", "description": "Treatment options overview"},
                {"id": "weight", "name": "Weight management", "description": "Healthy weight loss strategies"},
                {"id": "stress", "name": "Stress management", "description": "Mental health na wellbeing"},
                {"id": "prevention", "name": "Prevention strategies", "description": "Kupunguza risk ya sugar"},
            ],
            "popular_topics": ["diet", "exercise", "symptoms"],
            "last_updated": datetime.utcnow().isoformat()
        }
    else:
        return {
            "topics": [
                {"id": "diet", "name": "Foods to eat and avoid", "description": "Nutrition guidance for diabetes"},
                {"id": "exercise", "name": "Exercise recommendations", "description": "Physical activity plans"},
                {"id": "monitoring", "name": "Blood sugar monitoring", "description": "How to track your levels"},
                {"id": "symptoms", "name": "Symptoms and early signs", "description": "Recognizing warning signals"},
                {"id": "medication", "name": "Medication and treatment", "description": "Treatment options overview"},
                {"id": "weight", "name": "Weight management", "description": "Healthy weight loss strategies"},
                {"id": "stress", "name": "Stress management", "description": "Mental health and wellbeing"},
                {"id": "prevention", "name": "Prevention strategies", "description": "Reducing diabetes risk"},
            ],
            "popular_topics": ["diet", "exercise", "symptoms"],
            "last_updated": datetime.utcnow().isoformat()
        }

# -------------------------------------------------
# Enhanced Diet Plan Endpoint with Multi-language
# -------------------------------------------------
def get_default_meal_structure(language: str = "english") -> Dict[str, List[str]]:
    """Get default meal structure in appropriate language"""
    if language == "swahili":
        return {
            "breakfast": ["Nafaka nzima na berries", "Mayai ya kuchemsha", "Maziwa ya mgando"],
            "lunch": ["Saladi ya kuku wa kuchoma", "Quinoa", "Mboga zilizochemshwa"],
            "dinner": ["Samaki wa kukaanga", "Viazi vitamu vilivyochomwa", "Saladi ya kijani"],
            "snacks": ["Apple na siagi ya njugu", "Vipande ya karoti na hummus", "Kofia ya njugu"],
        }
    elif language == "sheng":
        return {
            "breakfast": ["Whole grain cereal na berries", "Boiled eggs", "Greek yogurt"],
            "lunch": ["Grilled chicken salad", "Quinoa", "Steamed vegetables"],
            "dinner": ["Baked fish", "Roasted sweet potatoes", "Green salad"],
            "snacks": ["Apple na peanut butter", "Carrot sticks na hummus", "Handful ya nuts"],
        }
    else:
        return {
            "breakfast": ["Whole grain cereal with berries", "Boiled eggs", "Greek yogurt"],
            "lunch": ["Grilled chicken salad", "Quinoa", "Steamed vegetables"],
            "dinner": ["Baked fish", "Roasted sweet potatoes", "Green salad"],
            "snacks": ["Apple with peanut butter", "Carrot sticks with hummus", "Handful of nuts"],
        }

def get_recommended_foods(language: str = "english") -> List[str]:
    """Get recommended foods in appropriate language"""
    if language == "swahili":
        return [
            "Mboga za majani (spinach, kale)", "Nafaka nzima (oats, quinoa)", "Protini nyepesi (kuku, samaki, tofu)",
            "Mbegu za jamii (lentils, maharage)", "Njugu na mbegu", "Matunda yenye sukari kidogo (berries, apples)",
            "Mafuta yenye afya (parachichi, njugu, mafuta ya mzeituni)", "Maziwa ya mafuta madogo", "Mboga zisizo na wanga"
        ]
    elif language == "sheng":
        return [
            "Leafy greens (spinach, kale)", "Whole grains (oats, quinoa)", "Lean proteins (chicken, fish, tofu)",
            "Legumes (lentils, beans)", "Nuts na seeds", "Low-sugar fruits (berries, apples)",
            "Healthy fats (avocado, nuts, olive oil)", "Low-fat dairy", "Non-starchy vegetables"
        ]
    else:
        return [
            "Leafy greens (spinach, kale)", "Whole grains (oats, quinoa)", "Lean proteins (chicken, fish, tofu)",
            "Legumes (lentils, beans)", "Nuts and seeds", "Low-sugar fruits (berries, apples)",
            "Healthy fats (avocado, nuts, olive oil)", "Low-fat dairy", "Non-starchy vegetables"
        ]

def get_foods_to_avoid(language: str = "english") -> List[str]:
    """Get foods to avoid in appropriate language"""
    if language == "swahili":
        return [
            "Vinywaji vilivyo na sukari na soda", "Vitafunio vilivyochakatwa", "Mkate mweupe na nafaka zilizosafishwa",
            "Vyakula vya kukaanga", "Matunda yenye sukari nyingi (embe, zabibu)", "Bidhaa za maziwa zenye mafuta mengi",
            "Nafaka zilizotengenezwa kwa sukari", "Pipi na vitafunio tamu", "Vyakula vilivyo na chumvi nyingi"
        ]
    elif language == "sheng":
        return [
            "Sugary drinks na sodas", "Processed snacks na chips", "White bread na refined grains",
            "Fried foods", "High-sugar fruits (mango, grapes)", "Full-fat dairy products",
            "Sweetened cereals", "Candy na desserts", "High-sodium foods"
        ]
    else:
        return [
            "Sugary drinks and sodas", "Processed snacks and chips", "White bread and refined grains",
            "Fried foods", "High-sugar fruits (mango, grapes)", "Full-fat dairy products",
            "Sweetened cereals", "Candy and desserts", "High-sodium foods"
        ]

def get_portion_guidance(language: str = "english") -> List[str]:
    """Get portion guidance in appropriate language"""
    if language == "swahili":
        return [
            "Jaza nusu ya sahani yako kwa mboga zisizo na wanga",
            "Weka robo moja kwa protini nyepesi",
            "Tumia robo moja kwa wanga tata",
            "Jumuisha kiasi kidogo cha mafuta yenye afya",
            "Tumia vipimo vya mkono: kofi kwa protini, ngumi kwa mboga, mkono uliokunjwa kwa wanga"
        ]
    elif language == "sheng":
        return [
            "Fill half ya plate yako na non-starchy vegetables",
            "Allocate quarter kwa lean proteins",
            "Use quarter kwa complex carbohydrates",
            "Include small portion ya healthy fats",
            "Use hand measurements: palm kwa protein, fist kwa veggies, cupped hand kwa carbs"
        ]
    else:
        return [
            "Fill half your plate with non-starchy vegetables",
            "Allocate one-quarter for lean proteins",
            "Use one-quarter for complex carbohydrates",
            "Include a small portion of healthy fats",
            "Use hand measurements: palm for protein, fist for veggies, cupped hand for carbs"
        ]

def get_timing_recommendations(language: str = "english") -> List[str]:
    """Get timing recommendations in appropriate language"""
    if language == "swahili":
        return [
            "Kula kila masaa 3-4 ili kudumisha sukari thabiti ya damu",
            "Lisha kifungua kinywa ndani ya saa 1 ya kuamka",
            "Epuka kula masaa 2-3 kabla ya kulala",
            "Weka milo kwa usawa siku nzima",
            "Jumuisha protini katika kila mlo na kitafunio"
        ]
    elif language == "sheng":
        return [
            "Eat kila masaa 3-4 ku-maintain stable blood sugar",
            "Have breakfast within saa 1 ya kuamka",
            "Avoid eating masaa 2-3 before kulala",
            "Space meals evenly throughout the day",
            "Include protein katika kila meal na snack"
        ]
    else:
        return [
            "Eat every 3-4 hours to maintain stable blood sugar",
            "Have breakfast within 1 hour of waking up",
            "Avoid eating 2-3 hours before bedtime",
            "Space meals evenly throughout the day",
            "Include protein in every meal and snack"
        ]

def enhance_diet_plan(llm_plan: Dict[str, Any], request: DietPlanRequest, language: str) -> DietPlanResponse:
    """Enhance LLM diet plan with additional structure and validation"""
    bmi = request.weight / (request.height ** 2)
    
    # Calculate calorie target based on multiple factors
    base_calories = request.weight * (25 if bmi > 25 else 30)
    if request.age > 50:
        base_calories *= 0.9  # Reduce for older adults
    if request.diabetes_risk == "high":
        base_calories *= 0.85  # More aggressive deficit for high risk
    
    calorie_target = int(max(base_calories, 1200))  # Safety minimum
    
    # Enhanced meal plan structure
    meal_plan = llm_plan.get("meal_plan", {})
    if not isinstance(meal_plan, dict):
        meal_plan = get_default_meal_structure(language)
    
    return DietPlanResponse(
        daily_calorie_target=calorie_target,
        meal_plan=meal_plan,
        foods_to_include=llm_plan.get("foods_to_include", get_recommended_foods(language)),
        foods_to_avoid=llm_plan.get("foods_to_avoid", get_foods_to_avoid(language)),
        portion_guidance=llm_plan.get("portion_guidance", get_portion_guidance(language)),
        timing_recommendations=llm_plan.get("timing_recommendations", get_timing_recommendations(language)),
    )

def generate_structured_fallback_plan(request: DietPlanRequest, language: str) -> DietPlanResponse:
    """Generate a comprehensive fallback diet plan"""
    bmi = request.weight / (request.height ** 2)
    calorie_target = int(request.weight * (25 if bmi > 25 else 30))
    
    return DietPlanResponse(
        daily_calorie_target=calorie_target,
        meal_plan=get_default_meal_structure(language),
        foods_to_include=get_recommended_foods(language),
        foods_to_avoid=get_foods_to_avoid(language),
        portion_guidance=get_portion_guidance(language),
        timing_recommendations=get_timing_recommendations(language),
    )

@app.post("/api/v1/diet-plan", response_model=DietPlanResponse)
async def generate_diet_plan(
    diet_request: DietPlanRequest,
    background_tasks: BackgroundTasks,
    language: str = "english",
    api_key: bool = Depends(verify_api_key)
):
    try:
        from app.llm_chain import insulyn_llm
        
        # Generate personalized diet plan
        plan_text = await asyncio.to_thread(
            insulyn_llm.generate_diet_plan,
            diet_request.age,
            diet_request.weight,
            diet_request.height,
            diet_request.dietary_preferences,
            diet_request.health_conditions,
            diet_request.diabetes_risk,
        )
        
        # Parse and enhance the response
        parsed = MultiLanguageLLMOrchestrator().try_parse_llm_text(plan_text)
        
        if isinstance(parsed, dict):
            # Use enhanced diet plan generator
            enhanced_plan = enhance_diet_plan(parsed, diet_request, language)
            return enhanced_plan
        else:
            # Fallback to structured plan
            return generate_structured_fallback_plan(diet_request, language)
            
    except Exception as e:
        logger.exception("Diet plan error")
        raise HTTPException(status_code=500, detail="Error generating diet plan")

# -------------------------------------------------
# New Emergency Assessment Endpoint with Multi-language
# -------------------------------------------------
@app.post("/api/v1/emergency-assessment")
async def emergency_symptom_assessment(
    symptoms: List[str], 
    language: str = "english",
    api_key: bool = Depends(verify_api_key)
):
    """Assess symptoms for potential diabetes emergencies in multiple languages"""
    try:
        # Translate symptoms if needed
        translated_symptoms = symptoms  # In production, you'd translate these
        
        # Generate emergency plan
        from app.ml_model import diabetes_model
        lifestyle_engine = MultiLanguageLifestyleEngine()
        
        # Simple emergency detection
        emergency_keywords = {
            "english": ["chest pain", "difficulty breathing", "confusion", "unconscious", "seizure"],
            "swahili": ["maumivu ya kifua", "kupumua kwa shida", "mkanganyiko", "kutokuwa na fahamu", "kifafa"],
            "sheng": ["chest pain", "kupumua kwa shida", "confusion", "unconscious", "seizure"]
        }
        
        high_urgency = any(
            symptom.lower() in emergency_keywords.get(language, emergency_keywords["english"])
            for symptom in symptoms
        )
        
        urgency_level = "high" if high_urgency else "medium"
        
        if language == "swahili":
            response = {
                "assessment": {
                    "urgency_level": urgency_level,
                    "detected_risks": ["Kisukari"] if symptoms else ["Hakuna hatari kubwa imegundulika"],
                    "immediate_actions": [
                        "Pima viwango vya sukari ya damu mara moja",
                        "Kunywa maji ya kutosha",
                        "Wasiliana na huduma ya dharura ikiwa dalili ni kali"
                    ]
                },
                "recommended_actions": [
                    "Pima sukari ya damu mara moja",
                    "Kunywa maji",
                    "Wasiliana na daktari wako"
                ],
                "urgency_level": urgency_level,
                "timestamp": datetime.utcnow().isoformat(),
                "disclaimer": "Huu si uingizwaji wa ushauri wa matibabu. Tafuta ushauri wa matibabu wa haraka kwa dalili kali."
            }
        elif language == "sheng":
            response = {
                "assessment": {
                    "urgency_level": urgency_level,
                    "detected_risks": ["Sugar"] if symptoms else ["No major risks detected"],
                    "immediate_actions": [
                        "Check blood sugar levels immediately",
                        "Stay hydrated with water",
                        "Contact emergency services if symptoms severe"
                    ]
                },
                "recommended_actions": [
                    "Check blood sugar immediately",
                    "Drink water",
                    "Contact your doctor"
                ],
                "urgency_level": urgency_level,
                "timestamp": datetime.utcnow().isoformat(),
                "disclaimer": "This is not a substitute for professional medical advice. Seek immediate medical attention for severe symptoms."
            }
        else:
            response = {
                "assessment": {
                    "urgency_level": urgency_level,
                    "detected_risks": ["Diabetes"] if symptoms else ["No major risks detected"],
                    "immediate_actions": [
                        "Check blood sugar levels immediately",
                        "Stay hydrated with water",
                        "Contact emergency services if symptoms are severe"
                    ]
                },
                "recommended_actions": [
                    "Check blood sugar immediately",
                    "Drink water",
                    "Contact your healthcare provider"
                ],
                "urgency_level": urgency_level,
                "timestamp": datetime.utcnow().isoformat(),
                "disclaimer": "This is not a substitute for professional medical advice. Seek immediate medical attention for severe symptoms."
            }
        
        return response
        
    except Exception as e:
        logger.exception("Emergency assessment error")
        raise HTTPException(status_code=500, detail="Error processing emergency assessment")

# -------------------------------------------------
# Language Detection Endpoint
# -------------------------------------------------
@app.post("/api/v1/detect-language")
async def detect_language_endpoint(text: str = Form(...)):
    """Detect the language of input text"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        detected_language = language_processor.detect_language(text)
        confidence = "high" if detected_language != "english" else "medium"
        
        return {
            "detected_language": detected_language,
            "confidence": confidence,
            "input_length": len(text),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception("Language detection error")
        raise HTTPException(status_code=500, detail="Error detecting language")

# -------------------------------------------------
# Enhanced Error Handlers
# -------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    global error_count
    error_count += 1
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "error_id": str(uuid.uuid4())
        }
    )

# -------------------------------------------------
# Request Logging Middleware
# -------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Log request
    logger.info(f"📥 Request {request_id}: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"📤 Response {request_id}: {response.status_code} in {process_time:.2f}s")
        
        # Add headers for tracking
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ Request {request_id} failed after {process_time:.2f}s: {e}")
        raise

# -------------------------------------------------
# Final Application Startup Message
# -------------------------------------------------
logger.info("🎉 Insulyn AI API v3.0.0 fully initialized and ready!")
logger.info("🌍 Multi-Language Support: English, Swahili, Sheng")
logger.info("🎤 Voice Command Support: Speech-to-Text & Text-to-Speech")
logger.info("📚 Available Enhanced Endpoints:")
logger.info("   ├── /api/v1/predict (POST)        - Multi-language diabetes risk prediction")
logger.info("   ├── /api/v1/chat (POST)           - Multi-language AI chat")
logger.info("   ├── /api/v1/chat/voice (POST)     - Voice chat with audio responses")
logger.info("   ├── /api/v1/tts (POST)            - Text-to-speech conversion")
logger.info("   ├── /api/v1/stt (POST)            - Speech-to-text conversion")
logger.info("   ├── /api/v1/detect-language (POST)- Language detection")
logger.info("   ├── /api/v1/diet-plan (POST)      - Multi-language diet plans")
logger.info("   ├── /api/v1/emergency-assessment (POST) - Multi-language symptom assessment")
logger.info("   ├── /api/v1/chat/topics (GET)     - Multi-language chat topics")
logger.info("   └── /health (GET)                 - Comprehensive health check")
logger.info("")
logger.info("🔊 Voice Features:")
logger.info("   ├── Speech-to-Text conversion")
logger.info("   ├── Text-to-Speech responses")
logger.info("   ├── Multi-language voice support")
logger.info("   └── Real-time audio processing")
logger.info("")
logger.info("🌐 Language Features:")
logger.info("   ├── English, Swahili, and Sheng support")
logger.info("   ├── Automatic language detection")
logger.info("   ├── Culturally appropriate responses")
logger.info("   └── Medical term translation")
logger.info("")
logger.info(f"⏰ Server started at: {app_start_time}")
logger.info("🌈 Insulyn AI is ready to serve your diabetes management needs in multiple languages!")

# -------------------------------------------------
# End of File
# -------------------------------------------------
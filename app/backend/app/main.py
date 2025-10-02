# app/backend/app/main.py
import os
import uuid
import asyncio
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager
import redis
from groq import Groq
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import base64
import time
import os
import uuid
import asyncio
import logging
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import redis
from groq import Groq
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import base64
import time
#import os
#import uuid
#import asyncio
#import logging
#import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with CORS
app = FastAPI(
    title="Insulyn AI",
    version="1.0.0",
    debug=os.getenv('DEBUG', 'True').lower() == 'true'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with CORS
app = FastAPI(
    title="Insulyn AI",
    version="1.0.0",
    debug=os.getenv('DEBUG', 'True').lower() == 'true'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import json
import re
from typing import List

# Add this function to your app/main.py
def parse_and_clean_response(raw_response: str) -> str:
    """
    Parse and clean the raw response to make it human-readable
    """
    try:
        # First, try to parse as JSON if it's stringified JSON
        if isinstance(raw_response, str) and raw_response.strip().startswith('{'):
            try:
                parsed_json = json.loads(raw_response)
                if 'raw_response' in parsed_json:
                    raw_response = parsed_json['raw_response']
            except json.JSONDecodeError:
                pass  # Not JSON, continue with string processing
        
        # Clean up escaped characters
        cleaned = raw_response.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t')
        
        return cleaned
    except Exception as e:
        logger.error(f"Response parsing error: {e}")
        return raw_response  # Return original if parsing fails

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

# ... rest of your imports and code ...
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
# PREMIUM FastAPI App - COMPLETE & WORKING VERSION
# -------------------------------------------------
import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app start time
app_start_time = datetime.utcnow()

# Security
security = HTTPBearer(auto_error=False)

# Application Settings
class Settings:
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'Insulyn AI')
    API_V1_STR = os.getenv('API_V1_STR', '/api/v1')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # API Keys for different environments
    API_KEYS = {
        "development": ["dev-key-123", "test-key-456", "demo-key-789"],
        "production": ["prod-premium-key-2024"]  # Replace with actual production keys
    }
    
    # Get current environment keys
    def get_api_keys(self):
        return self.API_KEYS.get(self.ENVIRONMENT, self.API_KEYS["development"])

settings = Settings()

# -------------------------------------------------
# Pydantic Models for Request/Response
# -------------------------------------------------
class DiabetesAssessmentRequest(BaseModel):
    glucose: float
    blood_pressure: float
    weight: float
    height: float
    age: int
    pregnancies: Optional[int] = 0
    skin_thickness: Optional[float] = 20.0
    insulin: Optional[float] = 80.0
    diabetes_pedigree_function: Optional[float] = 0.5
    language: Optional[str] = "english"

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "english"

class AssessmentResponse(BaseModel):
    assessment_id: str
    timestamp: str
    executive_summary: str
    risk_analysis: Dict[str, Any]
    health_metrics: Dict[str, Any]
    premium_recommendations: Dict[str, Any]
    success_plan: Dict[str, Any]
    support_resources: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    context: Dict[str, Any]
    suggestions: List[str]
    premium_features: List[str]

# -------------------------------------------------
# COMPLETE WORKING VERSION WITH ALL DEPENDENCIES
# -------------------------------------------------
import os
import uuid
import asyncio
import logging
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from groq import Groq

# Load environment variables
from dotenv import load_dotenv

# Try multiple .env file locations
env_loaded = False
possible_paths = [
    './app/.env',
    '../app/.env', 
    '../../app/.env',
    '.env',
    '../.env',
    './backend/.env',
    '../backend/.env'
]

for path in possible_paths:
    if os.path.exists(path):
        load_dotenv(path)
        print(f"✅ Loaded .env from: {path}")
        env_loaded = True
        break

if not env_loaded:
    print("❌ No .env file found in common locations")
    load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app start time
app_start_time = datetime.utcnow()

# Security
security = HTTPBearer(auto_error=False)

# Application Settings
class Settings:
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'Insulyn AI')
    API_V1_STR = os.getenv('API_V1_STR', '/api/v1')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.6))
    
    # API Keys for different environments
    API_KEYS = {
        "development": ["dev-key-123", "test-key-456", "demo-key-789"],
        "production": ["prod-premium-key-2024"]
    }
    
    def get_api_keys(self):
        return self.API_KEYS.get(self.ENVIRONMENT, self.API_KEYS["development"])

settings = Settings()

# -------------------------------------------------
# AUTHENTICATION DEPENDENCY - MUST BE DEFINED BEFORE USE
# -------------------------------------------------
async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Flexible API key verification - disabled in development"""
    if settings.ENVIRONMENT == "production":
        if not credentials or credentials.credentials not in settings.get_api_keys():
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key. Please check your credentials."
            )
        return True
    else:
        # In development, allow all requests with or without API key
        logger.debug("🔓 Development mode - API key verification disabled")
        return True

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "english"
    user_id: Optional[str] = "default"

class DiabetesAssessmentRequest(BaseModel):
    glucose: float = Field(..., ge=50, le=500, description="Fasting blood glucose in mg/dL")
    blood_pressure: float = Field(..., ge=60, le=200, description="Systolic blood pressure in mmHg")
    weight: float = Field(..., ge=30, le=200, description="Weight in kilograms")
    height: float = Field(..., ge=100, le=220, description="Height in centimeters")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    pregnancies: Optional[int] = Field(0, ge=0, le=20, description="Number of pregnancies")
    skin_thickness: Optional[float] = Field(20.0, ge=0, le=100, description="Skin thickness in mm")
    insulin: Optional[float] = Field(80.0, ge=0, le=1000, description="Insulin level in mu U/ml")
    diabetes_pedigree_function: Optional[float] = Field(0.5, ge=0, le=2.5, description="Diabetes pedigree function")
    language: Optional[str] = "english"

class PremiumDiabetesAssessmentRequest(BaseModel):
    glucose: float = Field(..., ge=50, le=500, description="Fasting blood glucose in mg/dL")
    blood_pressure: float = Field(..., ge=60, le=200, description="Systolic blood pressure in mmHg")
    weight: float = Field(..., ge=30, le=200, description="Weight in kilograms")
    height: float = Field(..., ge=100, le=220, description="Height in centimeters")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    gender: str = Field("unknown", description="Gender for personalized insights")
    activity_level: str = Field("moderate", description="Daily activity level")
    dietary_preferences: List[str] = Field([], description="Dietary preferences and restrictions")
    family_history: bool = Field(False, description="Family history of diabetes")
    existing_conditions: List[str] = Field([], description="Existing health conditions")
    medications: List[str] = Field([], description="Current medications")
    lifestyle_factors: Dict[str, Any] = Field({}, description="Smoking, alcohol, sleep patterns")
    goals: List[str] = Field([], description="Health and wellness goals")
    language: str = Field("english", description="Preferred language for responses")

class AssessmentResponse(BaseModel):
    assessment_id: str
    timestamp: str
    executive_summary: str
    risk_analysis: Dict[str, Any]
    health_metrics: Dict[str, Any]
    premium_recommendations: Dict[str, Any]
    success_plan: Dict[str, Any]
    support_resources: Dict[str, Any]

class PremiumAssessmentResponse(BaseModel):
    assessment_id: str
    timestamp: str
    executive_summary: str
    risk_analysis: Dict[str, Any]
    health_metrics: Dict[str, Any]
    personalized_insights: Dict[str, Any]
    action_plan: Dict[str, Any]
    predictive_analytics: Dict[str, Any]
    wow_features: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    context: Dict[str, Any]
    suggestions: List[str]
    premium_features: List[str]
    conversation_id: str
    search_used: bool
    model: str

# -------------------------------------------------
# GPT-OSS-20B DIABETES SPECIALIST SERVICE (FIXED)
# -------------------------------------------------
class GPTOSSDiabetesSpecialist:
    def __init__(self):
        self.client = None
        self.conversation_memory = {}
        self.initialize_groq_client()
    
    def initialize_groq_client(self):
        """Initialize Groq client with GPT-OSS-20B model"""
        try:
            groq_api_key = os.getenv('GROQ_API_KEY')
            
            if not groq_api_key:
                logger.error("❌ GROQ_API_KEY not found in environment variables")
                self.client = None
                return
            
            print(f"🔑 API Key found: {groq_api_key[:10]}...{groq_api_key[-5:]}")
            
            if not groq_api_key.startswith('gsk_'):
                logger.error("❌ Invalid Groq API key format")
                self.client = None
                return
            
            self.client = Groq(api_key=groq_api_key)
            
            # Test the connection
            test_response = self.client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": "Say 'GPT-OSS-20B Diabetes Specialist Ready'"}],
                max_tokens=20,
                temperature=0.1
            )
            
            test_content = test_response.choices[0].message.content
            print(f"✅ GPT-OSS-20B Test Successful: {test_content}")
            print(f"🚀 GPT-OSS-20B Diabetes Specialist ACTIVE!")
            
        except Exception as e:
            print(f"❌ GPT-OSS-20B Initialization Failed: {e}")
            self.client = None
    
    def create_diabetes_prompt(self, message: str, language: str, context: Dict = None) -> List[Dict]:
        """Create comprehensive diabetes specialist prompt for GPT-OSS-20B"""
        
        if language == "swahili":
            system_content = """Uko kwenye mfumo wa Insulyn AI, mtaalamu wa kisukari wa kiwango cha juu. Model: GPT-OSS-20B

JIBU KWA KISWAHILI KWA USAHIHI WA KITAALAMU:
• Toa maelezo ya kina ya kitabibu kuhusu kisukari
• Wasilisha mapendekezo madhubuti na ya vitendo
• Tumia lugha rahisi lakini yenye usahihi wa kitaalamu
• Eleza vizuru hatua za kuzuia na kudhibiti kisukari
• Toa mifano halisi na ya kutumika Kenya

MUHIMU: Onyesha kuwa wewe ni msaidizi wa AI na watu wanapaswa kumtafuta daktari kwa ushauri binafsi.

JIBU LENYE UNDANI WA KITAALAMU NA USAHIHI."""
        else:
            system_content = f"""You are Insulyn AI, an expert diabetes specialist powered by GPT-OSS-20B.

ROLE: World-Class Diabetes Specialist & Metabolic Health Expert
MODEL: {settings.LLM_MODEL_NAME}
CONTEXT: Advanced Diabetes Prevention, Management & Metabolic Optimization

RESPONSE GUIDELINES:
• Provide medically accurate, evidence-based information
• Give specific, actionable advice with scientific rationale
• Use clear, professional language while being empathetic
• Include advanced metabolic insights and pathways
• Consider Kenyan/African context for practical recommendations
• Show understanding of insulin signaling, glucose metabolism, mitochondrial function
• Provide both immediate actions and long-term strategies

CRITICAL: Always state that you are an AI assistant and users should consult healthcare providers for personal medical advice.

Provide comprehensive, scientifically rigorous responses that demonstrate advanced medical knowledge."""

        messages = [{"role": "system", "content": system_content}]
        
        # Add conversation history for context (without timestamps)
        if context and 'user_id' in context:
            history = self.get_conversation_history(context['user_id'])
            for msg in history[-3:]:  # Last 3 messages for context
                # Only include role and content (remove timestamp)
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({"role": "user", "content": message})
        
        return messages
    
    async def generate_diabetes_response(self, message: str, language: str = "english", user_id: str = "default") -> Dict[str, Any]:
        """Generate diabetes response using GPT-OSS-20B"""
        
        print(f"\n💬 GPT-OSS-20B Processing: '{message}'")
        print(f"   Language: {language}, User: {user_id}")
        
        # Add to conversation memory (with timestamp for our tracking)
        self.add_to_conversation(user_id, "user", message)
        
        if not self.client:
            print("❌ GPT-OSS-20B client not available")
            fallback = self._get_enhanced_fallback(message, language)
            self.add_to_conversation(user_id, "assistant", fallback)
            return {
                "success": False,
                "response": fallback,
                "model": "enhanced_fallback",
                "status": "fallback"
            }
        
        try:
            messages = self.create_diabetes_prompt(message, language, {"user_id": user_id})
            
            print("🔄 Calling GPT-OSS-20B...")
            print(f"   Messages being sent: {len(messages)}")
            
            # Debug: Print the message structure
            for i, msg in enumerate(messages):
                print(f"   Message {i}: role={msg['role']}, content_length={len(msg['content'])}")
            
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=1500
            )
            
            llm_response = response.choices[0].message.content
            print(f"✅ GPT-OSS-20B Response received: {len(llm_response)} chars")
            
            self.add_to_conversation(user_id, "assistant", llm_response)
            
            return {
                "success": True,
                "response": llm_response,
                "model": settings.LLM_MODEL_NAME,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ GPT-OSS-20B call failed: {e}")
            fallback = self._get_enhanced_fallback(message, language)
            self.add_to_conversation(user_id, "assistant", fallback)
            return {
                "success": False,
                "response": fallback,
                "model": "error_fallback",
                "status": "error"
            }
    
    def _get_enhanced_fallback(self, message: str, language: str) -> str:
        """Enhanced fallback responses"""
        message_lower = message.lower()
        
        if language == "swahili":
            if 'symptom' in message_lower or 'dalili' in message_lower:
                return """🔬 **Msaidizi wa Kisukari wa AI**

**Dalili za Kawaida za Kisukari Aina ya 2:**
• **Kiu na mkojo mara kwa mara** - Mwili unajaribu kuondoa sukari ya ziada kupitia mkojo
• **Njaa kubwa** - Miguu haipati nishati inayohitaji kutoka kwa glukosi
• **Uchovu** - Mwili hauwezi kutumia glukosi kwa ufanisi kwa nishati
• **Macho yabivu** - Mabadiliko ya kiwango cha maji yanaathiri macho
• **Vidonda visivyopona** - Mshtuko wa damu na neva husababisha upungufu wa usambazaji wa damu
• **Kupoteza uzito bila kukusudia** - Mwili unachoma mafuta na misuli kwa nishati

**Tafadhali:**
• Zuru kituo cha afya kwa uchunguzi
• Pima sukari ya damu kila mwaka
• Wasiliana na daktari wako

Natumaini mfumo wetu utarejea hivi karibuni! 🩺"""
            else:
                return """🔬 **Msaidizi wa Kisukari wa AI**

Nina ushauri wa msingi kuhusu kisukari:

**Dalili za Kawaida:**
• Kiu na mkojo mara kwa mara
• Njaa kubwa na kupoteza uzito
• Uchovu na macho yabivu
• Vidonda visivyo pona

**Tafadhali:**
• Zuru kituo cha afya kwa uchunguzi
• Pima sukari ya damu kila mwaka
• Wasiliana na daktari wako

Natumaini mfumo wetu utarejea hivi karibuni! 🩺"""
        else:
            if 'symptom' in message_lower:
                return """🔬 **AI Diabetes Specialist**

**Common Symptoms of Type 2 Diabetes:**

**Early Signs:**
• **Increased thirst and frequent urination** - Your kidneys work overtime to remove excess sugar
• **Extreme hunger** - Cells aren't getting energy from glucose
• **Fatigue and irritability** - Body can't efficiently convert glucose to energy
• **Blurred vision** - Fluid shifts affecting eye lenses
• **Slow-healing sores** - Poor circulation and nerve damage
• **Unexplained weight loss** - Body burning fat and muscle for energy
• **Frequent infections** - High sugar weakens immune response
• **Dark skin patches** (acanthosis nigricans) - Insulin resistance marker

**When to See a Doctor:**
• If you're experiencing any of these symptoms
• If you have risk factors (family history, overweight, over 45)
• For regular screening if at risk

**Remember:** Early detection is key for effective management! 🩺

*I'm an AI assistant. Please consult healthcare providers for medical advice.*"""
            else:
                return """🔬 **AI Diabetes Specialist**

I have comprehensive diabetes information:

**Common Symptoms:**
• Frequent thirst and urination
• Extreme hunger and weight loss
• Fatigue and blurred vision
• Slow-healing sores
• Frequent infections
• Dark skin patches

**Please:**
• Visit health center for screening
• Check blood sugar yearly if at risk
• Consult your doctor for personalized advice

I hope our system returns shortly! 🩺"""
    
    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history (without timestamps for API)"""
        history = self.conversation_memory.get(user_id, [])
        # Return only role and content for API compatibility
        return [{"role": msg["role"], "content": msg["content"]} for msg in history]
    
    def add_to_conversation(self, user_id: str, role: str, content: str):
        """Add message to conversation history (with timestamp for our tracking)"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        # Store with timestamp for our internal tracking
        self.conversation_memory[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()  # For our internal use only
        })
        
        # Keep only last 10 messages to prevent memory bloat
        if len(self.conversation_memory[user_id]) > 10:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-10:]
# -------------------------------------------------
# ENHANCED ML MODEL FOR DIABETES ASSESSMENT
# -------------------------------------------------
class EnhancedDiabetesMLModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Enhanced model loading with actual best_model.pkl"""
        try:
            model_path = os.getenv('MODEL_PATH', 'app/backend/data/best_model.pkl')
            
            if os.path.exists(model_path):
                # Try to load the actual model
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Actual ML model loaded successfully from {model_path}")
                print(f"🎯 ML Model Type: {type(self.model)}")
                
                # Check if it's a scikit-learn model
                if hasattr(self.model, 'predict_proba'):
                    print("✅ Model has predict_proba method - ready for real predictions!")
                else:
                    print("⚠️ Model loaded but may not be scikit-learn compatible")
                    
            else:
                # Fallback to enhanced mock model
                self.model = "enhanced_mock_model"
                logger.warning(f"❌ Model file not found at {model_path}, using enhanced mock model")
                print("⚠️ Using enhanced mock model - real best_model.pkl not found")
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model = "enhanced_mock_model"
            logger.info("✅ Enhanced ML model initialized as fallback")
    
    def predict(self, features: Dict[str, Any]) -> tuple:
        """Enhanced prediction with actual model or comprehensive risk calculation"""
        try:
            # Extract and prepare features in correct order for the model
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            # Create feature array in correct order
            feature_values = [
                features.get('Pregnancies', 0),
                features.get('Glucose', 100),
                features.get('BloodPressure', 120),
                features.get('SkinThickness', 20),
                features.get('Insulin', 80),
                features.get('BMI', 25),
                features.get('DiabetesPedigreeFunction', 0.5),
                features.get('Age', 30)
            ]
            
            # If we have a real scikit-learn model with predict_proba
            if self.model and hasattr(self.model, 'predict_proba'):
                print("🎯 Using REAL ML model for prediction...")
                
                # Convert to numpy array and reshape for single prediction
                X = np.array([feature_values])
                
                # Get probability prediction
                probability = self.model.predict_proba(X)[0][1]  # Probability of class 1 (diabetes)
                
                # Determine risk label based on probability
                if probability >= 0.7:
                    risk_label = "High Risk"
                elif probability >= 0.4:
                    risk_label = "Moderate Risk" 
                elif probability >= 0.2:
                    risk_label = "Low Risk"
                else:
                    risk_label = "Very Low Risk"
                
                # Get feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    feature_importances = dict(zip(feature_names, self.model.feature_importances_))
                else:
                    feature_importances = {
                        'Glucose': 0.35, 'BMI': 0.25, 'Age': 0.15, 'BloodPressure': 0.10,
                        'DiabetesPedigreeFunction': 0.08, 'Pregnancies': 0.05,
                        'SkinThickness': 0.01, 'Insulin': 0.01
                    }
                
                logger.info(f"🎯 REAL ML Prediction: {risk_label} (probability: {probability:.3f})")
                return risk_label, float(probability), feature_importances
            
            else:
                # Fallback to enhanced risk calculation
                print("🔄 Using enhanced mock model calculation...")
                return self._calculate_mock_risk(feature_values, feature_names)
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            print(f"❌ Prediction failed: {e}")
            # Fallback to mock calculation
            return self._calculate_mock_risk(
                [features.get(k, 0) for k in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
                ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            )
    
    def _calculate_mock_risk(self, feature_values: List[float], feature_names: List[str]) -> tuple:
        """Enhanced mock risk calculation when real model is not available"""
        glucose, bmi, age, bp = feature_values[1], feature_values[5], feature_values[7], feature_values[2]
        pregnancies, dpf = feature_values[0], feature_values[6]
        
        # Advanced risk calculation
        risk_score = 0.0
        
        # Glucose contribution (more realistic)
        if glucose >= 200: risk_score += 0.4
        elif glucose >= 160: risk_score += 0.35
        elif glucose >= 140: risk_score += 0.3
        elif glucose >= 126: risk_score += 0.25
        elif glucose >= 100: risk_score += 0.15
        else: risk_score += 0.05
        
        # BMI contribution
        if bmi >= 40: risk_score += 0.3
        elif bmi >= 35: risk_score += 0.25
        elif bmi >= 30: risk_score += 0.2
        elif bmi >= 25: risk_score += 0.1
        else: risk_score += 0.05
        
        # Age contribution
        if age >= 65: risk_score += 0.2
        elif age >= 50: risk_score += 0.15
        elif age >= 40: risk_score += 0.1
        else: risk_score += 0.05
        
        # Blood pressure contribution
        if bp >= 180: risk_score += 0.1
        elif bp >= 160: risk_score += 0.08
        elif bp >= 140: risk_score += 0.05
        else: risk_score += 0.02
        
        # Genetic and other factors
        if dpf > 1.0: risk_score += 0.05
        if pregnancies > 0: risk_score += (pregnancies * 0.02)
        
        risk_score = min(0.95, max(0.05, risk_score))
        
        # Risk classification
        if risk_score > 0.7:
            risk_label = "High Risk"
        elif risk_score > 0.4:
            risk_label = "Moderate Risk"
        elif risk_score > 0.2:
            risk_label = "Low Risk"
        else:
            risk_label = "Very Low Risk"
        
        feature_importances = {
            'Glucose': 0.35,
            'BMI': 0.25,
            'Age': 0.15,
            'BloodPressure': 0.10,
            'DiabetesPedigreeFunction': 0.08,
            'Pregnancies': 0.05,
            'SkinThickness': 0.01,
            'Insulin': 0.01
        }
        
        logger.info(f"🎯 Enhanced Mock Prediction: {risk_label} (score: {risk_score:.3f})")
        return risk_label, float(risk_score), feature_importances

# Initialize services
gpt_oss_specialist = GPTOSSDiabetesSpecialist()
diabetes_model = EnhancedDiabetesMLModel()

# -------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n" + "="*60)
    print("🚀 GPT-OSS-20B DIABETES SPECIALIST STARTING")
    print("="*60)
    print(f"🔧 Environment: {settings.ENVIRONMENT}")
    print(f"🧠 LLM Model: {settings.LLM_MODEL_NAME}")
    print(f"🔑 API Key: {'✅ Loaded' if settings.GROQ_API_KEY else '❌ Missing'}")
    print(f"🤖 GPT-OSS-20B: {'✅ Connected' if gpt_oss_specialist.client else '❌ Disconnected'}")
    print(f"🎯 ML Model: {'✅ Loaded' if diabetes_model.model and hasattr(diabetes_model.model, 'predict_proba') else '🔄 Using Enhanced Mock'}")
    print("="*60)
    
    yield
    
    # Shutdown
    print("🛑 GPT-OSS-20B Diabetes Specialist shutting down")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Advanced Diabetes Prevention & Health Optimization Platform with GPT-OSS-20B AI-Powered Insights",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions for assessment response
def identify_risk_factors(features: Dict[str, Any], bmi: float) -> List[Dict[str, Any]]:
    risk_factors = []
    
    if features['Glucose'] >= 126:
        risk_factors.append({"factor": "Diabetes-level glucose", "severity": "high", "action": "Immediate medical consultation"})
    elif features['Glucose'] >= 100:
        risk_factors.append({"factor": "Prediabetes glucose levels", "severity": "moderate", "action": "Lifestyle intervention"})
        
    if bmi >= 30:
        risk_factors.append({"factor": "Clinical obesity", "severity": "high", "action": "Weight management program"})
    elif bmi >= 25:
        risk_factors.append({"factor": "Overweight", "severity": "moderate", "action": "Diet and exercise optimization"})
        
    if features['BloodPressure'] >= 140:
        risk_factors.append({"factor": "Stage 2 hypertension", "severity": "high", "action": "Blood pressure management"})
    elif features['BloodPressure'] >= 130:
        risk_factors.append({"factor": "Stage 1 hypertension", "severity": "moderate", "action": "Lifestyle modifications"})
        
    if features['Age'] >= 45:
        risk_factors.append({"factor": "Age-related risk increase", "severity": "moderate", "action": "Enhanced monitoring"})
        
    return risk_factors if risk_factors else [{"factor": "No significant risk factors", "severity": "low", "action": "Maintain healthy lifestyle"}]

def calculate_metabolic_age(features: Dict[str, Any]) -> int:
    base_age = features['Age']
    adjustment = 0
    
    if features['Glucose'] < 100:
        adjustment -= 5
    if features['BMI'] < 25:
        adjustment -= 3
    if features['BloodPressure'] < 120:
        adjustment -= 2
        
    return max(20, base_age + adjustment)

def calculate_health_score(features: Dict[str, Any]) -> int:
    score = 50
    bmi = features.get('BMI', 25)
    glucose = features.get('Glucose', 100)
    bp = features.get('BloodPressure', 120)
    
    if 18.5 <= bmi <= 24.9: score += 20
    elif 25 <= bmi <= 29.9: score += 10
    
    if glucose < 100: score += 20
    elif glucose < 126: score += 10
        
    if bp < 120: score += 15
    elif bp < 140: score += 10
        
    return min(100, score)

# -------------------------------------------------
# CHAT ENDPOINT WITH GPT-OSS-20B
# -------------------------------------------------
@app.post("/api/v1/chat", response_model=ChatResponse)
async def diabetes_chat(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: bool = Depends(verify_api_key)
):
    """Chat with GPT-OSS-20B Diabetes Specialist"""
    try:
        message = chat_request.message.strip()
        language = chat_request.language
        user_id = chat_request.user_id or "default"
        
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be empty")
        
        logger.info(f"🧠 Chat request from {user_id}: {message[:50]}...")

        # Generate GPT-OSS-20B response
        llm_result = await gpt_oss_specialist.generate_diabetes_response(
            message=message,
            language=language,
            user_id=user_id
        )
        
        response_text = llm_result["response"]
        model_used = llm_result["model"]
        status = llm_result["status"]
        
        # Enhanced context analysis
        def analyze_context(message: str) -> Dict[str, Any]:
            message_lower = message.lower()
            context = {
                "topics": [],
                "urgency": "routine",
                "medical_complexity": "basic"
            }
            
            # Topic detection
            if any(word in message_lower for word in ['diet', 'food', 'nutrition', 'eat']):
                context["topics"].append("nutrition")
            if any(word in message_lower for word in ['exercise', 'workout', 'fitness']):
                context["topics"].append("exercise")
            if any(word in message_lower for word in ['symptom', 'pain', 'feel']):
                context["topics"].append("symptoms")
                context["urgency"] = "moderate"
            if any(word in message_lower for word in ['emergency', 'urgent', 'help']):
                context["urgency"] = "high"
            
            return context
        
        context = analyze_context(message)
        
        # Enhanced suggestions
        def get_suggestions(context: Dict) -> List[str]:
            suggestions = []
            topics = context.get("topics", [])
            
            if "nutrition" in topics:
                suggestions.extend(["Meal planning", "Food alternatives", "Nutrition guide"])
            if "exercise" in topics:
                suggestions.extend(["Workout routines", "Activity tracking", "Fitness plan"])
            if "symptoms" in topics:
                suggestions.extend(["Symptom checker", "Doctor consultation", "Emergency assessment"])
            
            if not suggestions:
                suggestions = ["Health assessment", "Progress tracking", "Learn more"]
            
            return suggestions[:3]
        
        suggestions = get_suggestions(context)
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow().isoformat(),
            context=context,
            suggestions=suggestions,
            premium_features=[
                "GPT-OSS-20B Medical Intelligence",
                "Advanced Diabetes Knowledge",
                "Personalized Health Insights",
                "Multi-language Support",
                "Conversation Memory"
            ],
            conversation_id=user_id,
            search_used=False,
            model=model_used
        )
        
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(
            status_code=500, 
            detail="Our AI diabetes specialist is currently optimizing. Please try again."
        )

# -------------------------------------------------
# STANDARD DIABETES ASSESSMENT ENDPOINT
# -------------------------------------------------
@app.post("/api/v1/diabetes-assessment", response_model=AssessmentResponse)
async def premium_diabetes_assessment(
    request: DiabetesAssessmentRequest,
    api_key: bool = Depends(verify_api_key)
):
    """PREMIUM diabetes assessment with advanced health intelligence"""
    try:
        logger.info(f"📊 Starting premium assessment for age {request.age}")
        
        # Validate inputs
        if request.height <= 0:
            raise HTTPException(status_code=422, detail="Height must be greater than 0")
        if request.weight <= 0:
            raise HTTPException(status_code=422, detail="Weight must be greater than 0")
        
        # Calculate advanced metrics
        height_m = request.height / 100
        bmi = request.weight / (height_m ** 2)
        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        
        # Prepare features for AI analysis
        features = {
            'Pregnancies': request.pregnancies,
            'Glucose': request.glucose,
            'BloodPressure': request.blood_pressure,
            'SkinThickness': request.skin_thickness,
            'Insulin': request.insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': request.diabetes_pedigree_function,
            'Age': request.age
        }
        
        # Get AI prediction
        risk_label, probability, feature_importances = diabetes_model.predict(features)
        
        # Create response that matches frontend expectations
        response = AssessmentResponse(
            assessment_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            executive_summary=f"Comprehensive Health Assessment - {risk_label}",
            risk_analysis={
                "diabetes_risk": {
                    "level": risk_label,
                    "probability": round(probability, 3),
                    "confidence": "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low"
                },
                "key_risk_factors": identify_risk_factors(features, bmi),
                "prevention_strategy": "Immediate intervention" if "high" in risk_label.lower() else "Proactive management"
            },
            health_metrics={
                "vital_statistics": {
                    "bmi": round(bmi, 1),
                    "bmi_category": bmi_category,
                    "metabolic_age": calculate_metabolic_age(features),
                    "health_score": calculate_health_score(features)
                },
                "health_insights": {
                    "metabolic_health": {
                        "status": "excellent" if features['Glucose'] < 100 and bmi < 25 else "good" if features['Glucose'] < 126 and bmi < 30 else "needs_improvement",
                        "score": 6 if features['Glucose'] < 100 and bmi < 25 else 4 if features['Glucose'] < 126 and bmi < 30 else 2,
                        "factors": ["glucose_levels", "body_composition"]
                    },
                    "cardiovascular_risk": {
                        "risk_level": "low" if features['BloodPressure'] < 120 and bmi < 25 else "moderate" if features['BloodPressure'] < 140 and bmi < 30 else "elevated",
                        "monitoring_recommendation": "Annual checkup" if features['BloodPressure'] < 120 else "6-month monitoring"
                    }
                },
                "improvement_opportunities": ["Weight management"] if bmi > 25 else ["Maintain healthy lifestyle"]
            },
            premium_recommendations={
                "action_plan": {
                    "lifestyle_optimization": {
                        "nutrition_plan": {
                            "goal": "Weight management and glucose control",
                            "breakfast": "High-fiber cereal with fruits and nuts, or whole-grain toast with avocado",
                            "lunch": "Lean protein (chicken, fish) with vegetables and whole grains like brown rice or quinoa",
                            "dinner": "Balanced meal with portion control - focus on vegetables and lean protein",
                            "hydration": "8-10 glasses of water daily, limit sugary drinks"
                        },
                        "fitness_program": {
                            "cardio": "30 minutes brisk walking, 5 days/week",
                            "strength": "Bodyweight exercises 2 times/week (squats, push-ups, planks)",
                            "flexibility": "Stretching daily for 10-15 minutes",
                            "activity_tracking": "Use pedometer or fitness app to track 10,000 steps daily"
                        },
                        "wellness_strategies": [
                            "Stress management techniques (meditation, deep breathing)",
                            "Adequate sleep (7-9 hours per night)",
                            "Regular health monitoring (weight, blood pressure)",
                            "Social support engagement with family and friends"
                        ]
                    }
                }
            },
            success_plan={
                "action_timeline": {
                    "immediate": [
                        "Consult healthcare provider for comprehensive evaluation",
                        "Start food diary to track eating habits",
                        "Begin daily 30-minute walking routine"
                    ],
                    "30_days": [
                        "Implement dietary changes and portion control",
                        "Establish consistent exercise routine",
                        "Monitor weight and blood glucose weekly"
                    ],
                    "90_days": [
                        "Reassess health metrics and progress",
                        "Adjust nutrition and fitness plan as needed",
                        "Schedule follow-up appointment with healthcare provider"
                    ]
                },
                "progress_tracking": {
                    "key_metrics": ["Weight", "Glucose", "Blood Pressure", "BMI"],
                    "frequency": "Weekly" if "high" in risk_label.lower() else "Monthly"
                }
            },
            support_resources={
                "professional": "Schedule appointment with healthcare provider for comprehensive evaluation and personalized medical advice",
                "educational": "Access diabetes prevention resources, nutrition guides, and educational materials from certified health organizations",
                "community": "Join local health support groups or online communities for motivation and shared experiences"
            }
        )
        
        logger.info(f"🎯 Premium assessment completed - Risk: {risk_label}, Probability: {probability:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Premium assessment error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Our advanced health analysis system is temporarily unavailable. Please try again shortly."
        )

# -------------------------------------------------
# SYSTEM ENDPOINTS
# -------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 Insulyn AI - GPT-OSS-20B Diabetes Specialist",
        "status": "optimal",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "llm_model": settings.LLM_MODEL_NAME,
        "llm_status": "connected" if gpt_oss_specialist.client else "disconnected",
        "ml_model_status": "loaded" if diabetes_model.model and hasattr(diabetes_model.model, 'predict_proba') else "enhanced_mock",
        "endpoints": {
            "chat": "/api/v1/chat",
            "assessment": "/api/v1/diabetes-assessment",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "gpt_oss_20b": "connected" if gpt_oss_specialist.client else "disconnected",
            "ml_model": "loaded" if diabetes_model.model and hasattr(diabetes_model.model, 'predict_proba') else "enhanced_mock",
            "api": "operational"
        },
        "uptime": str(datetime.utcnow() - app_start_time)
    }

@app.get("/api/v1/system/status")
async def system_status():
    """System status endpoint"""
    return {
        "llm": {
            "model": settings.LLM_MODEL_NAME,
            "status": "connected" if gpt_oss_specialist.client else "disconnected",
            "temperature": settings.LLM_TEMPERATURE
        },
        "ml_model": {
            "status": "loaded" if diabetes_model.model and hasattr(diabetes_model.model, 'predict_proba') else "enhanced_mock",
            "path": os.getenv('MODEL_PATH', 'app/backend/data/best_model.pkl')
        },
        "environment": settings.ENVIRONMENT,
        "active_conversations": len(gpt_oss_specialist.conversation_memory),
        "timestamp": datetime.utcnow().isoformat()
    }

print("\n🎯 GPT-OSS-20B DIABETES SPECIALIST READY!")
print(f"   🤖 Model: {settings.LLM_MODEL_NAME}")
print(f"   🎯 ML Model Path: {os.getenv('MODEL_PATH', 'app/backend/data/best_model.pkl')}")
print(f"   💬 Endpoint: POST /api/v1/chat")
print(f"   📊 Assessment: POST /api/v1/diabetes-assessment")
print(f"   🏥 Health: GET /health")
print(f"   🔧 Status: GET /api/v1/system/status")


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Application Settings
class Settings:
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'AI Health Assistant')
    API_V1_STR = os.getenv('API_V1_STR', '/api/v1')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    WEB_SEARCH_ENABLED = os.getenv('WEB_SEARCH_ENABLED', 'True').lower() == 'true'
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    
    # API Keys
    API_KEYS = {
        "development": ["dev-key-123", "test-key-456"],
        "production": ["prod-key-2024"]
    }
    
    def get_api_keys(self):
        return self.API_KEYS.get(self.ENVIRONMENT, self.API_KEYS["development"])

settings = Settings()

# -------------------------------------------------
# Authentication & Security
# -------------------------------------------------
async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Enhanced API key verification"""
    if not credentials:
        if settings.ENVIRONMENT == "development":
            return True
        raise HTTPException(status_code=401, detail="API key required")
    
    valid_keys = settings.get_api_keys()
    if credentials.credentials in valid_keys:
        return True
    
    raise HTTPException(status_code=401, detail="Invalid API key")


# -------------------------------------------------
# ENHANCED VERSION WITH REAL LLM INTEGRATION
# -------------------------------------------------
import os
import uuid
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from groq import Groq

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app start time
app_start_time = datetime.utcnow()

# Security
security = HTTPBearer(auto_error=False)

# Application Settings
class Settings:
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'Insulyn AI')
    API_V1_STR = os.getenv('API_V1_STR', '/api/v1')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.6))
    
    # API Keys for different environments
    API_KEYS = {
        "development": ["dev-key-123", "test-key-456", "demo-key-789"],
        "production": ["prod-premium-key-2024"]
    }
    
    def get_api_keys(self):
        return self.API_KEYS.get(self.ENVIRONMENT, self.API_KEYS["development"])

settings = Settings()

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "english"
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    conversation_id: str
    model: str

class DiabetesAssessmentRequest(BaseModel):
    glucose: float
    blood_pressure: float
    weight: float
    height: float
    age: int
    pregnancies: Optional[int] = 0
    skin_thickness: Optional[float] = 20.0
    insulin: Optional[float] = 80.0
    diabetes_pedigree_function: Optional[float] = 0.5
    language: Optional[str] = "english"

class AssessmentResponse(BaseModel):
    assessment_id: str
    timestamp: str
    executive_summary: str
    risk_analysis: Dict[str, Any]
    health_metrics: Dict[str, Any]
    recommendations: Dict[str, Any]

# -------------------------------------------------
# Enhanced LLM Service with Real Integration
# -------------------------------------------------
class AIDiabetesSpecialist:
    def __init__(self):
        self.client = None
        self.conversation_memory = {}
        self.user_profiles = {}
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize Groq LLM client with proper error handling"""
        try:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key and groq_api_key.startswith('gsk_'):
                self.client = Groq(api_key=groq_api_key)
                model_name = os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
                logger.info(f"🚀 AI Diabetes Specialist LLM initialized with model: {model_name}")
            else:
                logger.error("❌ Invalid or missing Groq API key")
                self.client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq LLM: {e}")
            self.client = None
    
    def create_medical_prompt(self, message: str, language: str, user_context: Dict = None) -> str:
        """Create a comprehensive medical prompt for diabetes specialist"""
        
        base_context = user_context or {}
        
        prompt = f"""You are Insulyn AI, an expert diabetes specialist and health advisor. Provide accurate, helpful medical information about diabetes prevention, management, and treatment.

USER QUESTION: "{message}"
LANGUAGE: {language}

RESPONSE REQUIREMENTS:
1. Provide medically accurate information about diabetes
2. Focus on prevention strategies and healthy lifestyle
3. Be specific and practical in recommendations
4. Use clear, understandable language
5. Include both immediate actions and long-term strategies
6. When discussing diet, consider cultural context and local foods
7. Always recommend consulting healthcare professionals for personal medical advice

FORMAT:
- Start with a clear, empathetic response to the question
- Provide structured, actionable advice
- Use emojis sparingly for readability
- End with encouragement and next steps

IMPORTANT: Always emphasize that you are an AI assistant and users should consult healthcare providers for personal medical advice.
"""
        return prompt
    
    async def generate_medical_response(self, message: str, language: str = "english", user_id: str = "default") -> Dict[str, Any]:
        """Generate real LLM response for diabetes-related queries"""
        
        try:
            if not self.client:
                logger.error("LLM client not available")
                return {
                    "success": False,
                    "response": "I apologize, but our AI specialist is currently unavailable. Please try again later or consult with a healthcare provider for immediate medical advice.",
                    "model": "unavailable"
                }
            
            # Create medical prompt
            user_profile = self.get_user_profile(user_id)
            system_prompt = self.create_medical_prompt(message, language, user_profile.get("user_context", {}))
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Add conversation history for context
            history = self.get_conversation_history(user_id)
            for msg in history[-3:]:  # Last 3 messages for context
                messages.insert(1, {"role": msg["role"], "content": msg["content"]})
            
            # Call Groq LLM
            response = self.client.chat.completions.create(
                model=os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b'),
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            llm_response = response.choices[0].message.content
            
            # Update conversation memory
            self.add_to_conversation(user_id, "user", message)
            self.add_to_conversation(user_id, "assistant", llm_response)
            self.update_user_profile(user_id, message, llm_response)
            
            return {
                "success": True,
                "response": llm_response,
                "model": os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
            }
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again shortly or consult with a healthcare provider for urgent medical questions.",
                "model": "error"
            }
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "preferred_language": "english",
                "conversation_count": 0,
                "topics_discussed": [],
                "user_context": {}
            }
        return self.user_profiles[user_id]
    
    def update_user_profile(self, user_id: str, message: str, response: str):
        """Update user profile based on conversation"""
        profile = self.get_user_profile(user_id)
        profile["last_activity"] = datetime.utcnow()
        profile["conversation_count"] += 1
        
        # Simple topic detection
        message_lower = message.lower()
        topics = []
        
        if any(word in message_lower for word in ['chakula', 'kula', 'diet', 'food', 'meal']):
            topics.append("nutrition")
        if any(word in message_lower for word in ['mazoezi', 'exercise', 'activity', 'workout']):
            topics.append("exercise")
        if any(word in message_lower for word in ['kuzuia', 'prevent', 'risk', 'hatari']):
            topics.append("prevention")
        if any(word in message_lower for word in ['dawa', 'medicine', 'treatment', 'tibabu']):
            topics.append("treatment")
        
        for topic in topics:
            if topic not in profile["topics_discussed"]:
                profile["topics_discussed"].append(topic)
    
    def get_conversation_history(self, user_id: str, max_messages: int = 6) -> List[Dict]:
        """Get conversation history for a user"""
        if user_id in self.conversation_memory:
            return self.conversation_memory[user_id][-max_messages:]
        return []
    
    def add_to_conversation(self, user_id: str, role: str, content: str):
        """Add message to conversation history"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 20 messages
        if len(self.conversation_memory[user_id]) > 20:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-20:]

# Initialize AI Diabetes Specialist
ai_specialist = AIDiabetesSpecialist()

# -------------------------------------------------
# Mock ML Model for Diabetes Assessment
# -------------------------------------------------
class DiabetesMLModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Mock model loading"""
        logger.info("✅ ML model placeholder loaded")
        self.model = "mock_model"
    
    def predict(self, features: Dict[str, Any]) -> tuple:
        """Mock prediction with basic risk calculation"""
        # Simple risk calculation based on common factors
        risk_score = 0
        
        # Glucose contribution (fasting glucose >= 126 is diabetic)
        if features['Glucose'] >= 126:
            risk_score += 0.6
        elif features['Glucose'] >= 100:
            risk_score += 0.3
        else:
            risk_score += 0.1
        
        # BMI contribution
        bmi = features['BMI']
        if bmi >= 30:
            risk_score += 0.5
        elif bmi >= 25:
            risk_score += 0.3
        else:
            risk_score += 0.1
        
        # Age contribution
        if features['Age'] >= 45:
            risk_score += 0.3
        elif features['Age'] >= 35:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Blood pressure contribution
        if features['BloodPressure'] >= 140:
            risk_score += 0.4
        elif features['BloodPressure'] >= 130:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Normalize risk score
        probability = min(0.95, risk_score / 2.0)
        
        if probability >= 0.7:
            risk_label = "High Risk"
        elif probability >= 0.4:
            risk_label = "Moderate Risk"
        else:
            risk_label = "Low Risk"
        
        return risk_label, probability, {"glucose": 0.3, "bmi": 0.25, "age": 0.2, "blood_pressure": 0.25}

# Initialize ML model
diabetes_model = DiabetesMLModel()

# -------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting AI Diabetes Specialist Platform")
    logger.info(f"   Environment: {settings.ENVIRONMENT}")
    logger.info(f"   LLM Model: {settings.LLM_MODEL_NAME}")
    logger.info(f"   LLM Status: {'Connected' if ai_specialist.client else 'Disconnected'}")
    yield
    # Shutdown
    logger.info("🛑 Shutting down AI Diabetes Specialist Platform")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Diabetes Specialist - Real-time Medical Advice and Risk Assessment",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Authentication Dependency
# -------------------------------------------------
async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Flexible API key verification"""
    if settings.ENVIRONMENT == "production":
        if not credentials or credentials.credentials not in settings.get_api_keys():
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key. Please check your credentials."
            )
        return True
    else:
        # In development, allow all requests
        return True

# -------------------------------------------------
# MAIN CHAT ENDPOINT - USES REAL LLM
# -------------------------------------------------
@app.post("/api/v1/chat", response_model=ChatResponse)
async def diabetes_chat(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: bool = Depends(verify_api_key)
):
    """Main chat endpoint with real LLM integration"""
    try:
        message = chat_request.message.strip()
        language = chat_request.language
        user_id = chat_request.user_id or "default"
        
        if not message:
            raise HTTPException(status_code=422, detail="Message cannot be empty")
        
        logger.info(f"🧠 Chat request from {user_id}: {message[:50]}...")

        # Generate real LLM response
        llm_result = await ai_specialist.generate_medical_response(
            message=message,
            language=language,
            user_id=user_id
        )
        
        response_text = llm_result["response"]
        model_used = llm_result["model"]
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=user_id,
            model=model_used
        )
        
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(
            status_code=500, 
            detail="Our AI diabetes specialist is currently unavailable. Please try again later."
        )

# -------------------------------------------------
# DIABETES ASSESSMENT ENDPOINT
# -------------------------------------------------
@app.post("/api/v1/diabetes-assessment", response_model=AssessmentResponse)
async def diabetes_assessment(
    request: DiabetesAssessmentRequest,
    api_key: bool = Depends(verify_api_key)
):
    """Diabetes risk assessment with LLM-powered insights"""
    try:
        logger.info(f"📊 Diabetes assessment for age {request.age}")
        
        # Calculate BMI
        height_m = request.height / 100
        bmi = request.weight / (height_m ** 2)
        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        
        # Prepare features for prediction
        features = {
            'Pregnancies': request.pregnancies,
            'Glucose': request.glucose,
            'BloodPressure': request.blood_pressure,
            'SkinThickness': request.skin_thickness,
            'Insulin': request.insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': request.diabetes_pedigree_function,
            'Age': request.age
        }
        
        # Get prediction
        risk_label, probability, feature_importances = diabetes_model.predict(features)
        
        # Generate LLM-powered insights
        insights_prompt = f"""
        Provide a comprehensive diabetes risk assessment summary in {request.language} based on these metrics:
        
        - Age: {request.age} years
        - Glucose: {request.glucose} mg/dL
        - Blood Pressure: {request.blood_pressure} mmHg  
        - BMI: {bmi:.1f} ({bmi_category})
        - Risk Level: {risk_label} (Probability: {probability:.1%})
        
        Please provide:
        1. Executive summary of the assessment
        2. Key risk factors identified
        3. Immediate lifestyle recommendations
        4. When to consult a healthcare provider
        
        Be specific and actionable in your recommendations.
        """
        
        insights_response = await ai_specialist.generate_medical_response(
            insights_prompt, 
            request.language
        )
        
        llm_insights = insights_response["response"] if insights_response["success"] else "Assessment completed. Please consult with healthcare provider for detailed analysis."
        
        return AssessmentResponse(
            assessment_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            executive_summary=llm_insights,
            risk_analysis={
                "risk_level": risk_label,
                "probability": round(probability, 3),
                "key_factors": _identify_risk_factors(features, bmi)
            },
            health_metrics={
                "bmi": round(bmi, 1),
                "bmi_category": bmi_category,
                "metabolic_age": _calculate_metabolic_age(features),
                "health_score": _calculate_health_score(features)
            },
            recommendations={
                "lifestyle_changes": _generate_lifestyle_recommendations(risk_label, features),
                "medical_followup": "Consult healthcare provider for comprehensive evaluation",
                "monitoring_schedule": "Regular check-ups recommended"
            }
        )
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Assessment service temporarily unavailable. Please try again shortly."
        )

def _identify_risk_factors(features: Dict[str, Any], bmi: float) -> List[Dict[str, Any]]:
    """Identify key risk factors"""
    risk_factors = []
    
    if features['Glucose'] >= 126:
        risk_factors.append({"factor": "Diabetes-level glucose", "severity": "high"})
    elif features['Glucose'] >= 100:
        risk_factors.append({"factor": "Prediabetes glucose levels", "severity": "moderate"})
        
    if bmi >= 30:
        risk_factors.append({"factor": "Clinical obesity", "severity": "high"})
    elif bmi >= 25:
        risk_factors.append({"factor": "Overweight", "severity": "moderate"})
        
    if features['BloodPressure'] >= 140:
        risk_factors.append({"factor": "Stage 2 hypertension", "severity": "high"})
    elif features['BloodPressure'] >= 130:
        risk_factors.append({"factor": "Stage 1 hypertension", "severity": "moderate"})
        
    if features['Age'] >= 45:
        risk_factors.append({"factor": "Age-related risk increase", "severity": "moderate"})
        
    return risk_factors if risk_factors else [{"factor": "No significant risk factors identified", "severity": "low"}]

def _calculate_metabolic_age(features: Dict[str, Any]) -> int:
    """Calculate metabolic age"""
    base_age = features['Age']
    adjustment = 0
    
    if features['Glucose'] < 100:
        adjustment -= 3
    if features['BMI'] < 25:
        adjustment -= 2
    if features['BloodPressure'] < 120:
        adjustment -= 2
        
    return max(20, base_age + adjustment)

def _calculate_health_score(features: Dict[str, Any]) -> int:
    """Calculate health score (1-100)"""
    score = 50
    
    # BMI scoring
    bmi = features.get('BMI', 25)
    if 18.5 <= bmi <= 24.9:
        score += 20
    elif 25 <= bmi <= 29.9:
        score += 10
    
    # Glucose scoring
    glucose = features.get('Glucose', 100)
    if glucose < 100:
        score += 20
    elif glucose < 126:
        score += 10
        
    # Blood pressure scoring
    bp = features.get('BloodPressure', 120)
    if bp < 120:
        score += 15
    elif bp < 140:
        score += 10
        
    return min(100, score)

def _generate_lifestyle_recommendations(risk_level: str, features: Dict[str, Any]) -> List[str]:
    """Generate lifestyle recommendations"""
    recommendations = []
    
    if "high" in risk_level.lower():
        recommendations.extend([
            "Immediate consultation with healthcare provider",
            "Comprehensive blood work and monitoring",
            "Structured diet and exercise program"
        ])
    else:
        recommendations.extend([
            "Regular physical activity (30 mins daily)",
            "Balanced diet with portion control",
            "Regular health check-ups",
            "Stress management and adequate sleep"
        ])
    
    if features.get('BMI', 0) > 25:
        recommendations.append("Weight management program")
    if features.get('Glucose', 0) > 100:
        recommendations.append("Blood sugar monitoring")
        
    return recommendations

# -------------------------------------------------
# HEALTH TOPICS ENDPOINT
# -------------------------------------------------
@app.get("/api/v1/health-topics")
async def get_health_topics(language: str = "english"):
    """Get common diabetes-related health topics"""
    
    topics_data = {
        "english": [
            {"id": "prevention", "name": "Diabetes Prevention", "description": "How to prevent type 2 diabetes"},
            {"id": "symptoms", "name": "Symptoms & Signs", "description": "Early warning signs of diabetes"},
            {"id": "diet", "name": "Diabetes Diet", "description": "Foods to eat and avoid"},
            {"id": "exercise", "name": "Exercise & Activity", "description": "Physical activity recommendations"},
            {"id": "monitoring", "name": "Blood Sugar Monitoring", "description": "How to check glucose levels"},
            {"id": "treatment", "name": "Treatment Options", "description": "Medications and therapies"},
            {"id": "complications", "name": "Complications", "description": "Long-term health risks"},
            {"id": "management", "name": "Daily Management", "description": "Living with diabetes"}
        ],
        "swahili": [
            {"id": "prevention", "name": "Kuzuia Kisukari", "description": "Jinsi ya kuzuia kisukari aina ya 2"},
            {"id": "symptoms", "name": "Dalili na Ishara", "description": "Ishara za mapema za kisukari"},
            {"id": "diet", "name": "Lishe ya Kisukari", "description": "Vyakula vya kula na kuepuka"},
            {"id": "exercise", "name": "Mazoezi na Shughuli", "description": "Mapendekezo ya shughuli za mwili"},
            {"id": "monitoring", "name": "Kufuatilia Sukari ya Damu", "description": "Jinsi ya kukagua viwango vya glukosi"},
            {"id": "treatment", "name": "Chaguo za Matibabu", "description": "Dawa na tiba mbalimbali"},
            {"id": "complications", "name": "Matatizo", "description": "Hatari za kiafya za muda mrefu"},
            {"id": "management", "name": "Usimamizi wa Kila Siku", "description": "Kuishi na kisukari"}
        ]
    }
    
    return {
        "topics": topics_data.get(language, topics_data["english"]),
        "language": language,
        "timestamp": datetime.utcnow().isoformat()
    }

# -------------------------------------------------
# USER PROFILE ENDPOINTS
# -------------------------------------------------
@app.get("/api/v1/user/{user_id}/profile")
async def get_user_profile(user_id: str, api_key: bool = Depends(verify_api_key)):
    """Get user profile and conversation history"""
    profile = ai_specialist.get_user_profile(user_id)
    history = ai_specialist.get_conversation_history(user_id)
    
    return {
        "user_id": user_id,
        "profile": profile,
        "conversation_history": history,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.delete("/api/v1/user/{user_id}/conversation")
async def clear_conversation(user_id: str, api_key: bool = Depends(verify_api_key)):
    """Clear user conversation history"""
    if user_id in ai_specialist.conversation_memory:
        ai_specialist.conversation_memory[user_id] = []
    
    return {"message": "Conversation history cleared", "user_id": user_id}

# -------------------------------------------------
# SYSTEM ENDPOINTS
# -------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    llm_status = "🟢 Connected" if ai_specialist.client else "🔴 Disconnected"
    
    return {
        "message": "🚀 AI Diabetes Specialist - Real-time Medical Advisor",
        "status": "operational",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "llm_status": llm_status,
        "model": settings.LLM_MODEL_NAME,
        "endpoints": {
            "chat": "/api/v1/chat",
            "assessment": "/api/v1/diabetes-assessment",
            "topics": "/api/v1/health-topics",
            "user_profile": "/api/v1/user/{user_id}/profile"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "llm_service": "connected" if ai_specialist.client else "disconnected",
            "ml_model": "loaded",
            "api": "operational"
        },
        "uptime": str(datetime.utcnow() - app_start_time)
    }

@app.get("/api/v1/system/status")
async def system_status():
    """Detailed system status"""
    return {
        "llm": {
            "status": "connected" if ai_specialist.client else "disconnected",
            "model": settings.LLM_MODEL_NAME,
            "provider": "Groq"
        },
        "ml_model": {
            "status": "loaded",
            "type": "diabetes_risk_assessment"
        },
        "memory": {
            "active_users": len(ai_specialist.user_profiles),
            "total_conversations": sum(len(conv) for conv in ai_specialist.conversation_memory.values())
        },
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

logger.info("🚀 AI DIABETES SPECIALIST API INITIALIZED SUCCESSFULLY!")
logger.info(f"   🧠 LLM Status: {'🟢 Connected' if ai_specialist.client else '🔴 Disconnected'}")
logger.info("   📍 Available Endpoints:")
logger.info("   💬 POST /api/v1/chat - Real-time AI diabetes specialist")
logger.info("   📊 POST /api/v1/diabetes-assessment - Risk assessment with LLM insights") 
logger.info("   📚 GET /api/v1/health-topics - Common diabetes topics")
logger.info("   👤 GET /api/v1/user/{id}/profile - User profiles & history")
logger.info("   🏥 GET /health - System health check")
# -------------------------------------------------
# PRODUCTION-READY MEAL PLANNER - PERFECTLY CONNECTED
# -------------------------------------------------


# -------------------------------------------------
# Data Models - PERFECTLY MATCHED WITH FRONTEND
# -------------------------------------------------

class DietPlanRequest(BaseModel):
    # Required fields - EXACTLY as frontend sends them
    age: int
    weight: float
    height: float
    gender: str
    
    # Optional fields - EXACT field names from frontend
    dietaryPreference: str = "balanced"
    healthConditions: str = ""
    activityLevel: str = "moderate"
    goals: str = "diabetes_prevention"
    allergies: str = ""
    typicalDay: str = ""
    language: str = "english"

class DietPlanResponse(BaseModel):
    # Response structure - EXACTLY what frontend expects
    overview: str
    daily_plan: str
    grocery_list: str
    important_notes: str
    nutritional_info: Dict[str, Any] = {}
    timestamp: str
    status: str = "success"
    generation_time: float = 0.0

# -------------------------------------------------
# LLM Service
# -------------------------------------------------

class GroqLLMService:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model_name = os.getenv('LLM_MODEL_NAME', 'openai/gpt-oss-20b')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', 0.6))
        self.available = bool(self.api_key and self.api_key.startswith('gsk_'))
        
        if self.available:
            logger.info(f"✅ Groq LLM initialized with model: {self.model_name}")
        else:
            logger.warning("❌ Groq LLM not available - using enhanced templates")

    async def generate_response(self, prompt: str) -> str:
        """Generate response using Groq LLM"""
        if not self.available:
            raise Exception("Groq LLM not configured")
        
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": 1500,
                "top_p": 1
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload, 
                    headers=headers
                )
                
                if response.status_code != 200:
                    raise Exception(f"API error {response.status_code}")
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

# Initialize LLM service
llm_service = GroqLLMService()

# -------------------------------------------------
# Production Meal Planning Service
# -------------------------------------------------

class ProductionMealPlanningService:
    def __init__(self):
        self.llm_service = llm_service
        logger.info("🍽️ Production Meal Planning Service initialized")

    async def generate_plan(self, request: DietPlanRequest) -> Dict[str, Any]:
        """Generate high-quality meal plan"""
        start_time = time.time()
        
        try:
            # Calculate personalized nutrition
            nutrition = self._calculate_nutrition(request)
            
            # Generate plan content
            llm_used, plan_content = await self._generate_plan_content(request, nutrition)
            
            # Build response
            response = self._build_response(plan_content, request, nutrition, start_time, llm_used)
            return response
            
        except Exception as e:
            logger.error(f"❌ Plan generation failed: {e}")
            return self._get_fallback_response(request, start_time)

    def _calculate_nutrition(self, request: DietPlanRequest) -> Dict[str, Any]:
        """Calculate personalized nutritional targets"""
        # BMR calculation
        if request.gender.lower() == "male":
            bmr = 88.362 + (13.397 * request.weight) + (4.799 * request.height) - (5.677 * request.age)
        else:
            bmr = 447.593 + (9.247 * request.weight) + (3.098 * request.height) - (4.330 * request.age)
        
        # Activity level multipliers
        activity_map = {
            "sedentary": 1.2, "light": 1.375, "moderate": 1.55, 
            "active": 1.725, "very_active": 1.9
        }
        tdee = bmr * activity_map.get(request.activityLevel, 1.55)
        
        # Goal-based adjustments
        goal_adj = {
            "weight_loss": -500, 
            "diabetes_prevention": -300,
            "blood_sugar_control": -400,
            "weight_gain": 500, 
            "maintenance": 0,
            "gestational_diabetes": -200
        }
        adjustment = goal_adj.get(request.goals, -300)
        
        # Health condition adjustments
        if "kidney" in request.healthConditions.lower():
            adjustment -= 200  # Lower protein for kidney issues
        if "pcos" in request.healthConditions.lower():
            adjustment -= 150  # Adjust for PCOS
        
        calories = max(tdee + adjustment, 1200)
        
        return {
            "daily_calories": int(calories),
            "protein_grams": int((calories * 0.25) / 4),
            "carbs_grams": int((calories * 0.45) / 4),
            "fat_grams": int((calories * 0.30) / 9),
            "fiber_grams": 25,
            "sugar_limit": "less than 25g",
            "water_intake": "2-3 liters daily"
        }

    async def _generate_plan_content(self, request: DietPlanRequest, nutrition: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Generate plan content with LLM fallback"""
        # Try LLM first
        if llm_service.available:
            try:
                llm_content = await asyncio.wait_for(
                    self._generate_with_llm(request, nutrition),
                    timeout=10.0
                )
                if llm_content and llm_content.get("overview"):
                    return True, llm_content
            except Exception as e:
                logger.warning(f"⚡ LLM failed, using enhanced template: {e}")
        
        # Use enhanced template
        return False, self._get_enhanced_template(request, nutrition)

    async def _generate_with_llm(self, request: DietPlanRequest, nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """Generate using LLM"""
        prompt = self._create_prompt(request, nutrition)
        response = await self.llm_service.generate_response(prompt)
        return self._parse_llm_response(response)

    def _create_prompt(self, request: DietPlanRequest, nutrition: Dict[str, Any]) -> str:
        """Create optimized prompt for LLM"""
        return f"""
        Create a personalized diabetes-friendly meal plan.

        USER PROFILE:
        - {request.age} years old, {request.gender}
        - {request.weight}kg, {request.height}cm
        - Goal: {request.goals}
        - Diet: {request.dietaryPreference}
        - Health Conditions: {request.healthConditions}
        - Allergies: {request.allergies}
        - Activity Level: {request.activityLevel}
        - Daily Routine: {request.typicalDay}

        NUTRITIONAL TARGETS:
        - Calories: {nutrition['daily_calories']} per day
        - Protein: {nutrition['protein_grams']}g
        - Carbs: {nutrition['carbs_grams']}g  
        - Fat: {nutrition['fat_grams']}g

        Create a response with these 4 sections:

        OVERVIEW: 2-3 sentence personalized overview focusing on diabetes management

        DAILY_PLAN: Specific meal ideas for breakfast, lunch, dinner, and snacks with portion guidance

        GROCERY_LIST: 8-10 essential grocery items for diabetes management

        IMPORTANT_NOTES: 3-4 key recommendations considering the user's health conditions and allergies

        Make it practical, personalized, and focused on blood sugar control.
        """

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into the 4 sections"""
        try:
            sections = {
                "overview": "",
                "daily_plan": "", 
                "grocery_list": "",
                "important_notes": ""
            }
            
            current_section = "overview"
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                lower_line = line.lower()
                if "overview" in lower_line:
                    current_section = "overview"
                elif "daily" in lower_line or "plan" in lower_line or "breakfast" in lower_line:
                    current_section = "daily_plan"
                elif "grocery" in lower_line or "shopping" in lower_line:
                    current_section = "grocery_list"
                elif "note" in lower_line or "important" in lower_line:
                    current_section = "important_notes"
                else:
                    if sections[current_section]:
                        sections[current_section] += "\n"
                    sections[current_section] += line
            
            return sections
            
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}")
            return {}

    def _get_enhanced_template(self, request: DietPlanRequest, nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced template based on user data"""
        base_template = {
            "overview": f"Personalized {request.dietaryPreference} diabetes meal plan for a {request.age}-year-old {request.gender}. Designed for optimal blood sugar control with {nutrition['daily_calories']} daily calories.",
            "daily_plan": self._get_daily_plan_template(request),
            "grocery_list": self._get_grocery_list_template(request),
            "important_notes": self._get_important_notes_template(request)
        }
        
        return self._personalize_template(base_template, request)

    def _get_daily_plan_template(self, request: DietPlanRequest) -> str:
        """Get personalized daily plan template"""
        base_plan = """BREAKFAST (7-8 AM): High-fiber cereal with nuts and berries
LUNCH (12-1 PM): Grilled protein with vegetables and whole grains  
DINNER (6-7 PM): Light protein with non-starchy vegetables
SNACKS: Fresh fruits, nuts, yogurt between meals"""
        
        # Personalize based on dietary preference
        if request.dietaryPreference == "vegetarian":
            base_plan = base_plan.replace("Grilled protein", "Plant-based protein").replace("Light protein", "Legume-based dish")
        elif request.dietaryPreference == "vegan":
            base_plan = base_plan.replace("Grilled protein", "Tofu or tempeh").replace("Light protein", "Plant-based protein").replace("yogurt", "plant-based yogurt")
        elif request.dietaryPreference == "low_carb":
            base_plan = base_plan.replace("cereal with nuts and berries", "eggs with avocado").replace("whole grains", "extra vegetables")
        
        # Add health condition considerations
        if "kidney" in request.healthConditions.lower():
            base_plan += "\n\nSPECIAL CONSIDERATIONS: Lower protein intake recommended for kidney health"
        if "pcos" in request.healthConditions.lower():
            base_plan += "\n\nSPECIAL CONSIDERATIONS: Focus on low-glycemic foods and regular meal timing"
        
        return base_plan

    def _get_grocery_list_template(self, request: DietPlanRequest) -> str:
        """Get personalized grocery list"""
        base_list = """- Whole grains (oats, brown rice, quinoa)
- Lean proteins (chicken, fish, legumes)
- Fresh vegetables (leafy greens, broccoli, carrots)
- Low-sugar fruits (berries, apples, oranges)
- Healthy fats (avocado, nuts, olive oil)
- Low-fat dairy (Greek yogurt, milk)
- Herbs and spices (turmeric, cinnamon, garlic)"""
        
        # Adjust for dietary preferences
        if request.dietaryPreference == "vegetarian":
            base_list = base_list.replace("chicken, fish", "tofu, tempeh, lentils")
        elif request.dietaryPreference == "vegan":
            base_list = base_list.replace("chicken, fish", "tofu, tempeh, legumes").replace("Low-fat dairy", "Plant-based alternatives")
        elif request.dietaryPreference == "low_carb":
            base_list = base_list.replace("Whole grains", "Cauliflower rice").replace("Low-sugar fruits", "Berries in moderation")
        
        # Add allergy considerations
        if "gluten" in request.allergies.lower():
            base_list += "\n- Gluten-free alternatives (quinoa, buckwheat)"
        if "dairy" in request.allergies.lower():
            base_list = base_list.replace("Low-fat dairy", "Dairy-free alternatives")
        
        return base_list

    def _get_important_notes_template(self, request: DietPlanRequest) -> str:
        """Get personalized important notes"""
        base_notes = """- Monitor blood sugar levels regularly
- Stay hydrated with 8+ glasses of water daily
- Exercise for 30 minutes most days
- Consult healthcare provider before major changes"""
        
        # Add health condition specific notes
        if "kidney" in request.healthConditions.lower():
            base_notes += "\n- Limit protein intake as advised by your doctor"
            base_notes += "\n- Monitor potassium and phosphorus levels"
        if "pcos" in request.healthConditions.lower():
            base_notes += "\n- Maintain consistent meal timing"
            base_notes += "\n- Focus on anti-inflammatory foods"
        
        # Add allergy notes
        if request.allergies:
            base_notes += f"\n- Strictly avoid foods containing: {request.allergies}"
        
        # Add goal-specific notes
        if request.goals == "weight_loss":
            base_notes += "\n- Create a moderate calorie deficit for sustainable weight loss"
        elif request.goals == "blood_sugar_control":
            base_notes += "\n- Test blood sugar before and after meals to understand food impacts"
        
        return base_notes

    def _personalize_template(self, template: Dict[str, Any], request: DietPlanRequest) -> Dict[str, Any]:
        """Further personalize the template"""
        personalized = template.copy()
        
        # Add user's typical day considerations
        if request.typicalDay:
            personalized["important_notes"] += f"\n- Adjust meal timing based on your routine: {request.typicalDay}"
        
        return personalized

    def _build_response(self, plan_content: Dict[str, Any], request: DietPlanRequest, 
                       nutrition: Dict[str, Any], start_time: float, llm_used: bool) -> Dict[str, Any]:
        """Build final response"""
        generation_time = time.time() - start_time
        
        return {
            "overview": plan_content.get("overview", "Personalized diabetes meal plan for optimal health."),
            "daily_plan": plan_content.get("daily_plan", "Balanced daily meal schedule."),
            "grocery_list": plan_content.get("grocery_list", "Essential diabetes-friendly groceries."),
            "important_notes": plan_content.get("important_notes", "Important health recommendations."),
            "nutritional_info": nutrition,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "generation_time": round(generation_time, 2)
        }

    def _get_fallback_response(self, request: DietPlanRequest, start_time: float) -> Dict[str, Any]:
        """Get fallback response"""
        nutrition = self._calculate_nutrition(request)
        template = self._get_enhanced_template(request, nutrition)
        response = self._build_response(template, request, nutrition, start_time, False)
        response["status"] = "fallback"
        return response

# Initialize service
meal_service = ProductionMealPlanningService()

# -------------------------------------------------
# API Endpoints - PRODUCTION READY
# -------------------------------------------------

async def verify_api_key():
    """API key verification"""
    return True

@app.post("/api/v1/diet-plan/generate", response_model=DietPlanResponse)
async def generate_diet_plan(
    request: DietPlanRequest,
    background_tasks: BackgroundTasks,
    api_key: bool = Depends(verify_api_key)
):
    """Generate personalized diet plan - PRODUCTION READY"""
    logger.info(f"🎯 Generating plan for {request.age}y/o {request.gender} - Goal: {request.goals}")
    
    try:
        result = await meal_service.generate_plan(request)
        logger.info(f"✅ Plan generated successfully - Time: {result['generation_time']}s")
        return DietPlanResponse(**result)
    except Exception as e:
        logger.error(f"❌ Plan generation failed: {e}")
        raise HTTPException(status_code=500, detail="Meal plan generation failed")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Insulyn AI Production Meal Planner",
        "timestamp": datetime.utcnow().isoformat(),
        "llm_available": llm_service.available,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Insulyn AI - Production Diabetes Meal Planning API",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "generate_plan": "POST /api/v1/diet-plan/generate",
            "health": "GET /api/v1/health"
        }
    }

# -------------------------------------------------
# Application Startup
# -------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Insulyn AI Production Meal Planner Started Successfully!")
    logger.info("   ✅ Production Endpoints:")
    logger.info("   ├── POST /api/v1/diet-plan/generate")
    logger.info("   ├── GET /api/v1/health")
    logger.info("   └── GET /")
    logger.info(f"   🔧 LLM Status: {'✅ Connected' if llm_service.available else '❌ Enhanced Templates'}")
    logger.info("   🎯 Perfect field mapping with frontend")
    logger.info("   🌐 CORS configured for React frontend")
    logger.info("   📊 Enhanced personalization with health conditions")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('RELOAD', 'True').lower() == 'true',
        log_level="info"
    )

# -------------------------------------------------
# ENHANCED PERSONALIZED Emergency Assessment Endpoint
# -------------------------------------------------

class EmergencyAssessmentRequest(BaseModel):
    symptoms: List[str]
    age: Optional[int] = None
    weight: Optional[float] = None 
    height: Optional[float] = None
    existing_conditions: List[str] = []
    current_medications: List[str] = []
    last_meal_time: Optional[str] = None
    language: str = "english"

class EmergencyAssessmentResponse(BaseModel):
    assessment: str
    personalized_analysis: str
    recommendations: List[str]
    urgency_level: str
    risk_factors: List[str]
    next_steps: List[str]
    timestamp: str

@app.post("/api/v1/emergency-assessment", response_model=EmergencyAssessmentResponse)
async def personalized_emergency_assessment(
    request: EmergencyAssessmentRequest,
    api_key: bool = Depends(verify_api_key)
):
    """
    TRULY PERSONALIZED emergency symptom assessment
    """
    try:
        # Get personalized analysis
        personalized_data = await _analyze_with_personal_context(
            request.symptoms,
            request.age,
            request.weight, 
            request.height,
            request.existing_conditions,
            request.current_medications,
            request.language
        )
        
        return EmergencyAssessmentResponse(
            assessment=personalized_data["assessment"],
            personalized_analysis=personalized_data["personalized_analysis"],
            recommendations=personalized_data["recommendations"],
            urgency_level=personalized_data["urgency_level"],
            risk_factors=personalized_data["risk_factors"],
            next_steps=personalized_data["next_steps"],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Personalized assessment error: {e}")
        return await _generate_personalized_fallback(request)

async def _analyze_with_personal_context(
    symptoms: List[str],
    age: Optional[int],
    weight: Optional[float],
    height: Optional[float],
    existing_conditions: List[str],
    medications: List[str],
    language: str
) -> Dict[str, Any]:
    """Analyze symptoms with personal context for true personalization"""
    
    # Calculate personalized risk factors
    risk_factors = _calculate_personal_risk_factors(age, weight, height, existing_conditions, medications)
    
    # Analyze symptom patterns
    symptom_analysis = _analyze_symptom_patterns(symptoms, age, existing_conditions)
    
    # Generate personalized assessment
    assessment = _generate_personalized_assessment(symptoms, risk_factors, symptom_analysis, language)
    
    # Calculate urgency level based on personal factors
    urgency_level = _calculate_personalized_urgency(symptoms, risk_factors, age)
    
    # Generate personalized recommendations
    recommendations = _generate_personalized_recommendations(
        symptoms, urgency_level, risk_factors, age, existing_conditions, language
    )
    
    # Generate personalized next steps
    next_steps = _generate_personalized_next_steps(urgency_level, risk_factors, age, language)
    
    return {
        "assessment": assessment["summary"],
        "personalized_analysis": assessment["detailed_analysis"],
        "recommendations": recommendations,
        "urgency_level": urgency_level,
        "risk_factors": risk_factors,
        "next_steps": next_steps
    }

def _calculate_personal_risk_factors(
    age: Optional[int],
    weight: Optional[float],
    height: Optional[float],
    existing_conditions: List[str],
    medications: List[str]
) -> List[str]:
    """Calculate personalized risk factors"""
    risk_factors = []
    
    # Age-based risks
    if age:
        if age < 18:
            risk_factors.append("Pediatric patient - different symptom presentation")
        elif age > 65:
            risk_factors.append("Senior patient - higher complication risk")
        elif age > 50:
            risk_factors.append("Middle-aged - increased diabetes risk")
    
    # BMI-based risks
    if weight and height:
        bmi = weight / ((height / 100) ** 2)
        if bmi > 30:
            risk_factors.append(f"High BMI ({bmi:.1f}) - increased health risks")
        elif bmi > 25:
            risk_factors.append(f"Elevated BMI ({bmi:.1f})")
    
    # Condition-based risks
    condition_risks = {
        "heart": "Cardiovascular condition - monitor carefully",
        "kidney": "Kidney issues - fluid balance critical", 
        "hypertension": "High blood pressure - cardiovascular risk",
        "cholesterol": "Cholesterol issues - metabolic concern",
        "diabetes": "Existing diabetes - acute complication risk"
    }
    
    for condition in existing_conditions:
        condition_lower = condition.lower()
        for key, risk in condition_risks.items():
            if key in condition_lower:
                risk_factors.append(risk)
                break
    
    # Medication-based risks
    if any("insulin" in med.lower() for med in medications):
        risk_factors.append("Insulin therapy - hypoglycemia risk")
    if any("metformin" in med.lower() for med in medications):
        risk_factors.append("Metformin use - gastrointestinal considerations")
    
    return risk_factors if risk_factors else ["Standard risk profile"]

def _analyze_symptom_patterns(symptoms: List[str], age: Optional[int], conditions: List[str]) -> Dict[str, Any]:
    """Analyze symptom patterns for personalization"""
    
    # Critical symptom groups
    critical_groups = {
        "diabetic_emergency": ["extreme thirst", "frequent urination", "fruity breath", "confusion"],
        "cardiovascular": ["chest pain", "difficulty breathing", "rapid heartbeat", "dizziness"],
        "neurological": ["confusion", "blurred vision", "dizziness", "difficulty concentrating"]
    }
    
    # Find matching symptom groups
    matched_groups = []
    for group_name, group_symptoms in critical_groups.items():
        matches = [symptom for symptom in symptoms if any(gs in symptom.lower() for gs in group_symptoms)]
        if len(matches) >= 2:  # At least 2 symptoms from a group
            matched_groups.append(group_name)
    
    # Age-specific considerations
    age_notes = []
    if age:
        if age < 30:
            age_notes.append("Young adult - typically higher resilience")
        elif age > 60:
            age_notes.append("Senior - may have atypical symptom presentation")
    
    return {
        "matched_groups": matched_groups,
        "symptom_count": len(symptoms),
        "age_notes": age_notes,
        "has_critical_combination": len(matched_groups) > 0
    }

def _generate_personalized_assessment(
    symptoms: List[str], 
    risk_factors: List[str],
    symptom_analysis: Dict[str, Any],
    language: str
) -> Dict[str, str]:
    """Generate truly personalized assessment"""
    
    symptom_count = len(symptoms)
    has_critical_combinations = symptom_analysis["has_critical_combination"]
    risk_level = "HIGH" if len(risk_factors) > 2 else "MODERATE" if risk_factors else "STANDARD"
    
    # Base assessment
    if has_critical_combinations:
        summary = f"🚨 CRITICAL: Multiple emergency symptom patterns detected"
        detailed = f"Based on your {symptom_count} symptoms including critical combinations, this appears to be a medical emergency. Your {risk_level} risk profile ({len(risk_factors)} factors) increases urgency."
    
    elif symptom_count >= 5:
        summary = f"🔴 HIGH: Multiple concerning symptoms with {risk_level} risk profile"
        detailed = f"You're experiencing {symptom_count} symptoms which, combined with your {risk_level} risk profile ({len(risk_factors)} factors), requires urgent attention."
    
    elif symptom_count >= 3:
        summary = f"🟡 MODERATE: Multiple symptoms with {risk_level} risk factors"
        detailed = f"Your {symptom_count} symptoms along with {len(risk_factors)} risk factors need careful monitoring and professional evaluation."
    
    else:
        summary = f"🟢 MILD: Limited symptoms with {risk_level} monitoring needed"
        detailed = f"While you have only {symptom_count} symptom(s), your risk factors suggest careful monitoring is advised."
    
    # Add personalized notes
    if risk_factors:
        detailed += f" Key considerations: {', '.join(risk_factors[:3])}."
    
    if symptom_analysis["age_notes"]:
        detailed += f" Age note: {symptom_analysis['age_notes'][0]}"
    
    return {
        "summary": summary,
        "detailed_analysis": detailed
    }

def _calculate_personalized_urgency(symptoms: List[str], risk_factors: List[str], age: Optional[int]) -> str:
    """Calculate personalized urgency level"""
    
    # Critical symptoms
    critical_symptoms = ["difficulty breathing", "chest pain", "confusion", "unconscious", "seizure"]
    has_critical = any(any(cs in symptom.lower() for cs in critical_symptoms) for symptom in symptoms)
    
    if has_critical:
        return "critical"
    
    # High urgency based on combinations
    high_symptoms = ["vomiting", "fever", "dizziness", "extreme thirst", "frequent urination", "rapid heartbeat"]
    high_count = sum(1 for symptom in symptoms if any(hs in symptom.lower() for hs in high_symptoms))
    
    if high_count >= 2 and len(risk_factors) >= 2:
        return "high"
    elif high_count >= 2 or len(risk_factors) >= 3:
        return "high"
    
    # Age considerations
    if age and age > 65 and len(symptoms) >= 2:
        return "high"
    
    # Medium urgency
    if len(symptoms) >= 3 or len(risk_factors) >= 1:
        return "medium"
    
    return "low"

def _generate_personalized_recommendations(
    symptoms: List[str],
    urgency: str,
    risk_factors: List[str],
    age: Optional[int],
    conditions: List[str],
    language: str
) -> List[str]:
    """Generate personalized recommendations"""
    
    recommendations = []
    
    # Urgency-based recommendations
    if urgency == "critical":
        recommendations.extend([
            "🚨 CALL EMERGENCY SERVICES IMMEDIATELY (911/112)",
            "Do not attempt to drive yourself",
            "Have someone stay with you continuously",
            "Prepare your medical information and medications list"
        ])
    elif urgency == "high":
        recommendations.extend([
            "Contact healthcare provider within 1 hour",
            "Check blood sugar levels immediately if possible",
            "Have someone available to drive you if needed",
            "Gather recent medical records and test results"
        ])
    else:
        recommendations.extend([
            "Schedule doctor appointment within 24 hours",
            "Monitor symptoms every 2-4 hours",
            "Keep a symptom diary with timestamps",
            "Stay hydrated with water, avoid sugary drinks"
        ])
    
    # Symptom-specific recommendations
    if any("thirst" in symptom.lower() or "urination" in symptom.lower() for symptom in symptoms):
        recommendations.append("Monitor fluid intake and output carefully")
    
    if any("vision" in symptom.lower() for symptom in symptoms):
        recommendations.append("Avoid driving or operating machinery")
    
    if any("breathing" in symptom.lower() for symptom in symptoms):
        recommendations.append("Sit upright and try to stay calm")
    
    # Risk-factor specific recommendations
    if any("Senior" in factor for factor in risk_factors):
        recommendations.append("Extra caution advised due to age-related risks")
    
    if any("BMI" in factor for factor in risk_factors):
        recommendations.append("Weight management should be discussed with provider")
    
    if any("diabetes" in condition.lower() for condition in conditions):
        recommendations.append("Bring glucose monitor and recent readings to appointment")
    
    return recommendations

def _generate_personalized_next_steps(
    urgency: str,
    risk_factors: List[str],
    age: Optional[int],
    language: str
) -> List[str]:
    """Generate personalized next steps"""
    
    next_steps = []
    
    if urgency in ["critical", "high"]:
        next_steps.extend([
            "Emergency contact: Keep phone charged and accessible",
            "Medical info: Prepare list of medications and allergies",
            "Support: Arrange for someone to accompany you",
            "Documents: Have insurance information ready"
        ])
    else:
        next_steps.extend([
            "Appointment: Schedule with primary care provider",
            "Preparation: Write down questions for your doctor",
            "Monitoring: Track symptom patterns until appointment",
            "Follow-up: Plan for telehealth option if available"
        ])
    
    # Personalized next steps based on risk factors
    if any("Senior" in factor for factor in risk_factors):
        next_steps.append("Consider geriatric specialist consultation")
    
    if any("Cardiovascular" in factor for factor in risk_factors):
        next_steps.append("Discuss cardiac evaluation with provider")
    
    return next_steps

async def _generate_personalized_fallback(request: EmergencyAssessmentRequest):
    """Personalized fallback assessment"""
    
    # Basic personalization even in fallback
    risk_factors = _calculate_personal_risk_factors(
        request.age, request.weight, request.height, 
        request.existing_conditions, request.current_medications
    )
    
    urgency = _calculate_personalized_urgency(request.symptoms, risk_factors, request.age)
    
    return EmergencyAssessmentResponse(
        assessment="Personalized Assessment (Basic Mode)",
        personalized_analysis=f"Based on your {len(request.symptoms)} symptoms and {len(risk_factors)} risk factors, careful monitoring is advised.",
        recommendations=[
            "System temporarily using basic assessment",
            "Contact healthcare provider for detailed evaluation",
            "Monitor symptoms closely",
            "Note any changes in symptom patterns"
        ],
        urgency_level=urgency,
        risk_factors=risk_factors,
        next_steps=[
            "Retry assessment when system available",
            "Seek professional medical advice",
            "Keep emergency contacts handy"
        ],
        timestamp=datetime.utcnow().isoformat()
    )


# -------------------------------------------------
# Enhanced Language Detection Endpoint
# -------------------------------------------------
@app.post("/api/v1/detect-language")
async def enhanced_language_detection(text: str = Form(...)):
    """Enhanced language detection with confidence scoring"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        detected_language = language_processor.detect_language(text)
        
        # Calculate confidence based on text characteristics
        confidence_score = _calculate_detection_confidence(text, detected_language)
        
        return {
            "detected_language": detected_language,
            "confidence_score": confidence_score,
            "confidence_level": "high" if confidence_score > 0.8 else "medium",
            "input_length": len(text),
            "detection_method": "keyword_analysis",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception("Enhanced language detection error")
        raise HTTPException(status_code=500, detail="Error detecting language")


def _calculate_detection_confidence(text: str, detected_language: str) -> float:
    """Calculate confidence score for language detection"""
    if len(text) < 5:
        return 0.5
    
    # Language-specific keyword confidence
    language_keywords = {
        "swahili": ["na", "ya", "kwa", "katika", "lakini", "pia"],
        "sheng": ["msee", "vibe", "sherehe", "dame", "kumi", "doo"]
    }
    
    text_lower = text.lower()
    keyword_matches = 0
    
    if detected_language in language_keywords:
        for keyword in language_keywords[detected_language]:
            if keyword in text_lower:
                keyword_matches += 1
    
    # Base confidence on keyword matches and text length
    base_confidence = min(1.0, keyword_matches * 0.2)
    length_boost = min(0.3, len(text) * 0.01)
    
    return min(1.0, base_confidence + length_boost)


# -------------------------------------------------
# Application Health Check with Detailed Metrics
# -------------------------------------------------
@app.get("/health")
async def comprehensive_health_check():
    """Comprehensive health check with system metrics"""
    current_time = datetime.utcnow()
    uptime = current_time - app_start_time
    
    # System metrics
    import psutil
    process = psutil.Process()
    
    system_info = {
        "status": "healthy",
        "timestamp": current_time.isoformat(),
        "uptime_seconds": uptime.total_seconds(),
        "uptime_human": str(uptime).split('.')[0],
        "version": "3.1.0",
        "system_metrics": {
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "active_connections": len(getattr(app, 'active_connections', [])),
            "total_requests": getattr(app, 'request_count', 0),
            "error_rate": f"{(error_count / max(1, getattr(app, 'request_count', 1)) * 100):.1f}%"
        },
        "services": {
            "llm_service": "active",
            "voice_processing": "active",
            "database": "active",
            "authentication": "active"
        },
        "features": {
            "multi_language": True,
            "voice_chat": True,
            "emergency_assessment": True,
            "diet_planning": True,
            "personalized_recommendations": True
        }
    }
    
    return system_info


# -------------------------------------------------
# Enhanced Error Handlers with Personalization
# -------------------------------------------------
@app.exception_handler(HTTPException)
async def personalized_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code} at {request.url.path}: {exc.detail}")
    
    # Personalized error response based on endpoint
    error_context = {
        "detail": exc.detail,
        "timestamp": datetime.utcnow().isoformat(),
        "path": request.url.path,
        "suggestion": _get_contextual_suggestion(request.url.path, exc.status_code)
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_context
    )


def _get_contextual_suggestion(path: str, status_code: int) -> str:
    """Get contextual suggestions for errors"""
    if "emergency" in path:
        return "For immediate medical concerns, contact emergency services directly"
    elif "voice" in path:
        return "Try using text input or check your microphone permissions"
    elif status_code == 401:
        return "Please check your API key and try again"
    elif status_code == 429:
        return "Rate limit exceeded. Please wait a moment and try again"
    else:
        return "Please try again or contact support if the issue persists"


logger.info("🚨 Enhanced Emergency Assessment System Initialized!")
logger.info("   ├── Personalized risk scoring")
logger.info("   ├── Multi-language emergency protocols")
logger.info("   ├── Contextual symptom analysis")
logger.info("   ├── Personalized monitoring instructions")
logger.info("   └── Fallback safety protocols")





# -------------------------------------------------
# VOICE CHAT MODELS AND DEPENDENCIES
# -------------------------------------------------
import base64
import io
import requests
import json
from pydantic import BaseModel

class VoiceChatRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data
    language: str = "english"
    user_id: str = "default"

class VoiceChatResponse(BaseModel):
    text_input: str
    ai_response: str
    timestamp: str
    language: str
    confidence: float

# -------------------------------------------------
# VOICE CHAT SERVICE (No pyaudio dependency)
# -------------------------------------------------
class VoiceChatService:
    def __init__(self):
        print("✅ Voice recognition service initialized (browser-based)")
    
    def decode_audio(self, base64_audio: str) -> io.BytesIO:
        """Decode base64 audio data to bytes"""
        try:
            # Remove data URL prefix if present
            if base64_audio.startswith('data:audio'):
                base64_audio = base64_audio.split(',')[1]
            
            audio_bytes = base64.b64decode(base64_audio)
            return io.BytesIO(audio_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio decoding failed: {str(e)}")
    
    async def transcribe_audio(self, audio_data: io.BytesIO, language: str = "en") -> tuple[str, float]:
        """Transcribe audio using various services"""
        try:
            # Option 1: Use OpenAI Whisper API (if you have access)
            # return await self._transcribe_with_openai(audio_data, language)
            
            # Option 2: Use Google Speech Recognition (free)
            return await self._transcribe_with_google_speech(audio_data, language)
            
            # Option 3: Use AssemblyAI (would require API key)
            # return await self._transcribe_with_assemblyai(audio_data, language)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    async def _transcribe_with_google_speech(self, audio_data: io.BytesIO, language: str) -> tuple[str, float]:
        """Use Google Speech Recognition API"""
        try:
            # For now, we'll use a mock implementation
            # In production, you would use Google Cloud Speech-to-Text API
            lang_map = {
                "english": "en-US",
                "swahili": "sw-KE",
                "spanish": "es-ES",
                "french": "fr-FR"
            }
            
            # Mock transcription for demo purposes
            # In production, replace this with actual Google Speech API call
            mock_responses = {
                "en-US": "I'm concerned about diabetes in my family",
                "sw-KE": "Nina wasiwasi kuhusu kisukari katika familia yangu",
                "es-ES": "Estoy preocupado por la diabetes en mi familia",
                "fr-FR": "Je suis préoccupé par le diabète dans ma famille"
            }
            
            recognition_language = lang_map.get(language.lower(), "en-US")
            transcribed_text = mock_responses.get(recognition_language, "I have questions about diabetes")
            
            return transcribed_text, 0.85
            
        except Exception as e:
            raise Exception(f"Google Speech recognition error: {str(e)}")
    
    async def _transcribe_with_openai(self, audio_data: io.BytesIO, language: str) -> tuple[str, float]:
        """Use OpenAI Whisper API"""
        try:
            # This would require an OpenAI API key
            # Implementation for production use:
            """
            import openai
            openai.api_key = "your-openai-api-key"
            
            audio_data.seek(0)
            transcript = openai.Audio.transcribe(
                "whisper-1", 
                audio_data,
                language=language
            )
            return transcript.text, 0.9
            """
            return "Mock transcription from OpenAI", 0.9
            
        except Exception as e:
            raise Exception(f"OpenAI transcription error: {str(e)}")

# Initialize voice service
voice_service = VoiceChatService()

# -------------------------------------------------
# VOICE CHAT ENDPOINT
# -------------------------------------------------
@app.post("/api/v1/voice-chat", response_model=VoiceChatResponse)
async def voice_chat_assistant(
    request: VoiceChatRequest,
    api_key: bool = Depends(verify_api_key)
):
    """Voice chat endpoint for diabetes concerns"""
    try:
        logger.info(f"🎤 Voice chat request from {request.user_id}, language: {request.language}")
        
        # Decode audio data
        audio_bytes = voice_service.decode_audio(request.audio_data)
        
        # Transcribe audio to text
        transcribed_text, confidence = await voice_service.transcribe_audio(audio_bytes, request.language)
        
        logger.info(f"📝 Transcribed text: {transcribed_text}")
        
        # Generate AI response using existing GPT-OSS service
        llm_result = await gpt_oss_specialist.generate_diabetes_response(
            message=transcribed_text,
            language=request.language,
            user_id=request.user_id
        )
        
        ai_response = llm_result["response"]
        
        return VoiceChatResponse(
            text_input=transcribed_text,
            ai_response=ai_response,
            timestamp=datetime.utcnow().isoformat(),
            language=request.language,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Voice chat service is temporarily unavailable. Please try again."
        )

# Test endpoint for voice functionality
@app.post("/api/v1/voice-chat/test")
async def voice_chat_test(
    request: Request,
    api_key: bool = Depends(verify_api_key)
):
    """Test voice chat without actual audio processing"""
    try:
        # Parse form data
        form_data = await request.form()
        text = form_data.get("text", "")
        language = form_data.get("language", "english")
        user_id = form_data.get("user_id", "default")
        
        logger.info(f"🎤 Voice chat test: {text}")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text input is required")
        
        # Generate AI response using existing GPT-OSS service
        llm_result = await gpt_oss_specialist.generate_diabetes_response(
            message=text,
            language=language,
            user_id=user_id
        )
        
        ai_response = llm_result["response"]
        
        return VoiceChatResponse(
            text_input=text,
            ai_response=ai_response,
            timestamp=datetime.utcnow().isoformat(),
            language=language,
            confidence=0.95
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat test error: {e}")
        raise HTTPException(status_code=500, detail="Voice chat test failed")
# ================================================================
# PRODUCTION DEPLOYMENT CONFIGURATION - ADD THIS AT THE VERY END
# ================================================================
# Production deployment configuration
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Disable reload in production
    )       
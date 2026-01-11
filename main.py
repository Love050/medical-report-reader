from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
from datetime import datetime
import os
import httpx
from dotenv import load_dotenv
import json
import json

# Load environment variables
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Google Gemini Direct API
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "models/gemini-2.5-flash")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Groq API (Fast inference with generous free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Hugging Face API (Good limits, less crowded)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# Multiple free models to rotate through if rate limited
FREE_MODELS = [
    os.getenv("MODEL_1", "google/gemini-2.0-flash-exp:free"),
    os.getenv("MODEL_2", "deepseek/deepseek-r1-0528:free"),
    os.getenv("MODEL_3", "meta-llama/llama-3.3-70b-instruct:free")
]

# Verify API configuration on startup
print("="*50)
print("Medical Report Reader - Backend Starting")
print(f"\n🔑 API Keys:")
print(f"  Google Gemini: {'✅ Yes' if GOOGLE_GEMINI_API_KEY else '❌ No'}")
if GOOGLE_GEMINI_API_KEY:
    print(f"    Model: {GOOGLE_GEMINI_MODEL}")
print(f"  Groq (Fast): {'✅ Yes' if GROQ_API_KEY else '❌ No'}")
if GROQ_API_KEY:
    print(f"    Model: {GROQ_MODEL}")
print(f"  Hugging Face: {'✅ Yes' if HUGGINGFACE_API_KEY else '❌ No'}")
if HUGGINGFACE_API_KEY:
    print(f"    Model: {HUGGINGFACE_MODEL}")
print(f"  OpenAI: {'✅ Yes' if OPENAI_API_KEY else '❌ No'}")
if OPENAI_API_KEY:
    print(f"    Model: {OPENAI_MODEL}")
print(f"  OpenRouter: {'✅ Yes' if OPENROUTER_API_KEY else '❌ No'}")
if OPENROUTER_API_KEY:
    print(f"    Models: {len(FREE_MODELS)} available")
print(f"\n📡 Priority: Gemini → Groq → Hugging Face → OpenAI → OpenRouter")
print("="*50)

# No OCR initialization needed - using lightweight API-based OCR for render.com
print("\n✅ Using OCR.space API (lightweight, no memory overhead)\n")

app = FastAPI(
    title="Medical Report Reader API",
    description="AI-powered medical report reader for health awareness",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ReportAnalysisRequest(BaseModel):
    report_text: str
    language: str = "en"

class TranslationRequest(BaseModel):
    text: str
    target_language: str

# Helper function to call OpenRouter API
async def call_openrouter_api(prompt: str, system_message: str = None, max_retries: int = 3) -> str:
    """
    Call OpenRouter API with automatic model rotation to avoid rate limits
    Tries all available free models until one succeeds
    """
    import asyncio
    
    # Try each model in the list
    for model_index, model in enumerate(FREE_MODELS):
        print(f"\nTrying model {model_index + 1}/{len(FREE_MODELS)}: {model}")
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Medical Report Reader"
                }
                
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
                
                print(f"  Attempt {attempt + 1}/{max_retries}...")
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
                    
                    # If rate limited, try next model immediately
                    if response.status_code == 429:
                        print(f"  ⚠️ Rate limited on {model}")
                        if model_index < len(FREE_MODELS) - 1:
                            print(f"  → Switching to next model...")
                            break  # Break attempt loop to try next model
                        elif attempt < max_retries - 1:
                            wait_time = 5
                            print(f"  → All models rate limited. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise HTTPException(
                                status_code=429,
                                detail="All AI models are currently busy. Please wait 60 seconds and try again."
                            )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        print(f"  ✅ Success with {model}!")
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception("No response from AI model")
                        
            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                print(f"  HTTP Error {e.response.status_code}: {error_body[:100]}")
                
                # If rate limited, try next model
                if e.response.status_code == 429:
                    print(f"  ⚠️ Rate limited on {model}")
                    if model_index < len(FREE_MODELS) - 1:
                        print(f"  → Switching to next model...")
                        break  # Try next model
                    elif attempt < max_retries - 1:
                        wait_time = 5
                        await asyncio.sleep(wait_time)
                        continue
                
                # For other HTTP errors, try next model
                if model_index < len(FREE_MODELS) - 1:
                    print(f"  → Trying next model...")
                    break
                    
                # If last model and last retry, raise error
                if model_index == len(FREE_MODELS) - 1 and attempt == max_retries - 1:
                    if e.response.status_code == 429:
                        raise HTTPException(
                            status_code=429,
                            detail="All AI models are temporarily busy. Please wait 60 seconds and try again."
                        )
                    elif e.response.status_code == 401:
                        raise HTTPException(status_code=500, detail="API authentication error.")
                    elif e.response.status_code == 402:
                        raise HTTPException(status_code=500, detail="API credits exhausted.")
                    else:
                        raise HTTPException(status_code=500, detail=f"AI service error. Please try again.")
                        
            except httpx.TimeoutException:
                print(f"  ⏱️ Timeout on {model}")
                if model_index < len(FREE_MODELS) - 1:
                    print(f"  → Trying next model...")
                    break
                elif attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    raise HTTPException(status_code=504, detail="Request timeout. Please try again.")
                    
            except Exception as e:
                print(f"  ❌ Error: {type(e).__name__} - {str(e)[:100]}")
                if model_index < len(FREE_MODELS) - 1:
                    print(f"  → Trying next model...")
                    break
                elif attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Service error. Please try again.")
    
    # If we get here, all models and retries failed
    raise HTTPException(
        status_code=503,
        detail="All AI services are currently unavailable. Please try again in a few minutes."
    )

# Helper function to call Google Gemini API directly
async def call_google_gemini_api(prompt: str, system_message: str = None) -> str:
    """
    Call Google Gemini API directly (not through OpenRouter)
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
        model = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)
        
        # Combine system message and prompt
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        
        print(f"Calling Google Gemini API directly with model: {GOOGLE_GEMINI_MODEL}")
        response = model.generate_content(full_prompt)
        
        print(f"✅ Success with Google Gemini!")
        return response.text
        
    except Exception as e:
        print(f"Google Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google Gemini error: {str(e)}")

# Helper function to call OpenAI API directly
async def call_openai_api(prompt: str, system_message: str = None) -> str:
    """
    Call OpenAI API directly
    """
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        print(f"Calling OpenAI API with model: {OPENAI_MODEL}")
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        print(f"✅ Success with OpenAI!")
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

# Helper function to call Groq API
async def call_groq_api(prompt: str, system_message: str = None) -> str:
    """
    Call Groq API - Fast inference with generous free tier
    """
    try:
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=GROQ_API_KEY)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        print(f"Calling Groq API with model: {GROQ_MODEL}")
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        print(f"✅ Success with Groq!")
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

# Helper function to call Hugging Face API
async def call_huggingface_api(prompt: str, system_message: str = None) -> str:
    """
    Call Hugging Face Inference API - Good limits, less crowded
    """
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Combine system message and prompt
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        api_url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
        
        print(f"Calling Hugging Face API with model: {HUGGINGFACE_MODEL}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                print(f"✅ Success with Hugging Face!")
                return result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                print(f"✅ Success with Hugging Face!")
                return result["generated_text"]
            else:
                raise Exception(f"Unexpected response format: {result}")
        
    except Exception as e:
        print(f"Hugging Face API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hugging Face error: {str(e)}")

# Root endpoint - serve frontend
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Note: OCR is handled client-side with Tesseract.js to save server memory on render.com free tier

# Combined analysis and recommendations endpoint
@app.post("/api/analyze-complete")
async def analyze_complete(request: ReportAnalysisRequest):
    """
    Combined endpoint: Analyze report AND get recommendations in ONE API call
    This reduces API usage by 50% to avoid rate limits
    """
    try:
        system_message = """You are a helpful medical information assistant. Your role is to:
1. Explain medical reports in simple, easy-to-understand language
2. Identify key health parameters and their meanings
3. Provide general health recommendations
4. NEVER diagnose, prescribe, or provide specific medical advice
5. Always remind users to consult healthcare professionals
6. Be accurate and avoid hallucination - only explain what's in the report
7. If something is unclear or you're unsure, say so

Remember: You provide educational information only, not medical advice."""

        prompt = f"""Analyze this medical report and provide BOTH analysis AND recommendations in ONE response.

**IMPORTANT: Respond ONLY in {request.language.upper()} language. Do NOT use English if another language is specified.**

REPORT TEXT:
{request.report_text}

You MUST provide a complete JSON response with ALL sections filled in {request.language} language:

PART 1 - ANALYSIS (10-15 lines):
1. Brief overview of what the report shows
2. For EACH parameter, clearly state if it is:
   - ✅ NORMAL (within range)
   - ⬇️ LOW (below normal range)
   - ⬆️ HIGH (above normal range)
3. Key findings with status indicators
4. Brief reminder to consult doctor

PART 2 - RECOMMENDATIONS (MANDATORY - 5 items each):
1. DIET ADVICE: Exactly 5 dietary tips based on the report
2. LIFESTYLE TIPS: Exactly 5 lifestyle changes
3. EXERCISE: Exactly 5 exercise suggestions
4. YOGA & MINDFULNESS: Exactly 5 yoga/meditation practices
5. THINGS TO AVOID: Exactly 5 things to avoid
6. BENEFITS: Exactly 5 benefits of following recommendations

**MANDATORY JSON FORMAT (copy this structure exactly):**
```json
{{
  "simplified_explanation": "Your analysis here with ✅ NORMAL, ⬇️ LOW, ⬆️ HIGH indicators for each parameter",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
  "diet_advice": {{
    "recommendations": ["Diet tip 1", "Diet tip 2", "Diet tip 3", "Diet tip 4", "Diet tip 5"]
  }},
  "lifestyle_advice": {{
    "recommendations": ["Lifestyle tip 1", "Lifestyle tip 2", "Lifestyle tip 3", "Lifestyle tip 4", "Lifestyle tip 5"]
  }},
  "exercise_tips": {{
    "recommendations": ["Exercise 1", "Exercise 2", "Exercise 3", "Exercise 4", "Exercise 5"]
  }},
  "yoga_tips": {{
    "recommendations": ["Yoga practice 1", "Yoga practice 2", "Yoga practice 3", "Yoga practice 4", "Yoga practice 5"]
  }},
  "things_to_avoid": {{
    "recommendations": ["Avoid 1", "Avoid 2", "Avoid 3", "Avoid 4", "Avoid 5"]
  }},
  "benefits": {{
    "recommendations": ["Benefit 1", "Benefit 2", "Benefit 3", "Benefit 4", "Benefit 5"]
  }}
}}
```

**CRITICAL RULES:**
1. Entire response in {request.language} language
2. ALL sections must have exactly 5 items
3. Return ONLY valid JSON, no extra text
4. Keep recommendations specific to the report findings"""

        try:
            # Priority 1: Try Google Gemini first (direct API with provided key)
            if GOOGLE_GEMINI_API_KEY:
                print("🔹 Priority 1: Using Google Gemini Direct API")
                response_text = await call_google_gemini_api(prompt, system_message)
            # Priority 2: Try Groq (fast inference, generous free tier)
            elif GROQ_API_KEY:
                print("🔹 Priority 2: Using Groq API (Fast)")
                response_text = await call_groq_api(prompt, system_message)
            # Priority 3: Try Hugging Face (good limits, less crowded)
            elif HUGGINGFACE_API_KEY:
                print("🔹 Priority 3: Using Hugging Face API")
                response_text = await call_huggingface_api(prompt, system_message)
            # Priority 4: Try OpenAI
            elif OPENAI_API_KEY:
                print("🔹 Priority 4: Using OpenAI API")
                response_text = await call_openai_api(prompt, system_message)
            else:
                # Priority 5: Fallback to OpenRouter with model rotation
                print("🔹 Priority 5: Using OpenRouter API (fallback)")
                response_text = await call_openrouter_api(prompt, system_message)
        except HTTPException as api_error:
            # If primary API fails, try fallback in order
            print(f"⚠️ Primary API failed: {api_error.detail}")
            try:
                # Try next available API as fallback
                if GROQ_API_KEY and "Gemini" in str(api_error.detail):
                    print("🔄 Trying Groq as fallback...")
                    response_text = await call_groq_api(prompt, system_message)
                elif HUGGINGFACE_API_KEY and ("Groq" in str(api_error.detail) or not GROQ_API_KEY):
                    print("🔄 Trying Hugging Face as fallback...")
                    response_text = await call_huggingface_api(prompt, system_message)
                elif OPENAI_API_KEY and ("Hugging Face" in str(api_error.detail) or not HUGGINGFACE_API_KEY):
                    print("🔄 Trying OpenAI as fallback...")
                    response_text = await call_openai_api(prompt, system_message)
                elif "OpenAI" in str(api_error.detail) or not OPENAI_API_KEY:
                    print("🔄 Trying OpenRouter as fallback...")
                    response_text = await call_openrouter_api(prompt, system_message)
                else:
                    raise api_error
            except HTTPException as fallback_error:
                # If all models are rate limited, return basic fallback response
                print(f"⚠️ All AI models busy, returning fallback response")
                return {
                    "success": True,
                    "analysis": {
                        "simplified_explanation": "⚠️ AI Analysis Temporarily Unavailable\n\nAll AI services are currently busy. Your report was received successfully.\n\n✅ What you can do:\n• Compare your values with reference ranges in the report\n• Note any values marked as HIGH or LOW\n• Consult your healthcare provider for interpretation\n\n💡 Try uploading again in 1-2 minutes for full AI analysis.",
                        "key_findings": ["AI temporarily unavailable - Please try again shortly", "Consult doctor for interpretation"]
                    },
                    "recommendations": {
                        "diet_advice": {"recommendations": ["Eat balanced diet with fruits & vegetables", "Drink 8-10 glasses of water daily", "Include whole grains", "Limit processed foods", "Stay hydrated"]},
                        "lifestyle_advice": {"recommendations": ["Regular sleep 7-8 hours", "Manage stress", "Stay active", "Maintain healthy weight", "Regular health checkups"]},
                        "exercise_tips": {"recommendations": ["30 min moderate exercise daily", "Walking or light jogging", "Consult doctor before new routine", "Start slow and build up", "Include stretching"]},
                        "yoga_tips": {"recommendations": ["Deep breathing exercises", "Basic stretching", "Meditation 10 min daily", "Gentle yoga poses", "Mindfulness practice"]},
                        "things_to_avoid": {"recommendations": ["Excessive stress", "Irregular sleep", "Unhealthy food", "Sedentary lifestyle", "Ignoring symptoms"]},
                        "benefits": {"recommendations": ["Better health", "More energy", "Improved wellbeing", "Disease prevention", "Better quality of life"]},
                        "lifestyle_advice": {"recommendations": ["Regular sleep 7-8 hours", "Manage stress", "Stay active"]},
                        "exercise_tips": {"recommendations": ["30 min moderate exercise daily", "Walking or light jogging", "Consult doctor before new routine"]},
                        "yoga_tips": {"recommendations": ["Deep breathing exercises", "Basic stretching", "Meditation 10 min daily"]},
                        "things_to_avoid": {"recommendations": ["Excessive stress", "Irregular sleep", "Unhealthy food"]},
                        "benefits": {"recommendations": ["Better health", "More energy", "Improved wellbeing"]}
                    },
                "language": request.language
            }
        
        # Parse JSON response
        try:
            print(f"🔍 Raw AI Response (first 500 chars): {response_text[:500]}")
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            print(f"🔍 Cleaned Response (first 500 chars): {response_text[:500]}")
            
            complete_data = json.loads(response_text)
            
            print(f"✅ JSON parsed successfully!")
            print(f"   - Has simplified_explanation: {'simplified_explanation' in complete_data}")
            print(f"   - Has diet_advice: {'diet_advice' in complete_data}")
            print(f"   - Has lifestyle_advice: {'lifestyle_advice' in complete_data}")
            print(f"   - Has exercise_tips: {'exercise_tips' in complete_data}")
            
            # Debug: print actual diet_advice structure
            if 'diet_advice' in complete_data:
                diet = complete_data.get('diet_advice')
                print(f"   - diet_advice type: {type(diet)}")
                print(f"   - diet_advice content: {diet}")
            
            result = {
                "success": True,
                "analysis": {
                    "simplified_explanation": complete_data.get("simplified_explanation", ""),
                    "key_findings": complete_data.get("key_findings", [])
                },
                "recommendations": {
                    "diet_advice": complete_data.get("diet_advice", {"recommendations": ["Balanced diet", "Stay hydrated", "Eat fruits", "Avoid junk food", "Regular meals"]}),
                    "lifestyle_advice": complete_data.get("lifestyle_advice", {"recommendations": ["Regular sleep", "Manage stress", "Stay active", "Social connections", "Regular checkups"]}),
                    "exercise_tips": complete_data.get("exercise_tips", {"recommendations": ["30 min daily exercise", "Walking", "Stretching", "Strength training", "Consult doctor"]}),
                    "yoga_tips": complete_data.get("yoga_tips", {"recommendations": ["Deep breathing", "Meditation", "Gentle stretches", "Mindfulness", "Relaxation"]}),
                    "things_to_avoid": complete_data.get("things_to_avoid", {"recommendations": ["Excessive stress", "Poor sleep", "Unhealthy food", "Sedentary lifestyle", "Ignoring health"]}),
                    "benefits": complete_data.get("benefits", {"recommendations": ["Better health", "More energy", "Disease prevention", "Improved mood", "Better quality of life"]})
                },
                "language": request.language
            }
            
            print(f"📤 Returning to frontend - recommendations structure:")
            print(f"   - diet_advice: {result['recommendations']['diet_advice']}")
            
            return result
        except Exception as parse_error:
            print(f"❌ JSON parsing error: {parse_error}")
            print(f"   Raw response: {response_text[:1000]}")
            # Return fallback response with explanation text
            return {
                "success": True,
                "analysis": {
                    "simplified_explanation": response_text if len(response_text) > 50 else "Analysis completed. Please see recommendations below.",
                    "key_findings": ["Please consult with your healthcare provider"]
                },
                "recommendations": {
                    "diet_advice": {"recommendations": ["Balanced diet with fruits and vegetables", "Stay well hydrated", "Include lean proteins", "Whole grains", "Limit processed foods"]},
                    "lifestyle_advice": {"recommendations": ["Regular 7-8 hours sleep", "Stress management", "Regular exercise", "Maintain healthy weight", "Regular health checkups"]},
                    "exercise_tips": {"recommendations": ["30 minutes daily exercise", "Walking or jogging", "Strength training twice weekly", "Stretching exercises", "Consult doctor before starting"]},
                    "yoga_tips": {"recommendations": ["Deep breathing exercises", "Basic yoga poses", "Meditation 10-15 min daily", "Mindfulness practice", "Gentle stretching"]},
                    "things_to_avoid": {"recommendations": ["Excessive stress", "Poor sleep habits", "Unhealthy processed foods", "Sedentary lifestyle", "Ignoring health warning signs"]},
                    "benefits": {"recommendations": ["Improved overall health", "Increased energy levels", "Better disease prevention", "Enhanced mental wellbeing", "Improved quality of life"]}
                },
                "language": request.language
            }
    except Exception as e:
        print(f"Complete analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analyze report endpoint
@app.post("/api/analyze-report")
async def analyze_report(request: ReportAnalysisRequest):
    """
    Analyze medical report and provide insights using AI
    """
    try:
        system_message = """You are a helpful medical information assistant. Your role is to:
1. Explain medical reports in simple, easy-to-understand language
2. Identify key health parameters and their meanings
3. Provide general health information only
4. NEVER diagnose, prescribe, or provide specific medical advice
5. Always remind users to consult healthcare professionals
6. Be accurate and avoid hallucination - only explain what's in the report
7. If something is unclear or you're unsure, say so

Remember: You provide educational information only, not medical advice."""

        prompt = f"""Analyze this medical report and provide a BRIEF, simple explanation. (10-15 lines maximum):

**IMPORTANT: Respond ONLY in {request.language.upper()} language. Do NOT use English if another language is specified.**

REPORT TEXT:
{request.report_text}

Provide (in {request.language} language):
1. Brief overview of what the report shows (2-3 lines)
2. For EACH parameter, clearly state if it is:
   - ✅ NORMAL (within range)
   - ⬇️ LOW (below normal range)  
   - ⬆️ HIGH (above normal range)
3. List key findings with their status (e.g., "Hemoglobin: 14.5 - NORMAL", "WBC: 3,500 - LOW")
4. Brief reminder to consult doctor (1 line)

Format as JSON:
- simplified_explanation: string (must clearly indicate which values are LOW, NORMAL, or HIGH using emojis/labels, 10-15 lines total in {request.language} language)
- key_findings: array of 3-5 strings with status indicators

**CRITICAL: Your entire response must be in {request.language} language, NOT English.**
IMPORTANT: Keep it SHORT and simple. ALWAYS indicate status (LOW/NORMAL/HIGH) for each parameter. Only explain what's in the report. No diagnosis."""

        response_text = await call_openrouter_api(prompt, system_message)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response if it's wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            analysis_data = json.loads(response_text)
        except:
            # If JSON parsing fails, create structured response
            analysis_data = {
                "simplified_explanation": response_text,
                "key_findings": ["Please consult with your healthcare provider for detailed interpretation"]
            }
        
        return {
            "success": True,
            "simplified_explanation": analysis_data.get("simplified_explanation", response_text),
            "key_findings": analysis_data.get("key_findings", []),
            "language": request.language
        }
    
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing report: {str(e)}")

# Get health recommendations
@app.post("/api/recommendations")
async def get_health_recommendations(request: ReportAnalysisRequest):
    """
    Generate diet, lifestyle, exercise, and yoga recommendations based on report
    """
    try:
        system_message = """You are a health and wellness advisor providing general health recommendations. Your guidelines:
1. Provide evidence-based, general health advice
2. Recommendations should be safe for most adults
3. Always include disclaimers about consulting healthcare providers
4. DO NOT provide specific medical treatment or diagnosis
5. Focus on general wellness and preventive health
6. Be practical and actionable
7. Avoid hallucination - give realistic, proven recommendations

Remember: These are general wellness tips, not personalized medical advice."""

        prompt = f"""Based on this medical report, provide concise health recommendations:

**IMPORTANT: Respond ONLY in {request.language.upper()} language. Do NOT use English if another language is specified.**

REPORT TEXT:
{request.report_text}

Provide SHORT, focused recommendations (3-5 items per category) in {request.language} language:

1. DIET ADVICE: 3-5 key dietary tips
2. LIFESTYLE TIPS: 3-5 lifestyle changes
3. EXERCISE: 3-5 exercise suggestions
4. YOGA & MINDFULNESS: 3-5 yoga/meditation practices
5. THINGS TO AVOID: 3-5 things to avoid
6. BENEFITS: 3-5 main benefits

Format as JSON with these exact keys:
- diet_advice: array of strings
- lifestyle_advice: array of strings
- exercise_tips: array of strings
- yoga_tips: array of strings
- things_to_avoid: array of strings
- benefits: array of strings

IMPORTANT: 
- Keep recommendations general and safe for most people
- Include a note that these should be adapted based on individual health conditions
- Be specific but not overly medical
- Focus on practical, achievable suggestions"""

        response_text = await call_openrouter_api(prompt, system_message)
        
        # Parse JSON response
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            recommendations_data = json.loads(response_text)
        except:
            # Fallback recommendations
            recommendations_data = {
                "diet_advice": [
                    "Eat a balanced diet with plenty of fruits and vegetables",
                    "Stay hydrated - drink 8-10 glasses of water daily",
                    "Include whole grains in your meals",
                    "Limit processed and sugary foods",
                    "Add lean proteins to your diet"
                ],
                "lifestyle_advice": [
                    "Maintain regular sleep schedule (7-8 hours)",
                    "Practice stress management",
                    "Avoid smoking and limit alcohol",
                    "Spend time outdoors",
                    "Maintain good hygiene"
                ],
                "exercise_tips": [
                    "30 minutes of moderate exercise daily",
                    "Include cardio activities",
                    "Strength training 2-3 times per week",
                    "Regular stretching",
                    "Start slowly and increase gradually"
                ],
                "yoga_tips": [
                    "Practice deep breathing (Pranayama)",
                    "Meditation for 10-15 minutes daily",
                    "Gentle yoga for flexibility",
                    "Sun salutation in the morning",
                    "Relaxation techniques"
                ],
                "things_to_avoid": [
                    "Excessive caffeine",
                    "Late night screen time",
                    "Chronic stress",
                    "Skipping meals",
                    "Ignoring health symptoms"
                ],
                "benefits": [
                    "Improved overall health",
                    "Better energy levels",
                    "Stronger immune system",
                    "Reduced health risks",
                    "Better mental wellbeing"
                ]
            }
        
        return {
            "success": True,
            "diet_advice": {
                "title": "Diet Recommendations",
                "recommendations": recommendations_data.get("diet_advice", [])
            },
            "lifestyle_advice": {
                "title": "Lifestyle Tips",
                "recommendations": recommendations_data.get("lifestyle_advice", [])
            },
            "exercise_tips": {
                "title": "Exercise Recommendations",
                "recommendations": recommendations_data.get("exercise_tips", [])
            },
            "yoga_tips": {
                "title": "Yoga & Mindfulness",
                "recommendations": recommendations_data.get("yoga_tips", [])
            },
            "things_to_avoid": {
                "title": "Things to Avoid",
                "recommendations": recommendations_data.get("things_to_avoid", [])
            },
            "benefits": {
                "title": "Benefits of Following These Recommendations",
                "recommendations": recommendations_data.get("benefits", [])
            }
        }
    
    except Exception as e:
        print(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Translate text endpoint
@app.post("/api/translate")
async def translate_text(request: TranslationRequest):
    """
    Translate text to target language using AI
    """
    try:
        system_message = """You are a professional translator. Translate the given text accurately while:
1. Maintaining the original meaning and context
2. Using appropriate medical terminology when applicable
3. Keeping the tone and formality of the original
4. Being culturally appropriate for the target language"""

        # Language name mapping
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ar": "Arabic",
            "bn": "Bengali",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi",
            "ur": "Urdu"
        }
        
        target_lang_name = language_names.get(request.target_language, request.target_language)
        
        prompt = f"""Translate the following text to {target_lang_name}:

{request.text}

Provide ONLY the translation, no explanations or additional text."""

        translated_text = await call_openrouter_api(prompt, system_message)
        
        return {
            "success": True,
            "original_text": request.text,
            "translated_text": translated_text.strip(),
            "target_language": request.target_language,
            "message": "Translation completed successfully"
        }
    
    except Exception as e:
        print(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}")

# Get supported languages
@app.get("/api/languages")
async def get_supported_languages():
    """
    Return list of supported languages
    """
    languages = [
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "Hindi (हिंदी)"},
        {"code": "es", "name": "Spanish (Español)"},
        {"code": "fr", "name": "French (Français)"},
        {"code": "de", "name": "German (Deutsch)"},
        {"code": "zh", "name": "Chinese (中文)"},
        {"code": "ar", "name": "Arabic (العربية)"},
        {"code": "bn", "name": "Bengali (বাংলা)"},
        {"code": "pt", "name": "Portuguese (Português)"},
        {"code": "ru", "name": "Russian (Русский)"},
        {"code": "ja", "name": "Japanese (日本語)"},
        {"code": "ta", "name": "Tamil (தமிழ்)"},
        {"code": "te", "name": "Telugu (తెలుగు)"},
        {"code": "mr", "name": "Marathi (मराठी)"},
        {"code": "ur", "name": "Urdu (اردو)"}
    ]
    
    return {"success": True, "languages": languages}

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

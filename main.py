from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from PIL import Image
import io
import base64


app = FastAPI(title="AgriShield Backend")
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os

# Add this block to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for hackathon dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- MongoDB Integration ---
# Provide default connection string if not found in env
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://g486822_db_user:Naseer@cluster0.x8v75pd.mongodb.net/")

@app.on_event("startup")
async def startup_db_client():
    print(f"Connecting to MongoDB...")
    app.mongodb_client = AsyncIOMotorClient(MONGO_URI)
    app.database = app.mongodb_client.get_database("agrishield_db")
    print("Connected to MongoDB!")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    print("Closed MongoDB connection.")

# --- Onboarding Endpoint ---
from typing import Optional

class UserRegistration(BaseModel):
    language: str
    name: str
    mobile_number: str
    password: Optional[str] = None
    farm_location: str
    farm_size_acres: float
    crop: str

@app.get("/users/check")
async def check_user_exists(mobile: str):
    if hasattr(app, "database"):
        users_collection = app.database.get_collection("users")
        existing_user = await users_collection.find_one({"mobile_number": mobile})
        if existing_user:
            return {"exists": True}
    return {"exists": False}

class UserLogin(BaseModel):
    mobile_number: str
    password: str

@app.post("/users/login")
async def login_user(req: UserLogin):
    try:
        if hasattr(app, "database"):
            users_collection = app.database.get_collection("users")
            user = await users_collection.find_one({"mobile_number": req.mobile_number})
            if not user:
                return JSONResponse(status_code=404, content={"status": "error", "message": "Account not found."})
            
            if user.get("password") != req.password:
                return JSONResponse(status_code=401, content={"status": "error", "message": "Incorrect password."})
            
            # Remove MongoDB _id before returning
            user.pop("_id", None)
            return {"status": "success", "message": "Login successful", "user": user}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "DB not connected"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/users/onboard")
async def register_user(req: UserRegistration):
    try:
        if hasattr(app, "database"):
            users_collection = app.database.get_collection("users")
            
            # Check if user already exists
            existing_user = await users_collection.find_one({"mobile_number": req.mobile_number})
            if existing_user:
                return JSONResponse(status_code=400, content={"status": "error", "message": "User with this mobile number already exists."})

            user_doc = req.dict()
            user_doc["created_at"] = datetime.now().isoformat()
            
            # Simple password storing for hackathon prototype (In real app, MUST hash password)
            await users_collection.insert_one(user_doc)
            return {"status": "success", "message": "User registered successfully"}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "DB not connected"})
    except Exception as e:
        print(f"Error saving user to MongoDB: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/users/{mobile_number}")
async def get_user(mobile_number: str):
    try:
        if hasattr(app, "database"):
            users_collection = app.database.get_collection("users")
            user = await users_collection.find_one({"mobile_number": mobile_number})
            if not user:
                return JSONResponse(status_code=404, content={"status": "error", "message": "User not found."})
            
            user.pop("_id", None)
            return {"status": "success", "user": user}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "DB not connected"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class UserUpdate(BaseModel):
    mobile_number: str
    name: Optional[str] = None
    language: Optional[str] = None
    about: Optional[str] = None
    email: Optional[str] = None

@app.put("/users/update")
async def update_user(req: UserUpdate):
    try:
        if hasattr(app, "database"):
            users_collection = app.database.get_collection("users")
            user = await users_collection.find_one({"mobile_number": req.mobile_number})
            if not user:
                return JSONResponse(status_code=404, content={"status": "error", "message": "User not found."})
            
            update_data = {k: v for k, v in req.dict().items() if v is not None and k != "mobile_number"}
            
            if update_data:
                await users_collection.update_one(
                    {"mobile_number": req.mobile_number},
                    {"$set": update_data}
                )
            
            return {"status": "success", "message": "Profile updated successfully"}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "DB not connected"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AgriShield</title>
        </head>
        <body style="font-family: Arial; text-align: center; padding: 80px;">
            <h1>ðŸŒ± AgriShield</h1>
            <h2>Plant Disease Diagnosis + MRL Safety Check</h2>
            <p>âœ… Backend is running successfully</p>
            <p>MobileNetV2 + Grad-CAM &nbsp;|&nbsp; MRL/PHI Assessment Engine</p>
            <br>
            <a href="/docs">Open API Testing</a>
        </body>
    </html>
    """

# Define the expected JSON payload shape
class MRLRequest(BaseModel):
    crop: str
    pesticide: str
    initial_residue: float = 2.0
    spray_date: str
    destination: Optional[str] = "Domestic"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    mobile_number: Optional[str] = None

import google.generativeai as genai
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel('gemini-1.5-flash')

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...), mobile_number: str = Form(None)):
    contents = await file.read()

    try:
        # Use Gemini Vision for instant, accurate diagnosis - no heavy ML model needed
        image_b64 = base64.b64encode(contents).decode("utf-8")
        content_type = file.content_type or "image/jpeg"

        diagnosis_prompt = """
        You are an expert agricultural plant disease AI for the AgriShield app.
        Analyze this crop leaf image and return ONLY a raw JSON object (no markdown, no code blocks).

        If it IS a plant/leaf image, return:
        {
          "decision": "confident",
          "primary_disease": "<disease name or Healthy>",
          "crop": "<detected crop type, e.g. Tomato, Rice, Wheat, Unknown>",
          "confidence": <0.0-1.0>,
          "severity": "<None/Mild/Moderate/Severe>",
          "affected_area_pct": <0-100>,
          "description": "<2-3 sentence description of the disease or health status>",
          "treatment": "<specific actionable treatment recommendation>",
          "prevention": "<prevention tip>",
          "is_healthy": <true/false>
        }

        If the image is NOT a plant/leaf (e.g. a selfie, random object), return:
        {"decision": "reject_not_leaf"}

        If the image is too blurry or dark to analyze:
        {"decision": "reject_quality"}
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            diagnosis_prompt,
            {"mime_type": content_type, "data": image_b64}
        ])

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        response_data = json.loads(response_text)

        # Save to MongoDB if it's a real diagnosis
        if response_data.get("decision") not in ["reject_not_leaf", "reject_quality", "reject_unknown"]:
            if hasattr(app, "database"):
                disease_collection = app.database.get_collection("disease_reports")
                db_record = response_data.copy()
                db_record["created_at"] = datetime.now().isoformat()
                if mobile_number:
                    db_record["mobile_number"] = mobile_number
                await disease_collection.insert_one(db_record)

        return JSONResponse(response_data)
    except Exception as e:
        print(f"Error in Gemini diagnosis: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "pipeline_error", "message": str(e)}
        )

import base64
import google.generativeai as genai
import os
import json

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

@app.post("/voice-command")
async def process_voice_command(file: UploadFile = File(...), currentScreen: str = Form(default="Unknown")):
    try:
        contents = await file.read()
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are the voice assistant for the AgriShield farming app. You are a helpful, empathetic, and highly intelligent human-like friend.
        The user will speak in any language (Hindi, Marathi, English, Kannada, etc). You must reply in the language they used, using casual, friendly conversational tones (e.g. "Bhai", "Dost").

        CRITICAL CONTEXT:
        The user is CURRENTLY on the app screen: "{currentScreen}"

        Your job is to understand their intent. They might want to:
        1. NAVIGATE to another page.
        2. ASK a question about what is on their current screen or what they should do next.
        3. Just chat.

        Available screens they can navigate to:
        - "Onboarding" (Welcome/Onboarding page)
        - "Login" (Login page)
        - "Home" (Main dashboard)
        - "CameraScan" (Take a photo of a leaf to detect disease, or "open scanner")
        - "MRLSprayHistory" (Check pesticide safe spray history)
        - "ReportsHistory" (View past diagnosis reports)
        - "Profile" (User profile)
        
        IF THEY ASK ABOUT THE CURRENT SCREEN:
        Explain what they can do on this screen.
        Example: If currentScreen is "Login" and they ask "isme kya kya hai?" -> Reply: "Yahan aapko apna mobile number aur password daalna hoga login karne ke liye bhai."
        Example: If currentScreen is "Home" -> Reply: "Bhai yeh main dashboard hai, yahan se aap crop scan kar sakte ho ya apni reports dekh sakte ho."

        Return ONLY a JSON object in this exact format. Do NOT include markdown blocks, just the raw JSON:

        If they want to NAVIGATE:
        {{"action": "navigate", "screen": "ScreenName", "reply": "A friendly confirmation that you are opening it."}}

        If they want to CONVERSE or ask about the current screen:
        {{"action": "converse", "reply": "Your intelligent, context-aware answer explaining the screen or answering their query."}}

        If you don't understand:
        {{"action": "unknown", "reply": "Bhai thoda clear bologe? Samajh nahi aaya."}}
        """
        
        response = model.generate_content([
            prompt,
            {
                "mime_type": file.content_type or "audio/m4a",
                "data": contents
            }
        ])
        
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error processing voice: {e}")
        return JSONResponse(status_code=500, content={"action": "error", "message": str(e)})

from drone.drone_analyze import analyze_drone_image

@app.post("/drone-analyze")
async def drone_analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = analyze_drone_image(image)
    return result

from mrl.mrl_assessment import assess_crop_safety
import math
import requests

@app.post("/mrl-risk")
async def check_mrl_risk(req: MRLRequest):
    # 1. Parse dates to YYYY-MM-DD
    try:
        spray_date = datetime.strptime(req.spray_date, "%d %b %Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            spray_date = datetime.strptime(req.spray_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            spray_date = datetime.now().strftime("%Y-%m-%d") # Fallback

    d_spray = datetime.strptime(spray_date, "%Y-%m-%d")
    days_elapsed = (datetime.now() - d_spray).days
    if days_elapsed < 0:
        return {"status": "ERROR", "explanation": "Spray date cannot be in the future."}

    # 2. Assume a default initial residue since farmer shouldn't enter this
    default_initial_residue = req.initial_residue
    
    # Check if unknown pesticide
    if req.pesticide == "Unknown" or not req.pesticide:
        return {
            "status": "HOLD",
            "risk_level": "unknown",
            "estimated_residue_risk": 0.0,
            "mrl_limit": 0.0,
            "safe_harvest_date": "Unknown",
            "phi_remaining_days": 0,
            "confidence": 0.0,
            "weather_adjustment": "No pesticide provided.",
            "explanation": "AgriShield could not verify the required pesticide/MRL information."
        }

    # 3. Call Pavan's real logic to get base assessment
    assessment = assess_crop_safety(
        crop=req.crop,
        pesticide=req.pesticide,
        initial_residue=default_initial_residue,
        spray_date=spray_date
    )

    if assessment.get("status") in ["UNKNOWN", "ERROR"]:
        return {
            "status": "HOLD",
            "risk_level": "unknown",
            "estimated_residue_risk": 0.0,
            "mrl_limit": 0.0,
            "safe_harvest_date": "Unknown",
            "phi_remaining_days": 0,
            "confidence": 0.0,
            "weather_adjustment": "Error in assessment.",
            "explanation": assessment.get("message", "AgriShield could not verify the required pesticide/MRL information.")
        }

    # Weather Integration: Rain Wash-off calculation
    weather_modifier = 1.0
    weather_note = "No location provided; using standard decay."

    if req.latitude and req.longitude:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={req.latitude}&longitude={req.longitude}&current=precipitation&timezone=auto"
            resp = requests.get(url, timeout=3).json()
            rain_mm = resp.get("current", {}).get("precipitation", 0)

            if rain_mm > 0:
                # Hackathon logic: 15% residue wash-off per mm of rain (capped at 50% reduction)
                wash_off = min(0.50, rain_mm * 0.15)
                weather_modifier = 1.0 - wash_off
                weather_note = f"Rainfall detected ({rain_mm}mm). Applied {int(wash_off * 100)}% wash-off reduction."
            else:
                weather_note = "Clear weather detected at location. Standard decay applied."
        except Exception:
            weather_note = "Weather API timeout; using standard decay."

    # Adjust predicted residue based on weather
    adjusted_residue = assessment["predicted_residue_mg_per_kg"] * weather_modifier
    mrl = assessment["mrl_mg_per_kg"]
    
    # 4. Map statuses to UI
    if adjusted_residue <= mrl:
        ui_status = "SAFE"
        risk_level = "low"
        phi_remaining = 0
        explanation = "The recommended waiting period has been satisfied and no known rule conflict is detected."
    else:
        ui_status = "WAIT"
        risk_level = "high" if adjusted_residue > mrl * 2 else "moderate"
        explanation = "Your spray was applied recently and the estimated residue is above the safe limit."
        
        # Calculate PHI remaining
        from mrl.mrl_lookup import find_mrl
        mrl_data = find_mrl(req.crop, req.pesticide)
        dt50 = mrl_data["half_life_days"]
        k = math.log(2) / dt50
        safe_days = math.log(adjusted_residue / mrl) / k
        phi_remaining = math.ceil(safe_days)

    current_date = datetime.now()
    safe_harvest_date = (current_date + timedelta(days=phi_remaining)).strftime("%d %b %Y")

    response_data = {
        "status": ui_status,
        "risk_level": risk_level,
        "estimated_residue_risk": round((adjusted_residue / mrl), 2) if mrl > 0 else 0.0,
        "mrl_limit": mrl,
        "safe_harvest_date": safe_harvest_date,
        "phi_remaining_days": phi_remaining,
        "confidence": 0.85,
        "weather_adjustment": weather_note,
        "explanation": explanation,
        "crop": req.crop,
        "pesticide": req.pesticide,
        "spray_date": spray_date,
        "created_at": current_date.isoformat(),
        "mobile_number": req.mobile_number
    }
    
    # Save to MongoDB
    try:
        if hasattr(app, "database"):
            mrl_collection = app.database.get_collection("mrl_reports")
            await mrl_collection.insert_one(response_data.copy())
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

    return response_data

@app.get("/reports/{mobile_number}")
async def get_user_reports(mobile_number: str):
    try:
        if hasattr(app, "database"):
            disease_col = app.database.get_collection("disease_reports")
            mrl_col = app.database.get_collection("mrl_reports")
            
            # Fetch disease reports
            disease_cursor = disease_col.find({"mobile_number": mobile_number})
            disease_reports = await disease_cursor.to_list(length=100)
            
            for rep in disease_reports:
                rep.pop("_id", None)
                rep["type"] = "disease"
                
            # Fetch MRL reports
            mrl_cursor = mrl_col.find({"mobile_number": mobile_number})
            mrl_reports = await mrl_cursor.to_list(length=100)
            
            for rep in mrl_reports:
                rep.pop("_id", None)
                rep["type"] = "mrl"
                
            all_reports = disease_reports + mrl_reports
            # Sort by created_at descending
            all_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return {"status": "success", "reports": all_reports}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "DB not connected"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class MRLCheckRequest(BaseModel):
    crop: str
    pesticide: str
    predicted_residue: float

if __name__ == "__main__":
    import uvicorn
    # This tells PyCharm to actually start the web server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# @app.post("/mrl-check")
# def mrl_check(request: MRLCheckRequest):
#     result = assess_crop_safety(
#         crop=request.crop,
#         pesticide=request.pesticide,
#         predicted_residue=request.predicted_residue
#     )
#     return JSONResponse(result)

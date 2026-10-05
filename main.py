import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Self-Ping Background Task Definition ---
RENDER_APP_URL = os.getenv(
    "RENDER_EXTERNAL_URL", 
    "https://beauty-agent-backends.onrender.com/"
)

async def keep_alive_ping():
    """Background loop that pings the app every 10 minutes to prevent Render auto-sleep."""
    # Wait 10 seconds after server boot before starting the loop
    await asyncio.sleep(10)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                logging.info(f"Sending keep-alive self-ping to {RENDER_APP_URL}")
                response = await client.get(RENDER_APP_URL, timeout=10.0)
                logging.info(f"Self-ping response status: {response.status_code}")
            except Exception as e:
                logging.warning(f"Self-ping keep-alive failed: {str(e)}")
            
            # Wait 10 minutes (600 seconds) between pings
            await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the keep-alive task in the background
    ping_task = asyncio.create_task(keep_alive_ping())
    yield
    # Shutdown: Cancel the task cleanly when the server stops
    ping_task.cancel()

# --- 2. Initialize Async OpenAI Client for Groq ---
GROQ_KEY = os.getenv(
    "GROQ_API_KEY", 
    "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"
)

client = AsyncOpenAI(
    api_key=GROQ_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Active Groq Models (Deprecated models removed)
MODEL_PRIORITY_LIST = [
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b"
]

# --- 3. FastAPI Setup ---
app = FastAPI(title="Global Beauty Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Global Beauty Assistant API is live!"}

# Data Schemas
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None

# --- 4. Streaming Chat Endpoint ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_generator():
        system_instruction = {
            "role": "system",
            "content": (
                "You are an expert global beauty, skincare, and wellness assistant. "
                "Provide clear, practical, and helpful advice on beauty trends, "
                "skincare routines, and ingredient insights."
            )
        }

        conversation_history = [system_instruction]

        if request.messages:
            for msg in request.messages:
                conversation_history.append({"role": msg.role, "content": msg.content})
        elif request.prompt:
            conversation_history.append({"role": "user", "content": request.prompt})
        else:
            error_data = json.dumps({"output": "Error: No prompt or messages provided."})
            yield f"data: {error_data}\n\n"
            return

        stream = None
        last_error = None

        for model in MODEL_PRIORITY_LIST:
            try:
                logging.info(f"Attempting stream generation with model: {model}")
                stream = await client.chat.completions.create(
                    model=model,
                    messages=conversation_history,
                    stream=True
                )
                break
            except Exception as e:
                logging.warning(f"Model '{model}' failed: {str(e)}")
                last_error = e
                continue

        if not stream:
            error_data = json.dumps({"output": f"Error: All models failed. Last error: {str(last_error)}"})
            yield f"data: {error_data}\n\n"
            return

        try:
            full_response = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response += content
                    response_data = json.dumps({"output": full_response})
                    yield f"data: {response_data}\n\n"
        except Exception as e:
            error_data = json.dumps({"output": f"Error during streaming: {str(e)}"})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Reads Render's dynamic PORT environment variable (defaults to 8000 locally)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
import os
import re
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.agent import CVAgent
from app.retrieval import index_cv
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Limitador de tasa de peticiones por IP
limiter = Limiter(key_func=get_remote_address)

MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_LENGTH = 2000

# Deteccion de patrones comunes de prompt injection para trazabilidad
SUSPICIOUS_PATTERNS = re.compile(
    r"(ignore (previous|all) instructions|olvida (tus |las )?instrucciones"
    r"|system prompt|actúa como|act as|you are now|eres ahora)",
    re.IGNORECASE,
)


# Inicializacion de la aplicacion e indexacion de documentos al arrancar
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Indexando CV en Qdrant...")
    n = index_cv("data/cv.md")
    logger.info(f"CV indexado correctamente: {n} chunks.")
    yield


app = FastAPI(title="CV Agent - Open Responses API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")


# Schemas para la especificacion Open Responses
class InputMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ResponsesRequest(BaseModel):
    model: str = "cv-agent"
    input: str | list[InputMessage]


class OutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class OutputItem(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[OutputTextContent]


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    model: str
    created_at: int
    output: list[OutputItem]


# Autenticacion basada en Bearer token
def verify_api_key(authorization: str | None) -> None:
    if not SERVICE_API_KEY:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Falta el header Authorization: Bearer <key>")

    token = authorization.removeprefix("Bearer ").strip()
    if token != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")


# Endpoint de respuestas de la API
@app.post("/v1/responses", response_model=ResponsesResponse)
@limiter.limit("20/minute")
def create_response(
    request: Request,
    body: ResponsesRequest,
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)

    agent = CVAgent()

    if isinstance(body.input, str):
        user_message = body.input
    else:
        if not body.input:
            raise HTTPException(status_code=400, detail="input no puede ser una lista vacia")

        trimmed_input = body.input[-MAX_HISTORY_MESSAGES:]

        for msg in trimmed_input[:-1]:
            role = "model" if msg.role == "assistant" else "user"
            agent.history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

        if trimmed_input[-1].role != "user":
            raise HTTPException(status_code=400, detail="El ultimo mensaje de 'input' debe ser del usuario")
        user_message = trimmed_input[-1].content

    if len(user_message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Mensaje demasiado largo (máx {MAX_MESSAGE_LENGTH} caracteres)")

    if SUSPICIOUS_PATTERNS.search(user_message):
        logger.warning(f"[posible_injection] mensaje sospechoso recibido: {user_message[:200]!r}")

    try:
        answer_text = agent.chat(user_message)
    except Exception:
        logger.exception("Error generando respuesta del agente")
        raise HTTPException(status_code=500, detail="Error interno generando la respuesta")

    return ResponsesResponse(
        id=f"resp_{uuid.uuid4().hex}",
        model=body.model,
        created_at=int(time.time()),
        output=[
            OutputItem(content=[OutputTextContent(text=answer_text)]),
        ],
    )


# Health check del servicio
@app.get("/health")
def health():
    return {"status": "ok"}
import os
from google import genai
from google.genai.types import EmbedContentConfig

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    # Genera embeddings para una lista de documentos a indexar
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )
    return [e.values for e in response.embeddings]


def embed_query(text: str) -> list[float]:
    # Genera el embedding para una consulta de usuario
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )
    return response.embeddings[0].values
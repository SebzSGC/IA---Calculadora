"""
Configuración centralizada del proyecto IA Calculadora IO.
Carga variables de entorno y define constantes.
"""
import os
from dotenv import load_dotenv
from logging_config import get_logger

log = get_logger("config")

# Cargar variables de entorno
load_dotenv()

# API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    log.critical("No se encontró GOOGLE_API_KEY en el archivo .env")
    raise ValueError("❌ No se encontró GOOGLE_API_KEY en el archivo .env")

log.info(f"API Key cargada (termina en ...{GOOGLE_API_KEY[-4:]})")

# Rutas
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "Insumos",
                        "Investigacion-Operaciones10Edicion-Frederick-S-Hillier.pdf")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

log.debug(f"PDF_PATH: {PDF_PATH}")
log.debug(f"CHROMA_DB_DIR: {CHROMA_DB_DIR}")

# Modelos
EMBEDDING_MODEL = "models/text-embedding-004" # Modelo Gemini con 2048+ tokens de contexto
CHAT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]  # Optimizado para razonamiento matemático IO

log.info(f"Modelos chat: {CHAT_MODELS}")
log.info(f"Modelo embeddings: {EMBEDDING_MODEL}")

# Rate limiting (pay-as-you-go — límites muy holgados)
BATCH_SIZE = 50        # Lote más pequeño para la API de Gemini Embeddings
WAIT_SECONDS = 3       # Pausa mínima entre lotes (cortesía)
MAX_CHUNKS = None      # None = procesar TODOS los fragmentos del PDF

# Chunking Inteligente (Math-Aware)
CHUNK_SIZE = 3000      # Suficiente para que quepa un problema con sus tablas Símplex
CHUNK_OVERLAP = 500    # Solapamiento generoso para no perder continuidad en algoritmos

log.info("Configuración cargada correctamente")

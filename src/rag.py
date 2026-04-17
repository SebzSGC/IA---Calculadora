"""
Módulo RAG (Retrieval Augmented Generation).
Consulta a Gemini con contexto del libro de IO.
Optimizado para pay-as-you-go (reintentos rápidos).
"""
import os
import json
import time
import hashlib
from langchain_google_genai import ChatGoogleGenerativeAI
from embeddings import search_context
from config import CHAT_MODELS
from logging_config import get_logger

log = get_logger("rag")

# Caché de respuestas en disco
RESPONSE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "response_cache.json")


def _load_response_cache():
    """Carga el caché de respuestas desde disco."""
    if os.path.exists(RESPONSE_CACHE_FILE):
        with open(RESPONSE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_response_cache(cache):
    """Guarda el caché de respuestas en disco."""
    with open(RESPONSE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _get_cache_key(pregunta):
    """Genera una clave de caché basada en la pregunta."""
    return hashlib.md5(pregunta.strip().lower().encode()).hexdigest()


def preguntar_io(pregunta_usuario, vectorstore, use_cache=True):
    """
    Pregunta a Gemini con contexto del libro de IO.

    Features:
    - Modelo adaptativo: intenta modelos en orden de CHAT_MODELS
    - Caché de respuestas: no repite consultas idénticas
    - Reintentos con backoff ante rate limit (429)
    - Logging completo de cada paso

    Args:
        pregunta_usuario: La pregunta del usuario
        vectorstore: ChromaDB vector store
        use_cache: Si True, busca/guarda en caché

    Returns:
        str: Respuesta de Gemini
    """
    log.info("=" * 60)
    log.info(f"NUEVA CONSULTA: {pregunta_usuario[:100]}...")
    log.info("=" * 60)

    # 1. Verificar caché
    if use_cache:
        cache = _load_response_cache()
        cache_key = _get_cache_key(pregunta_usuario)
        if cache_key in cache:
            log.info(f"Cache HIT — clave: {cache_key}")
            print("⚡ Respuesta cargada desde caché (0 API calls)")
            return cache[cache_key]
        log.debug(f"Cache MISS — clave: {cache_key}")

    # 2. Buscar contexto relevante
    log.info("Buscando contexto relevante en ChromaDB...")
    print("🔍 Buscando contexto relevante...")
    contexto = search_context(pregunta_usuario, vectorstore)
    log.debug(f"Contexto obtenido: {len(contexto)} caracteres")

    prompt = f"""
Eres un experto en Investigación de Operaciones. Analiza el siguiente problema.

CONTEXTO (teoría extraída del PDF):
{contexto}

PROBLEMA:
{pregunta_usuario}

INSTRUCCIONES:
1. Clasifica el tipo de problema (LP, Colas, Redes, PERT)
2. Presenta la formulación matemática completa
3. Explica paso a paso la resolución
4. Da los resultados numéricos finales

Si es un problema de redes, indica claramente nodos y aristas.
"""

    log.debug(f"Prompt construido: {len(prompt)} caracteres, ~{len(prompt)//4} tokens estimados")

    # 3. Modelo adaptativo con reintentos
    last_error = None
    for model_idx, model_name in enumerate(CHAT_MODELS):
        log.info(f"--- Intentando modelo {model_idx+1}/{len(CHAT_MODELS)}: {model_name} ---")

        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
            log.debug(f"Modelo {model_name} inicializado correctamente")
        except Exception as e:
            log.error(f"No se pudo inicializar {model_name}: {e}")
            last_error = e
            continue

        for intento in range(3):
            try:
                log.info(f"Enviando prompt a {model_name} (intento {intento+1}/3)...")
                print(f"🤖 Consultando {model_name}...")

                t_start = time.time()
                respuesta = llm.invoke(prompt)
                t_elapsed = time.time() - t_start

                resultado = respuesta.content
                log.info(f"✅ Respuesta recibida de {model_name} en {t_elapsed:.1f}s ({len(resultado)} chars)")

                # Guardar en caché
                if use_cache:
                    cache[cache_key] = resultado
                    _save_response_cache(cache)
                    log.info(f"Respuesta guardada en caché (clave: {cache_key})")
                    print("💾 Respuesta guardada en caché")

                return resultado

            except Exception as e:
                last_error = e
                error_str = str(e)
                log.error(f"Error con {model_name} (intento {intento+1}/3): {error_str}")
                print(f"❌ Error con {model_name} (intento {intento+1}): {error_str[:200]}")

                if '429' in error_str and intento < 2:
                    espera = 2 ** intento * 2  # 2s, 4s
                    log.warning(f"Rate limit (429). Esperando {espera}s antes de reintentar...")
                    print(f"⚠️ Rate limit. Esperando {espera}s...")
                    time.sleep(espera)
                elif '404' in error_str or '429' in error_str:
                    log.warning(f"{model_name} no disponible ({error_str[:100]}). Saltando a siguiente modelo.")
                    print(f"⚠️ {model_name} no disponible, probando siguiente modelo...")
                    break  # Probar siguiente modelo
                else:
                    log.critical(f"Error inesperado (no es 429/404): {error_str}")
                    raise

    log.critical(f"TODOS LOS MODELOS FALLARON. Modelos probados: {CHAT_MODELS}")
    log.critical(f"Último error: {last_error}")
    raise RuntimeError(f"❌ Todos los modelos fallaron. Último error: {last_error}")


def limpiar_cache_respuestas():
    """Elimina el caché de respuestas."""
    if os.path.exists(RESPONSE_CACHE_FILE):
        os.remove(RESPONSE_CACHE_FILE)
        log.info("Caché de respuestas eliminado")
        print("🗑️ Caché de respuestas eliminado.")
    else:
        log.info("No hay caché de respuestas para eliminar")
        print("No hay caché de respuestas.")

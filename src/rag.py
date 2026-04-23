"""
Módulo RAG (Retrieval Augmented Generation).
Consulta a Gemini con contexto del libro de IO.
Optimizado para pay-as-you-go (reintentos rápidos).
"""
import os
import time
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from embeddings import search_context, get_embeddings_model
from config import CHAT_MODELS
from logging_config import get_logger

log = get_logger("rag")

# Caché Semántico usando ChromaDB
CACHE_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "semantic_cache_db")
_cache_vectorstore = None

def _get_cache_store():
    global _cache_vectorstore
    if _cache_vectorstore is None:
        _cache_vectorstore = Chroma(
            collection_name="semantic_cache",
            embedding_function=get_embeddings_model(),
            persist_directory=CACHE_DB_DIR
        )
    return _cache_vectorstore

def _get_from_cache(pregunta_usuario, umbral=0.15):
    """Busca preguntas similares en el vectorstore. Score menor a 0.15 es casi idéntico."""
    try:
        store = _get_cache_store()
        resultados = store.similarity_search_with_score(pregunta_usuario, k=1)
        if resultados:
            doc, score = resultados[0]
            if score <= umbral:
                log.info(f"Cache Semántico HIT (Score L2: {score:.4f})")
                return doc.metadata.get("response")
            else:
                log.debug(f"Cache Semántico MISS (Mejor Score L2: {score:.4f} > {umbral})")
    except Exception as e:
        log.error(f"Error leyendo caché semántico: {e}")
    return None

def _save_to_cache(pregunta_usuario, respuesta):
    """Guarda la pregunta como vector y la respuesta como metadato."""
    try:
        store = _get_cache_store()
        doc = Document(
            page_content=pregunta_usuario,
            metadata={"response": respuesta}
        )
        store.add_documents([doc])
        log.debug("Guardado en caché semántico exitoso.")
    except Exception as e:
        log.error(f"Error guardando en caché semántico: {e}")


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

    # 1. Verificar caché semántico
    if use_cache:
        cached_response = _get_from_cache(pregunta_usuario)
        if cached_response:
            print("⚡ Respuesta cargada desde Caché Semántico (0 API calls)")
            return cached_response

    # 2. Buscar contexto relevante
    log.info("Buscando contexto relevante en ChromaDB...")
    print("🔍 Buscando contexto relevante...")
    contexto = search_context(pregunta_usuario, vectorstore)
    log.debug(f"Contexto obtenido: {len(contexto)} caracteres")

    prompt = f"""
Actúa como un solver estricto y experto en Investigación de Operaciones. Responde directamente a lo que se pregunta sin rodeos, introducciones ni conclusiones innecesarias.

CONTEXTO (teoría extraída del libro de texto):
{contexto}

PROBLEMA:
{pregunta_usuario}

INSTRUCCIONES ESTRUCTURALES ESTRICTAS:
1. Clasificación: Indica el tipo de problema.
2. Formulación: Directo al grano (Variables, Objetivo, Restricciones). Identifica correctamente el orden lógico de predecesores (A va antes que B, etc) sin invertir el orden cronológico.
3. Resolución (Clara y Sencilla): Resume los cálculos paso a paso. Si hay muchísimos cálculos, preséntalos DIRECTAMENTE EN UNA TABLA Markdown estricta usando barras verticales y guiones. 
   EJEMPLO DE FORMATO OBLIGATORIO:
   | Actividad | Duración |
   |---|---|
   | A | 5 |
   NO escribas ecuación por ecuación hacia abajo ni uses simples espacios o tabulaciones.
4. Resultados: Conclusión clara, en términos muy sencillos y destacada en negritas.

REGLAS DE RESPUESTA CRÍTICAS (SIMPLICIDAD Y PEDAGOGÍA):
- Sé extremadamente didáctico, directo al grano y fácil de entender. Actúa como un tutor excelente.
- Evita saturar la pantalla con derivaciones matemáticas largas. Resume todo en tablas legibles.
- Escribe las fórmulas matemáticas compactas en una sola línea. No repitas la fórmula y luego el reemplazo en líneas separadas.
- Utiliza la metodología matemática descrita en el 'CONTEXTO' del libro, pero usa tu propio motor matemático interno para resolver y simplificar.
- NO uses conversacionalismos ni rellenos. Inicia directo con la solución.
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

                # Guardar en caché semántico
                if use_cache:
                    _save_to_cache(pregunta_usuario, resultado)
                    print("💾 Respuesta guardada en Caché Semántico local")

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
    """Destruye la base de datos de caché semántico."""
    if os.path.exists(CACHE_DB_DIR):
        try:
            global _cache_vectorstore
            _cache_vectorstore = None
            shutil.rmtree(CACHE_DB_DIR)
            log.info("Caché Semántico destruido con éxito.")
            print("🗑️ Historial de caché semántico limpiado.")
        except Exception as e:
            log.error(f"Error borrando caché semántico: {e}")
    else:
        log.info("No hay caché de respuestas para eliminar")
        print("No hay caché de respuestas.")

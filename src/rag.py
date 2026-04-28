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
            return {
                "response": cached_response,
                "pages": ["Caché Semántico"],
                "chunks": []
            }

    # 2. Buscar contexto relevante
    log.info("Buscando contexto relevante en ChromaDB...")
    print("🔍 Buscando contexto relevante...")
    contexto, paginas, chunks = search_context(pregunta_usuario, vectorstore)
    log.debug(f"Contexto obtenido: {len(contexto)} caracteres")

    prompt = f"""
Actúa como un tutor práctico y directo de Investigación de Operaciones. Tu objetivo es explicar los problemas al grano, combinando la matemática con una explicación humana, sencilla y muy fácil de entender paso a paso.

CONTEXTO (teoría extraída del libro de texto):
{contexto}

PROBLEMA:
{pregunta_usuario}

INSTRUCCIONES DE ESTILO Y ESTRUCTURA (EL FORMATO PERFECTO):
1. **Introducción Directa**: Inicia con un pequeño párrafo (1-2 líneas) que explique en lenguaje súper sencillo qué es lo que busca resolver este tipo de problema en la vida real.
2. **Paso a Paso Lógico**: Divide la solución en pasos numerados (ej. `1. El Modelo Matemático`, `2. El Paso hacia Adelante`, etc.).
   - En cada paso, explica *la lógica humana* antes de poner la matemática. Ej: "Aquí está el truco: como necesitas que ambas terminen, D debe esperar a la más lenta. Por eso se usa el máximo: $\max(10, 6) = 10$."
3. **Tablas Resumen**: Si hay muchos datos o es el resultado final (Holguras, Ruta Crítica, Iteración Símplex), preséntalos SIEMPRE en una tabla de Markdown limpia.
   | Actividad | Holgura | ¿Es Crítica? |
   |---|---|---|
   | A | 0 | Sí |
4. **Conclusión**: Termina con un mini resumen en viñetas de la decisión final (ej. Ruta Crítica, Costo Total, Tiempo de Espera).

REGLAS DE ORO DE FORMATO:
- **LaTeX Limpio**: Usa `$$ ... $$` para fórmulas que merezcan su propia línea, y `$ ... $` para variables dentro del texto (ej. "la actividad $E$ termina en $15$").
- **Cero Relleno**: NO saludes, NO te despidas, NO digas "Aquí tienes la respuesta". Ve directo a la "Introducción Directa".
- **Doble Salto de Línea**: Usa doble salto de línea entre párrafos y listas para que el texto respire y sea fácil de leer en la web.
- **Sin Redundancias**: No pongas la misma fórmula en texto plano y luego en LaTeX. Usa solo LaTeX.
- **Negritas**: Úsalas para resaltar conceptos clave como **Ruta Crítica** o **Función Objetivo**.
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

                return {
                    "response": resultado,
                    "pages": paginas,
                    "chunks": chunks
                }

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
    return {
        "response": f"Lo siento, ocurrió un error tras reintentar con múltiples modelos. Detalles del último error:\n\n{last_error}",
        "pages": paginas if 'paginas' in locals() else [],
        "chunks": chunks if 'chunks' in locals() else []
    }


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

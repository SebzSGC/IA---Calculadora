"""
Módulo de embeddings: Generación, caché con ChromaDB, y búsqueda semántica.
Usa ChromaDB como vector store persistente (local, sin costo adicional de API).
"""
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (PDF_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL,
                    BATCH_SIZE, WAIT_SECONDS, MAX_CHUNKS,
                    CHUNK_SIZE, CHUNK_OVERLAP)
from logging_config import get_logger

log = get_logger("embeddings")


def get_embeddings_model():
    """Retorna el modelo de embeddings configurado."""
    log.info(f"Inicializando modelo de embeddings: {EMBEDDING_MODEL}")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def load_and_split_pdf():
    """Carga el PDF y lo divide en fragmentos optimizados."""
    if not os.path.exists(PDF_PATH):
        log.error(f"PDF no encontrado en {PDF_PATH}")
        raise FileNotFoundError(f"❌ No se encontró el PDF en {PDF_PATH}")

    log.info(f"Cargando PDF desde {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documentos = loader.load()
    log.info(f"PDF cargado: {len(documentos)} páginas")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documentos)
    log.info(f"{len(chunks)} fragmentos creados (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_vectorstore_with_rate_limit(chunks, embeddings_model):
    """
    Crea el vector store ChromaDB procesando chunks en lotes.
    Optimizado para pay-as-you-go (rate limits holgados).
    """
    total = len(chunks) if MAX_CHUNKS is None else min(MAX_CHUNKS, len(chunks))
    chunks_to_process = chunks[:total]
    num_lotes = (total + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(f"Creando vectorstore: {total} fragmentos en {num_lotes} lotes (batch_size={BATCH_SIZE})")

    vectorstore = None

    for lote_idx in range(num_lotes):
        start = lote_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch_chunks = chunks_to_process[start:end]

        # Pausa entre lotes (excepto el primero)
        if lote_idx > 0 and WAIT_SECONDS > 0:
            log.debug(f"Pausa de {WAIT_SECONDS}s entre lotes")
            time.sleep(WAIT_SECONDS)

        # Crear o agregar al vector store con reintentos
        for intento in range(5):
            try:
                if vectorstore is None:
                    log.debug(f"Creando ChromaDB con primer lote ({len(batch_chunks)} chunks)")
                    vectorstore = Chroma.from_documents(
                        documents=batch_chunks,
                        embedding=embeddings_model,
                        persist_directory=CHROMA_DB_DIR
                    )
                else:
                    vectorstore.add_documents(batch_chunks)
                break
            except Exception as e:
                log.warning(f"Error en lote {lote_idx+1}, intento {intento+1}/5: {e}")
                if '429' in str(e) and intento < 4:
                    espera = 2 ** intento * 2  # 2s, 4s, 8s, 16s
                    log.info(f"Rate limit. Reintento en {espera}s...")
                    time.sleep(espera)
                else:
                    log.error(f"Fallo definitivo en lote {lote_idx+1}: {e}")
                    raise

        log.info(f"Lote {lote_idx + 1}/{num_lotes} completado ({end}/{total} fragmentos)")

    log.info(f"Vector store creado: {total} fragmentos. Persistido en {CHROMA_DB_DIR}")
    return vectorstore


def _add_chunks_in_batches(vectorstore, chunks, embeddings_model):
    """Agrega chunks al vectorstore existente en lotes con reintentos."""
    total = len(chunks)
    num_lotes = (total + BATCH_SIZE - 1) // BATCH_SIZE
    log.info(f"Agregando {total} chunks nuevos en {num_lotes} lotes")

    for lote_idx in range(num_lotes):
        start = lote_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = chunks[start:end]

        if lote_idx > 0 and WAIT_SECONDS > 0:
            time.sleep(WAIT_SECONDS)

        for intento in range(5):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                log.warning(f"Error agregando lote {lote_idx+1}, intento {intento+1}/5: {e}")
                if '429' in str(e) and intento < 4:
                    espera = 2 ** intento * 2
                    log.info(f"Rate limit. Reintento en {espera}s...")
                    time.sleep(espera)
                else:
                    log.error(f"Fallo definitivo agregando lote {lote_idx+1}: {e}")
                    raise

        log.info(f"Lote {lote_idx + 1}/{num_lotes} agregado ({end}/{total} fragmentos nuevos)")


def load_or_create_vectorstore():
    """
    Carga el vector store desde ChromaDB si existe,
    o lo crea desde el PDF si es la primera vez.
    Soporta actualización incremental: si el PDF tiene más chunks
    que los ya almacenados, solo genera embeddings para los nuevos.

    Retorna: (vectorstore, embeddings_model)
    """
    log.info("=== Iniciando carga/creación de vectorstore ===")
    embeddings_model = get_embeddings_model()

    # Cargar chunks del PDF (operación local, 0 API calls)
    chunks = load_and_split_pdf()
    total_deseado = len(chunks) if MAX_CHUNKS is None else min(MAX_CHUNKS, len(chunks))
    chunks = chunks[:total_deseado]
    log.info(f"Total de chunks deseados: {total_deseado}")

    # Verificar si ya existe ChromaDB persistido
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        log.info(f"ChromaDB encontrado en {CHROMA_DB_DIR}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings_model
        )
        existing = vectorstore._collection.count()
        log.info(f"Chunks existentes en ChromaDB: {existing}")

        if existing >= total_deseado:
            log.info(f"Vector store completo: {existing} fragmentos (0 API calls)")
            print(f"⚡ Vector store completo: {existing} fragmentos (0 API calls)")
            return vectorstore, embeddings_model

        # Modo incremental: solo procesar los chunks faltantes
        missing = chunks[existing:]
        log.info(f"Modo incremental: agregando {len(missing)} chunks faltantes")
        print(f"📈 ChromaDB tiene {existing}/{total_deseado} fragmentos")
        print(f"   Agregando {len(missing)} fragmentos nuevos...\n")
        _add_chunks_in_batches(vectorstore, missing, embeddings_model)

        final = vectorstore._collection.count()
        log.info(f"Vector store actualizado: {final} fragmentos totales")
        print(f"\n✅ Vector store actualizado: {final} fragmentos totales")
        return vectorstore, embeddings_model

    # Primera vez: generar desde cero
    log.info("ChromaDB no encontrado. Generando desde cero...")
    print("🔄 No se encontró ChromaDB. Generando embeddings por primera vez...\n")
    vectorstore = create_vectorstore_with_rate_limit(chunks, embeddings_model)
    return vectorstore, embeddings_model


def search_context(query, vectorstore, k=5):
    """
    Busca los k fragmentos más relevantes usando similitud coseno.
    ChromaDB usa coseno por defecto.

    Retorna: (contexto_texto, resultados_con_metadata)
    """
    log.debug(f"Buscando contexto para: '{query[:80]}...' (k={k})")
    results = vectorstore.similarity_search_with_score(query, k=k)

    paginas = [doc.metadata.get("page", "?") for doc, _ in results]
    scores = [f"{score:.3f}" for _, score in results]
    log.info(f"Contexto encontrado — Páginas: {paginas}, Scores: {scores}")
    print(f"  📖 Contexto extraído de páginas: {paginas}")
    print(f"  📊 Scores de distancia: {scores}")

    contexto = "\n---\n".join([doc.page_content for doc, _ in results])
    return contexto


def delete_vectorstore():
    """Elimina el vector store persistido para regenerar."""
    import shutil
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        log.info("ChromaDB eliminado")
        print("🗑️ ChromaDB eliminado. Re-ejecuta para regenerar.")
    else:
        log.info("No hay ChromaDB para eliminar")
        print("No hay ChromaDB para eliminar.")

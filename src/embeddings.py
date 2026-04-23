"""
Módulo de embeddings: Generación, caché con ChromaDB, y búsqueda semántica.
Usa ChromaDB como vector store persistente (local, sin costo adicional de API).
"""
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
import pymupdf4llm
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import (PDF_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL,
                    BATCH_SIZE, WAIT_SECONDS, MAX_CHUNKS,
                    CHUNK_SIZE, CHUNK_OVERLAP)
from logging_config import get_logger

log = get_logger("embeddings")


def get_embeddings_model():
    """Retorna el modelo de embeddings configurado."""
    log.info(f"Inicializando modelo local HF de embeddings: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_split_pdf():
    """Carga el PDF y lo divide semánticamente respetando jerarquías Markdown."""
    if not os.path.exists(PDF_PATH):
        log.error(f"PDF no encontrado en {PDF_PATH}")
        raise FileNotFoundError(f"❌ No se encontró el PDF en {PDF_PATH}")

    log.info(f"Cargando PDF y convirtiendo a Markdown con pymupdf4llm desde {PDF_PATH}. Esto puede demorar unos minutos...")
    try:
        md_chunks = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)
    except Exception as e:
        log.error(f"Error al procesar PDF con pymupdf4llm: {e}")
        raise
        
    full_text = ""
    for idx, chunk in enumerate(md_chunks):
        page_num = chunk.get("metadata", {}).get("page", idx) + 1
        page_text = chunk.get("text", "")
        if page_text.strip():
            # Inyectar marcador de página para que el LLM sepa de dónde viene
            full_text += f"\n\n---\n*PÁGINA {page_num}*\n\n" + page_text
            
    log.info("PDF cargado. Aplicando Semantic Chunking...")

    # 1. División Semántica basada en Títulos (mantiene ecuaciones unidas bajo su título)
    headers_to_split_on = [
        ("#", "Capítulo"),
        ("##", "Sección"),
        ("###", "Subsección"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(full_text)

    # 2. División secundaria para secciones exageradamente largas
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(md_header_splits)
    
    # Agregar source genérico a los metadatos
    for c in chunks:
        c.metadata["source"] = PDF_PATH

    log.info(f"{len(chunks)} fragmentos semánticos creados con éxito.")
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
    """
    log.info("=== Iniciando carga/creación de vectorstore ===")
    embeddings_model = get_embeddings_model()

    # Verificar si ya existe ChromaDB persistido
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        try:
            log.info(f"ChromaDB encontrado en {CHROMA_DB_DIR}")
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings_model
            )
            existing = vectorstore._collection.count()
            log.info(f"Chunks existentes en ChromaDB: {existing}")

            if existing > 0:
                log.info(f"Vector store completo: {existing} fragmentos cargados de memoria (0 API calls)")
                print(f"⚡ Vector store completo: {existing} fragmentos cargados rápido (0 API calls)")
                return vectorstore, embeddings_model
        except Exception as e:
            log.warning(f"ChromaDB parece estar corrupto o usando modelo viejo, se recreará. Error: {e}")

    # Si no existe o estaba corrupto: generar desde cero
    log.info("ChromaDB no encontrado o vacío. Generando desde cero...")
    print("🔄 No se encontró ChromaDB válido. Generando embeddings por primera vez...\n")
    
    # Cargar chunks del PDF (operación local)
    chunks = load_and_split_pdf()
    total_deseado = len(chunks) if MAX_CHUNKS is None else min(MAX_CHUNKS, len(chunks))
    chunks = chunks[:total_deseado]
    log.info(f"Total de chunks deseados: {total_deseado}")
    
    vectorstore = create_vectorstore_with_rate_limit(chunks, embeddings_model)
    return vectorstore, embeddings_model


from sentence_transformers import CrossEncoder

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        log.info("Inicializando modelo local de Re-Ranking (Cross-Encoder)...")
        # Este modelo pesa ~80MB y es extremadamente efectivo para reordenar resultados RAG
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def search_context(query, vectorstore, k=25, top_n=6):
    """
    1. Retrieval rápido: Busca k=25 fragmentos con similitud coseno (ChromaDB).
    2. Compresión (Re-Ranking): Evalúa los 25 fragmentos con un Cross-Encoder y retorna los top_n=6.
    """
    log.debug(f"Retrieval inicial: Buscando {k} contextos rápidos para: '{query[:80]}...'")
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = [doc for doc, _ in results]

    log.info("Aplicando Context Compression (Re-Ranking Cross-Encoder)...")
    ce = get_cross_encoder()
    
    # Crear pares [Pregunta, Contexto] para que la IA decida si el contexto responde la pregunta
    pairs = [[query, doc.page_content] for doc in docs]
    scores = ce.predict(pairs)
    
    # Emparejar documentos con sus nuevas calificaciones y ordenar de mayor a menor
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Extraer los top_n más perfectos
    best_results = scored_docs[:top_n]

    paginas = [doc.metadata.get("page", "?") for doc, _ in best_results]
    ce_scores = [f"{score:.2f}" for _, score in best_results]
    
    log.info(f"Contexto comprimido (Top {top_n}) — Páginas: {paginas}, CE Scores: {ce_scores}")
    print(f"  📖 Contexto ultra-preciso (Re-Rank) extraído de páginas: {paginas}")

    contexto = "\n---\n".join([doc.page_content for doc, _ in best_results])
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

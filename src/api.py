import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import preguntar_io, limpiar_cache_respuestas
from embeddings import load_or_create_vectorstore

app = FastAPI(title="IA Calculadora API", description="API para el solver de Investigación de Operaciones con RAG")

# Permitir CORS para el frontend en React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción se debe restringir al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SolveRequest(BaseModel):
    problem: str
    use_cache: bool = True

class SolveResponse(BaseModel):
    response: str
    source: str = "Gemini/RAG"

# Variable global para cargar la base vectorial en memoria al iniciar
vectorstore = None

@app.on_event("startup")
async def startup_event():
    global vectorstore
    print("Iniciando API y asegurando conexión a ChromaDB local...")
    try:
        # Carga silenciosa del vectorstore
        vs, _ = load_or_create_vectorstore()
        vectorstore = vs
        print("Vectorstore cargado exitosamente en RAM para la API.")
    except Exception as e:
        print(f"Error cargando vectorstore: {e}")

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore no inicializado. Revisa los logs.")
    
    if not request.problem or not request.problem.strip():
        raise HTTPException(status_code=400, detail="El problema no puede estar vacío.")

    try:
        resultado = preguntar_io(request.problem, vectorstore, use_cache=request.use_cache)
        return SolveResponse(response=resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la resolución: {str(e)}")

@app.delete("/cache")
async def clear_cache():
    try:
        limpiar_cache_respuestas()
        return {"status": "success", "message": "Caché limpiado correctamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Inicia en el puerto 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

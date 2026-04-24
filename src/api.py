import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag import preguntar_io, limpiar_cache_respuestas
from embeddings import load_or_create_vectorstore, delete_vectorstore

app = FastAPI(title="IA Calculadora API", description="API para el solver de Investigación de Operaciones con RAG")

# Permitir CORS para el frontend en React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción se debe restringir al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar la carpeta estática para poder ver el PDF desde el navegador (RAG Explorer)
insumos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Insumos')
if os.path.exists(insumos_path):
    app.mount("/pdf", StaticFiles(directory=insumos_path), name="pdf")

class SolveRequest(BaseModel):
    problem: str
    use_cache: bool = True

class SolveResponse(BaseModel):
    response: str
    source: str = "Gemini/RAG"
    pages: list = []
    chunks: list = []

# Variable global para cargar la base vectorial en memoria al iniciar
vectorstore = None
db_status = "initializing"

@app.on_event("startup")
async def startup_event():
    global vectorstore, db_status
    print("Iniciando API y asegurando conexión a ChromaDB local...")
    try:
        # Carga silenciosa del vectorstore
        vs, _ = load_or_create_vectorstore()
        vectorstore = vs
        db_status = "ready"
        print("Vectorstore cargado exitosamente en RAM para la API.")
    except Exception as e:
        db_status = "error"
        print(f"Error cargando vectorstore: {e}")

def rebuild_task():
    global vectorstore, db_status
    try:
        delete_vectorstore()
        vectorstore = None
        vs, _ = load_or_create_vectorstore()
        vectorstore = vs
        db_status = "ready"
    except Exception as e:
        db_status = "error"
        print(f"Error reconstruyendo vectorstore: {e}")

@app.post("/rebuild-db")
async def rebuild_db(background_tasks: BackgroundTasks):
    global db_status
    if db_status == "building":
        raise HTTPException(status_code=400, detail="La base de datos ya se está reconstruyendo.")
    db_status = "building"
    background_tasks.add_task(rebuild_task)
    return {"status": "started", "message": "Reconstrucción iniciada en segundo plano."}

@app.get("/db-status")
async def get_db_status():
    return {"status": db_status}

@app.post("/solve", response_model=SolveResponse)
async def solve_problem(request: SolveRequest):
    global db_status
    if db_status == "building":
        raise HTTPException(status_code=503, detail="La base de conocimientos se está reconstruyendo. Por favor espera.")
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore no inicializado. Revisa los logs.")
    
    if not request.problem or not request.problem.strip():
        raise HTTPException(status_code=400, detail="El problema no puede estar vacío.")

    try:
        resultado_dict = preguntar_io(request.problem, vectorstore, use_cache=request.use_cache)
        return SolveResponse(
            response=resultado_dict["response"],
            pages=resultado_dict["pages"],
            chunks=resultado_dict["chunks"]
        )
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

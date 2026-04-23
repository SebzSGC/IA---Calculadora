# 🧮 IA Calculadora — Investigación de Operaciones

Sistema inteligente con **RAG (Retrieval Augmented Generation)** sobre Gemini para resolver problemas de Investigación de Operaciones.

> Le da a Gemini el contexto de un libro de IO (Hillier, 10ª edición) para que responda con base en la teoría, no en su conocimiento general.

---

## 🏗️ Arquitectura

```
📕 PDF del libro (1,229 páginas)
    ↓ se fragmenta y vectoriza (una sola vez)
💾 ChromaDB (base de datos vectorial local)
    ↓ ante cada pregunta, busca los fragmentos más relevantes
🤖 Gemini 2.5 Flash (responde con contexto del libro)
    ↓ guarda la respuesta
📦 Caché local (no repite consultas idénticas)
```

### Módulos

| Archivo | Función |
|---|---|
| `config.py` | Configuración centralizada (API key, modelos, rutas) |
| `logging_config.py` | Logging a consola y archivo en `logs/` |
| `embeddings.py` | Generación de embeddings, ChromaDB, búsqueda semántica |
| `rag.py` | Consultas a Gemini con contexto del libro (RAG) |
| `codigo.ipynb` | **Notebook principal** con ejemplos y consulta libre |

### Capacidades del RAG con Gemini

El sistema puede resolver cualquier problema de IO basándose estrictamente en el texto, incluyendo:
- Programación Lineal Continua y Entera
- Flujo máximo y ruta más corta en redes
- PERT/CPM (gestión de proyectos)
- Cualquier problema de IO descrito en lenguaje natural

---

## 🚀 Instalación y ejecución local

### Prerrequisitos

- Python 3.11+
- API Key de [Google AI Studio](https://aistudio.google.com/apikey) (plan gratuito o pay-as-you-go)

### 1. Clonar el repositorio

```bash
git clone https://github.com/SebzSGC/IA---Calculadora.git
cd IA---Calculadora
```

### 2. Crear el ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install langchain-classic langchain-community langchain-google-genai langchain-text-splitters chromadb pypdf python-dotenv
```

### 4. Configurar API Key

Crea el archivo `src/.env`:

```
GOOGLE_API_KEY=tu_api_key_aqui
```

### 5. Agregar el PDF del libro

Coloca el PDF en:

```
Insumos/Investigacion-Operaciones10Edicion-Frederick-S-Hillier.pdf
```

### 6. Ejecutar

Abre `src/codigo.ipynb` en VS Code o Jupyter y ejecuta las celdas en orden:

1. **Celda 1** — Carga los módulos
2. **Celda 2** — Genera embeddings del PDF (solo la primera vez, ~5 min). Las siguientes veces carga instantáneamente desde ChromaDB.
3. **Celdas 3-7** — Diferentes ejemplos de resolución de problemas usando Gemini (RAG).

---

## 💰 Costos

| Operación | Costo estimado |
|---|---|
| Generar embeddings (una vez) | ~$0.10 USD |
| Cada consulta RAG | ~$0.004 USD |

Con uso moderado (~100 consultas/mes): **~$0.40 USD/mes**.

---

## 📁 Estructura del proyecto

```
IA---Calculadora/
├── .gitignore
├── Insumos/                    # PDF del libro
├── chroma_db/                  # Base vectorial (se genera automáticamente)
├── logs/                       # Logs de ejecución
├── response_cache.json         # Caché de respuestas de Gemini
└── src/
    ├── .env                    # API Key (no se sube a Git)
    ├── config.py               # Configuración
    ├── logging_config.py       # Sistema de logging
    ├── embeddings.py           # Embeddings + ChromaDB
    ├── rag.py                  # RAG con Gemini
    └── codigo.ipynb            # Notebook principal
```

---

## 🛠️ Tecnologías

- **Python 3.14** — Lenguaje principal
- **LangChain** — Orquestación de LLMs y embeddings
- **Google Gemini 2.5 Flash** — Modelo de IA para razonamiento matemático
- **ChromaDB** — Base de datos vectorial local
- **PyPDF** — Extracción de texto de PDFs

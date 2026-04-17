"""
Script de prueba — ejecuta una consulta RAG y captura todos los logs.
Usar para diagnosticar errores sin depender del notebook.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TEST: Cargando módulos...")
print("=" * 60)

from config import GOOGLE_API_KEY, CHAT_MODELS
from logging_config import LOG_FILE

print(f"✅ API Key cargada (termina en ...{GOOGLE_API_KEY[-4:]})")
print(f"✅ Modelos configurados: {CHAT_MODELS}")
print(f"📝 Log guardándose en: {LOG_FILE}")

print("\n" + "=" * 60)
print("TEST: Cargando vectorstore...")
print("=" * 60)

from embeddings import load_or_create_vectorstore
vectorstore, embeddings_model = load_or_create_vectorstore()

print("\n" + "=" * 60)
print("TEST: Ejecutando consulta RAG (flujo máximo)...")
print("=" * 60)

from rag import preguntar_io

pregunta_redes = """
Encontrar el flujo máximo desde el nodo origen S hasta el nodo destino T en la siguiente red:

S -> A: capacidad 10
S -> B: capacidad 5
A -> B: capacidad 15
A -> C: capacidad 10
B -> C: capacidad 5
B -> T: capacidad 10
C -> T: capacidad 10
"""

try:
    resultado = preguntar_io(pregunta_redes, vectorstore)
    print("\n" + "=" * 60)
    print("✅ RESULTADO:")
    print("=" * 60)
    print(resultado[:500])
except Exception as e:
    print(f"\n❌ ERROR: {e}")

print(f"\n📝 Log completo guardado en: {LOG_FILE}")

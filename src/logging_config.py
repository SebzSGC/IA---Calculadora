"""
Configuración centralizada de logging para IA Calculadora IO.
Genera logs en consola y en archivo dentro de logs/.
"""
import os
import logging
from datetime import datetime

# Directorio de logs
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Nombre del archivo de log con timestamp
LOG_FILE = os.path.join(LOGS_DIR, f"io_calc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def get_logger(name: str) -> logging.Logger:
    """
    Crea un logger con salida a consola y archivo.

    Args:
        name: Nombre del módulo (ej: 'rag', 'embeddings', 'solvers')

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(f"io_calc.{name}")

    # Evitar duplicar handlers si se llama múltiples veces
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formato detallado para archivo
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Formato limpio para consola
    console_formatter = logging.Formatter(
        "%(levelname)-8s | %(name)-20s | %(message)s"
    )

    # Handler: archivo (DEBUG y superior — captura todo)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)

    # Handler: consola (INFO y superior — solo lo importante)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# Logger raíz del proyecto
root_logger = get_logger("main")
root_logger.info(f"📝 Log iniciado: {LOG_FILE}")

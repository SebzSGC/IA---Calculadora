"""
Solucionadores numéricos para Investigación de Operaciones.
Incluye: scipy linprog, PuLP (LP/IP), Dijkstra.
"""
import heapq
from collections import defaultdict
from scipy.optimize import linprog
import pulp


# ==================== PROGRAMACIÓN LINEAL (scipy) ====================

def resolver_lp_scipy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Resuelve problemas de programación lineal con scipy.

    Parámetros:
    - c: vector de coeficientes de la función objetivo (minimización)
    - A_ub, b_ub: restricciones de desigualdad (A_ub * x <= b_ub)
    - A_eq, b_eq: restricciones de igualdad (A_eq * x == b_eq)
    - bounds: lista de tuplas (min, max) para cada variable

    Retorna: (estado, solución, valor_objetivo, mensaje)
    """
    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            return "ÓPTIMO", result.x, result.fun, result.message
        else:
            return "NO_ÓPTIMO", None, None, result.message
    except Exception as e:
        return "ERROR", None, None, str(e)


# ==================== PROGRAMACIÓN LINEAL ENTERA (PuLP) ====================

def resolver_lp_pulp(problema_tipo="maximizar"):
    """
    Crea un problema de LP/IP con PuLP.
    Retorna un objeto problema listo para agregar variables y restricciones.
    """
    if problema_tipo == "maximizar":
        prob = pulp.LpProblem("Problema_IO", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("Problema_IO", pulp.LpMinimize)
    return prob


def agregar_variable_pulp(prob, nombre, lowBound=0, upBound=None, cat='Continuous'):
    """
    Agrega una variable al problema de PuLP.
    cat: 'Continuous', 'Integer', 'Binary'
    """
    variable = pulp.LpVariable(nombre, lowBound=lowBound, upBound=upBound, cat=cat)
    return variable


def resolver_pulp(prob):
    """
    Resuelve el problema de PuLP y retorna resultados.
    Retorna: (estado, resultados_dict, valor_objetivo)
    """
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == 'Optimal':
        resultados = {}
        for v in prob.variables():
            resultados[v.name] = v.varValue
        return "ÓPTIMO", resultados, pulp.value(prob.objective)
    else:
        return "NO_ÓPTIMO", None, None


# ==================== DIJKSTRA (Rutas más cortas) ====================

class GrafoDijkstra:
    """Grafo para encontrar rutas más cortas con el algoritmo de Dijkstra."""

    def __init__(self):
        self.grafo = defaultdict(list)

    def agregar_arista(self, origen, destino, peso):
        """Agrega una arista dirigida."""
        self.grafo[origen].append((destino, peso))

    def agregar_arista_no_dirigida(self, nodo1, nodo2, peso):
        """Agrega una arista no dirigida (bidireccional)."""
        self.grafo[nodo1].append((nodo2, peso))
        self.grafo[nodo2].append((nodo1, peso))

    def ruta_mas_corta(self, inicio, fin):
        """
        Encuentra la ruta más corta usando Dijkstra.
        Retorna: (distancia_minima, lista_de_nodos_ruta)
        """
        distancias = {nodo: float('infinity') for nodo in self.grafo}
        distancias[inicio] = 0
        padres = {nodo: None for nodo in self.grafo}

        pq = [(0, inicio)]
        visitados = set()

        while pq:
            distancia_actual, nodo_actual = heapq.heappop(pq)

            if nodo_actual in visitados:
                continue

            visitados.add(nodo_actual)

            if nodo_actual == fin:
                break

            for vecino, peso in self.grafo[nodo_actual]:
                distancia = distancia_actual + peso

                if distancia < distancias[vecino]:
                    distancias[vecino] = distancia
                    padres[vecino] = nodo_actual
                    heapq.heappush(pq, (distancia, vecino))

        # Reconstruir ruta
        ruta = []
        nodo_actual = fin
        while nodo_actual is not None:
            ruta.insert(0, nodo_actual)
            nodo_actual = padres[nodo_actual]

        if distancias[fin] == float('infinity'):
            return None, None

        return distancias[fin], ruta

    def mostrar_grafo(self):
        """Muestra la estructura del grafo."""
        for nodo in self.grafo:
            print(f"{nodo} -> {dict(self.grafo[nodo])}")

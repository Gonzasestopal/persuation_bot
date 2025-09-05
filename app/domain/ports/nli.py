from typing import Dict, Protocol


class NLIPort(Protocol):
    """
    Puerto (interfaz) para proveedores de NLI (Natural Language Inference).
    Define los métodos mínimos que debe cumplir cualquier adaptador NLI.
    """

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Evalúa una relación premise → hypothesis.
        Retorna probabilidades normalizadas por etiqueta
        (ej: {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}).
        """
        pass

    def bidirectional_scores(
        self, premise: str, hypothesis: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evalúa en ambas direcciones (premise→hypothesis y hypothesis→premise),
        y devuelve además un dict `agg_max` con el máximo por etiqueta.
        """
        pass

    def contradiction_max(self, premise: str, hypothesis: str) -> float:
        """
        Devuelve la probabilidad máxima de contradicción entre ambas direcciones.
        """
        pass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Normaliza posibles variantes de etiquetas que traen algunos modelos
_LABEL_ALIAS = {
    'entailment': 'entailment',
    'entailed': 'entailment',
    'neutral': 'neutral',
    'contradiction': 'contradiction',
    'contradict': 'contradiction',
    'contradictory': 'contradiction',
}


class HFNLIProvider:
    def __init__(
        self,
        model_name: str = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli',
        device=None,
        max_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # buena práctica para inferencia
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_length = max_length

        # id2label puede venir con claves str/int y nombres en mayúsculas
        id2label = getattr(self.model.config, 'id2label', {}) or {}
        self.id2label = {}
        for k, v in id2label.items():
            try:
                idx = int(k)
            except Exception:
                idx = k
            name = _LABEL_ALIAS.get(str(v).strip().lower(), str(v).strip().lower())
            self.id2label[idx] = name

        # Fallback defensivo (raro, pero por si acaso)
        if not self.id2label:
            self.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        self.label2id = {v: k for k, v in self.id2label.items()}
        self.labels = set(self.id2label.values())  # evita AttributeError más adelante

    @torch.inference_mode()
    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return {self.id2label[i]: float(probs[i]) for i in range(probs.shape[-1])}

    @torch.inference_mode()
    def bidirectional_scores(self, premise: str, hypothesis: str) -> dict:
        """
        Devuelve p→h, h→p y agg_max (máximo por etiqueta entre direcciones).
        Mantiene compatibilidad con tests que esperan 'agg_max'.
        """
        s_ph = self.score(premise, hypothesis)
        s_hp = self.score(hypothesis, premise)

        # Robusto: no dependas sólo de self.labels; une claves detectadas
        labels = set(s_ph.keys()) | set(s_hp.keys()) | self.labels
        agg = {lbl: max(s_ph.get(lbl, 0.0), s_hp.get(lbl, 0.0)) for lbl in labels}

        return {'p_to_h': s_ph, 'h_to_p': s_hp, 'agg_max': agg}

    def contradiction_max(self, premise: str, hypothesis: str) -> float:
        return self.bidirectional_scores(premise, hypothesis)['agg_max'].get(
            'contradiction', 0.0
        )

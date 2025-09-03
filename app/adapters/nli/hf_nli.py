import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFNLIProvider:
    def __init__(self, model_name='roberta-large-mnli', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

    def score(self, premise: str, hypothesis: str):
        enc = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=512, return_tensors='pt'
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return {self.label_map[i]: float(probs[i]) for i in range(len(probs))}

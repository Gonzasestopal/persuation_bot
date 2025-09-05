from app.domain.concession_policy import DebateState

_VERDICT_LINE = {
    'en': 'On balance, the opposing argument addressed key counters with evidence and causality. I concede the point.',
    'es': 'En conjunto, el argumento contrario abordó los puntos clave con evidencia y causalidad. Cedo el punto.',
    'pt': 'No conjunto, o argumento oposto tratou os pontos-chave com evidência e causalidade. Eu cedo o ponto.',
    'fr': 'Dans l’ensemble, l’argument adverse a répondu aux points clés avec des preuves et une chaîne causale. J’accorde le point.',
    'de': 'Insgesamt hat das Gegenargument die wichtigsten Einwände mit Belegen und Kausalität adressiert. Ich gebe den Punkt ab.',
    'it': 'Nel complesso, l’argomentazione opposta ha affrontato i punti chiave con prove e causalità. Concedo il punto.',
}

AFTER_END_MESSAGE = {
    'en': 'The debate has already ended. Please start a new conversation if you want to debate another topic.',
    'es': 'El debate ya terminó. Por favor inicia una nueva conversación si quieres debatir otro tema.',
    'pt': 'O debate já terminou. Por favor, inicie uma nova conversa se quiser debater outro tema.',
    # ... add more as needed
}


def build_verdict(state: DebateState) -> str:
    """
    Return a localized verdict string. Optionally append the invariant token
    'Match concluded.' in English to satisfy tests / logs.

    Server already knows the match is over; this text is purely user-facing.
    """
    return _VERDICT_LINE.get(state.lang)


def after_end_message(state: DebateState) -> str:
    return AFTER_END_MESSAGE.get(state.lang)

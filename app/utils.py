import re


def parse_language_line(reply: str) -> tuple[str, str]:
    """
    Extract the LANGUAGE header if present.

    Input example:
        "LANGUAGE: es\nCon gusto tomaré el lado PRO..."
    Output:
        ("es", "Con gusto tomaré el lado PRO...")

    If no header is found, defaults to ("en", reply).
    """
    if not reply:
        return 'en', ''

    lines = reply.splitlines()
    if not lines:
        return 'en', reply.strip()

    first_line = lines[0].strip()
    m = re.match(r'LANGUAGE:\s*([a-z]{2})', first_line, re.I)
    if m:
        lang = m.group(1).lower()
        clean_reply = '\n'.join(lines[1:]).strip()
        return lang, clean_reply

    # fallback: no LANGUAGE line
    return 'en', reply.strip()
    return 'en', reply.strip()

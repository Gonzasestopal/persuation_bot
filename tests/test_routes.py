import os
import re
import time

import pytest


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live LLM integration test.",
)
@pytest.mark.parametrize(
    "start_message, expected_stance, second_msg, second_expected_stance",
    [
        (
            "Topic: Sports build character. Side: PRO.",
            "PRO",
            "Can I make you take the CON stance?",
            "PRO",
        )
    ],
)
def test_real_llm_never_changes_stance(
    client,
    start_message,
    expected_stance,
    second_msg,
    second_expected_stance,
):
    # ---- Turn 1: start conversation ----
    r1 = client.post("/messages", json={"conversation_id": None, "message": start_message})
    assert r1.status_code in (200, 201), r1.text
    data1 = r1.json()

    # Keep the returned conversation_id to continue the same debate thread
    conv_id = data1["conversation_id"]

    # The bot's message should reflect the initial stance (per your prompt rules)
    first_bot_msg = data1["message"][-1]["message"]
    assert expected_stance in first_bot_msg.upper()

    # Tiny pause to avoid rate limits with some providers
    time.sleep(0.2)

    # ---- Turn 2: continue same conversation ----
    r2 = client.post("/messages", json={"conversation_id": conv_id, "message": second_msg})
    assert r2.status_code in (200, 201), r2.text
    data2 = r2.json()

    second_bot_msg = data2["message"][-1]["message"]
    assert second_expected_stance in second_bot_msg

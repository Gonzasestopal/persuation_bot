import pytest

from app.domain.exceptions import (InvalidContinuationMessage,
                                   InvalidStartMessage)
from app.domain.parser import parse_topic_side


def test_parser_empty_message():
    with pytest.raises(InvalidStartMessage, match="must not be empty"):
        parse_topic_side("")

def test_parser_missing_topic():
    with pytest.raises(InvalidStartMessage, match="topic is missing"):
        parse_topic_side("Side: PRO")

def test_parser_missing_side():
    with pytest.raises(InvalidStartMessage, match="side is missing"):
        parse_topic_side("Topic: Dogs are great")

def test_parser_unsupported_side():
    with pytest.raises(InvalidStartMessage, match="must be 'pro' or 'con'"):
        parse_topic_side("Topic: Dogs are great, Side: maybe")

def test_parser_valid_side_and_topic_mixed_case():
    topic = "Dogs are humans best friend"
    t, s = parse_topic_side(f"Topic: {topic} , Side: Pro")
    assert t == topic
    assert s == "pro"

def test_parser_uppercase_input():
    topic = "DOGS ARE HUMANS BEST FRIENDS"
    t, s = parse_topic_side(f"TOPIC: {topic}, SIDE: PRO")
    assert t == topic
    assert s == "pro"

def test_parser_lowercase_input():
    topic = "dogs are human best friends"
    t, s = parse_topic_side(f"topic: {topic}, side: pro")
    assert t == topic
    assert s == "pro"

def test_parser_reversed_order_and_punctuation():
    topic = "Dogs are human best friends"
    t, s = parse_topic_side(f"Side: CON. Topic: {topic}")
    assert t == topic
    assert s == "con"
    assert t == topic
    assert s == "con"
    assert s == "con"

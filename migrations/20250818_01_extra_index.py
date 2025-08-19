from yoyo import step

steps = [
    step(
        """
        CREATE INDEX idx_messages_conv_created_id_desc
        ON messages (conversation_id, created_at DESC, message_id DESC);
        """,
        """
        DROP INDEX IF EXISTS idx_messages_conv_created_id_desc;
        """
    ),
]

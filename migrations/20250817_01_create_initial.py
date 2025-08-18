from yoyo import step

steps = [
    step(
        """
        CREATE TABLE conversations (
            conversation_id BIGSERIAL PRIMARY KEY,
            topic VARCHAR(50) NOT NULL,
            side VARCHAR(10) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '1 hour')
        );
        CREATE INDEX idx_conversations_expires_at ON conversations(expires_at);
        """,
        """
        DROP INDEX IF EXISTS idx_conversations_expires_at;
        DROP TABLE conversations;
        """
    ),

    step(
        """
        CREATE TABLE messages (
            message_id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL
                REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            message TEXT NOT NULL,
            role VARCHAR(10) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_messages_conv_created_at
        ON messages (conversation_id, created_at);
        """,
        """
        DROP INDEX IF EXISTS idx_messages_conv_created_at;
        DROP TABLE messages;
        """
    ),
]

from yoyo import step

steps = [
    step(
        """
        ALTER TABLE conversations
        ALTER COLUMN topic TYPE TEXT;
        """,
        """
        ALTER TABLE conversations
        ALTER COLUMN topic TYPE VARCHAR(50);
        """,
    ),
]

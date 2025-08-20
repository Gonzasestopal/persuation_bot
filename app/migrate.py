import os

from dotenv import load_dotenv
from yoyo import get_backend, read_migrations


def run_migrations():
    load_dotenv()
    dburl = os.environ["DATABASE_URL"]
    backend = get_backend(dburl)
    migrations = read_migrations("migrations")
    with backend.lock():
        to_apply = backend.to_apply(migrations)
        if to_apply:
            backend.apply_migrations(to_apply)

"""Alembic environment.

init_db() passes in an open connection; the alembic CLI falls back to the configured database.
"""

from alembic import context
from sqlalchemy import Connection

from review_analyzer.config import Settings
from review_analyzer.storage import Base, make_engine


def run_migrations(connection: Connection) -> None:
    # Batch mode lets SQLite handle ALTER operations it doesn't support natively.
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    run_migrations(connection)
else:
    with make_engine(Settings().database_url).connect() as conn:
        run_migrations(conn)

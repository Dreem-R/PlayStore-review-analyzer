from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect, text

from review_analyzer.storage import Base, init_db, make_engine


def columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("reviews")}


def test_migrations_match_the_models() -> None:
    engine = make_engine("sqlite://")
    init_db(engine)
    with engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_init_db_is_idempotent() -> None:
    engine = make_engine("sqlite://")
    init_db(engine)
    init_db(engine)
    assert "text_sentiment" in columns(engine)


def test_upgrades_database_created_before_migrations() -> None:
    """Databases from v0.2 were made with create_all and have no alembic_version table."""
    engine = make_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE reviews (review_id VARCHAR PRIMARY KEY, app_id VARCHAR NOT NULL, "
                "user_name VARCHAR NOT NULL, rating INTEGER NOT NULL, content TEXT NOT NULL, "
                "created_at DATETIME NOT NULL, app_version VARCHAR, thumbs_up INTEGER NOT NULL, "
                "fetched_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO reviews VALUES ('r1', 'com.example.notes', 'Ann', 5, 'Great', "
                "'2026-10-01 00:00:00', '1.0', 0, '2026-10-01 00:00:00')"
            )
        )

    init_db(engine)

    assert {"text_sentiment", "sentiment_score"} <= columns(engine)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM reviews")).scalar() == 1
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0002"
        )

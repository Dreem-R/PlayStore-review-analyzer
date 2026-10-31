"""Log each Play Store fetch.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "syncs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("app_id", sa.String(), nullable=False),
        sa.Column("synced_at", sa.DateTime(), nullable=False),
        sa.Column("requested", sa.Integer(), nullable=False),
        sa.Column("fetched", sa.Integer(), nullable=False),
    )
    op.create_index("ix_syncs_app_synced", "syncs", ["app_id", "synced_at"])
    op.create_index("ix_syncs_synced_at", "syncs", ["synced_at"])


def downgrade() -> None:
    op.drop_index("ix_syncs_synced_at", table_name="syncs")
    op.drop_index("ix_syncs_app_synced", table_name="syncs")
    op.drop_table("syncs")

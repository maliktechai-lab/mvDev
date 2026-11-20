"""add_login_lockout_columns

Revision ID: a1f3c8e02b47
Revises: d4b7fe266094
Create Date: 2026-11-20 00:00:00.000000

Adds brute-force / account-lockout tracking columns to the users table:
  - failed_login_attempts  INTEGER  NOT NULL DEFAULT 0
  - locked_until           TIMESTAMPTZ  NULLABLE

These are used by the login endpoint to implement exponential-backoff
account lockouts after repeated failed attempts.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'a1f3c8e02b47'
down_revision: Union[str, None] = 'd4b7fe266094'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'failed_login_attempts',
            sa.Integer(),
            nullable=False,
            server_default='0',
        ),
    )
    op.add_column(
        'users',
        sa.Column(
            'locked_until',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column('users', 'locked_until')
    op.drop_column('users', 'failed_login_attempts')

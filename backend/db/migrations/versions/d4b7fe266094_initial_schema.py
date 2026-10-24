"""initial_schema

Revision ID: d4b7fe266094
Revises:
Create Date: 2026-10-23 16:09:37.283973

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd4b7fe266094'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enums ──────────────────────────────────────────────────────────────
    mediatype_enum = sa.Enum('image', 'video', 'audio', name='mediatype')
    jobstatus_enum = sa.Enum('pending', 'running', 'completed', 'failed', name='jobstatus')
    verdict_enum   = sa.Enum('real', 'fake', 'unknown', name='verdict')

    mediatype_enum.create(op.get_bind(), checkfirst=True)
    jobstatus_enum.create(op.get_bind(), checkfirst=True)
    verdict_enum.create(op.get_bind(), checkfirst=True)

    # ── users ──────────────────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id',              postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email',           sa.String(255), nullable=False),
        sa.Column('username',        sa.String(50),  nullable=False),
        sa.Column('full_name',       sa.String(100), nullable=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active',       sa.Boolean(),   nullable=False, server_default=sa.true()),
        sa.Column('is_admin',        sa.Boolean(),   nullable=False, server_default=sa.false()),
        sa.Column('created_at',      sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at',      sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_users_email',    'users', ['email'],    unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    # ── detection_jobs ─────────────────────────────────────────────────────
    op.create_table(
        'detection_jobs',
        sa.Column('id',                postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id',           postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('celery_task_id',    sa.String(255), nullable=True),
        sa.Column('original_filename', sa.String(500), nullable=False),
        sa.Column('stored_filename',   sa.String(500), nullable=False),
        sa.Column('media_type',        sa.Enum('image', 'video', 'audio', name='mediatype',
                                               create_type=False), nullable=False),
        sa.Column('file_size_bytes',   sa.Integer(),   nullable=False),
        sa.Column('file_path',         sa.String(1000), nullable=False),
        sa.Column('status',            sa.Enum('pending', 'running', 'completed', 'failed',
                                               name='jobstatus', create_type=False),
                  nullable=False, server_default='pending'),
        sa.Column('error_msg',         sa.Text(),       nullable=True),
        sa.Column('started_at',        sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at',          sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at',        sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_detection_jobs_user_id',       'detection_jobs', ['user_id'])
    op.create_index('ix_detection_jobs_status',        'detection_jobs', ['status'])
    op.create_index('ix_detection_jobs_celery_task_id','detection_jobs', ['celery_task_id'])

    # ── detection_results ──────────────────────────────────────────────────
    op.create_table(
        'detection_results',
        sa.Column('id',              postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id',          postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('detection_jobs.id', ondelete='CASCADE'),
                  nullable=False, unique=True),
        sa.Column('verdict',         sa.Enum('real', 'fake', 'unknown', name='verdict',
                                             create_type=False), nullable=False),
        sa.Column('confidence',      sa.Float(), nullable=False),
        sa.Column('branch_scores',   postgresql.JSON(), nullable=True),
        sa.Column('frame_scores',    postgresql.JSON(), nullable=True),
        sa.Column('chunk_scores',    postgresql.JSON(), nullable=True),
        sa.Column('heatmap_path',    sa.String(1000), nullable=True),
        sa.Column('heatmap_url',     sa.String(1000), nullable=True),
        sa.Column('duration_sec',    sa.Float(),   nullable=True),
        sa.Column('frames_analyzed', sa.Integer(), nullable=True),
        sa.Column('faces_detected',  sa.Integer(), nullable=True),
        sa.Column('raw_output',      postgresql.JSON(), nullable=True),
        sa.Column('created_at',      sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── refresh_tokens ─────────────────────────────────────────────────────
    op.create_table(
        'refresh_tokens',
        sa.Column('id',         postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id',    postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('revoked',    sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])


def downgrade() -> None:
    op.drop_table('refresh_tokens')
    op.drop_table('detection_results')
    op.drop_table('detection_jobs')
    op.drop_table('users')

    # Drop enums explicitly (PostgreSQL keeps them after table drop)
    sa.Enum(name='verdict').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='jobstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='mediatype').drop(op.get_bind(), checkfirst=True)

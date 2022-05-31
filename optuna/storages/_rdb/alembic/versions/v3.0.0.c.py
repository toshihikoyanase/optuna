"""Add intermediate_value_type column to represent +inf and -inf

Revision ID: v3.0.0.c
Revises: v3.0.0.b
Create Date: 2022-05-16 17:17:28.810792

"""
import enum

import numpy as np
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import orm
from typing import Optional
from typing import Tuple


# revision identifiers, used by Alembic.
revision = "v3.0.0.c"
down_revision = "v3.0.0.b"
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


FLOAT_PRECISION = 53


class FloatTypeEnum(enum.Enum):
    FINITE_OR_NAN = 1
    INF_POS = 2
    INF_NEG = 3


def _float_without_nan_to_stored_repr(value: float) -> Tuple[float, FloatTypeEnum]:
    if np.isposinf(value):
        return (0.0, FloatTypeEnum.INF_POS)
    elif np.isneginf(value):
        return (0.0, FloatTypeEnum.INF_NEG)
    else:
        return (value, FloatTypeEnum.FINITE_OR_NAN)


def _float_with_nan_to_stored_repr(value: float) -> Tuple[Optional[float], FloatTypeEnum]:
    if np.isnan(value):
        return (None, FloatTypeEnum.FINITE_OR_NAN)
    else:
        return _float_without_nan_to_stored_repr(value)


class IntermediateValueModel(BaseModel):
    __tablename__ = "trial_intermediate_values"
    trial_intermediate_value_id = sa.Column(sa.Integer, primary_key=True)
    intermediate_value = sa.Column(sa.Float(precision=FLOAT_PRECISION), nullable=True)
    intermediate_value_type = sa.Column(sa.Enum(FloatTypeEnum), nullable=False)


def upgrade():
    bind = op.get_bind()

    sa.Enum(FloatTypeEnum).create(bind, checkfirst=True)

    # MySQL and PostgreSQL supports DEFAULT clause like 'ALTER TABLE <tbl_name>
    # ADD COLUMN <col_name> ... DEFAULT "FINITE_OR_NAN"', but seemingly Alembic
    # does not support such a SQL statement. So first add a column with schema-level
    # default value setting, then remove it by `batch_op.alter_column()`.
    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.add_column(
            sa.Column(
                "intermediate_value_type",
                sa.Enum("FINITE_OR_NAN", "INF_POS", "INF_NEG", name="floattypeenum"),
                nullable=False,
                server_default="FINITE_OR_NAN",
            ),
        )
    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.alter_column("intermediate_value_type", server_default=None)

    session = orm.Session(bind=bind)
    try:
        records = session.query(IntermediateValueModel).all()
        mapping = []
        for r in records:
            value: float
            if np.isclose(r.intermediate_value, RDB_MAX_FLOAT) or np.isposinf(
                r.intermediate_value
            ):
                value = np.inf
            elif np.isclose(r.intermediate_value, RDB_MIN_FLOAT) or np.isneginf(
                r.intermediate_value
            ):
                value = -np.inf
            elif np.isnan(r.intermediate_value):
                value = np.nan
            else:
                value = r.intermediate_value
            (sanitized_value, float_type) = _float_with_nan_to_stored_repr(value)
            mapping.append(
                {
                    "trial_intermediate_value_id": r.trial_intermediate_value_id,
                    "intermediate_value_type": float_type,
                    "intermediate_value": sanitized_value,
                }
            )
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def downgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    try:
        records = session.query(IntermediateValueModel).all()
        mapping = []
        for r in records:
            if r.intermediate_value_type == FloatTypeEnum.FINITE_OR_NAN:
                continue

            _intermediate_value = r.intermediate_value
            if r.intermediate_value_type == FloatTypeEnum.INF_POS:
                _intermediate_value = RDB_MAX_FLOAT
            else:
                _intermediate_value = RDB_MIN_FLOAT

            mapping.append(
                {
                    "trial_intermediate_value_id": r.trial_intermediate_value_id,
                    "intermediate_value": _intermediate_value,
                }
            )
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("trial_intermediate_values", schema=None) as batch_op:
        batch_op.drop_column("intermediate_value_type")

    sa.Enum(IntermediateValueModel.FloatTypeEnum).drop(bind, checkfirst=True)

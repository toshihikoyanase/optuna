"""unify existing distributions to {int,float} distribution

Revision ID: v3.0.0.a
Revises: v2.6.0.a
Create Date: 2021-11-21 23:48:42.424430

"""
from typing import Any

from alembic import op
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import Text
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import UniqueConstraint
from sqlalchemy import orm
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.declarative import declarative_base

from optuna.distributions import (
    BaseDistribution,
    DiscreteUniformDistribution,
    FloatDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    UniformDistribution,
)
from optuna.distributions import IntDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.trial import TrialState


MAX_INDEXED_STRING_LENGTH = 512


BaseModel = declarative_base()


class StudyModel(BaseModel):
    __tablename__ = "studies"
    study_id = Column(Integer, primary_key=True)
    study_name = Column(String(MAX_INDEXED_STRING_LENGTH), index=True, unique=True, nullable=False)


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = Column(Integer, primary_key=True)
    # No `UniqueConstraint` is put on the `number` columns although it in practice is constrained
    # to be unique. This is to reduce code complexity as table-level locking would be required
    # otherwise. See https://github.com/optuna/optuna/pull/939#discussion_r387447632.
    number = Column(Integer)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    state = Column(Enum(TrialState), nullable=False)
    datetime_start = Column(DateTime)
    datetime_complete = Column(DateTime)

    study = orm.relationship(
        StudyModel, backref=orm.backref("trials", cascade="all, delete-orphan")
    )


class TrialParamModel(BaseModel):
    __tablename__ = "trial_params"
    __table_args__: Any = (UniqueConstraint("trial_id", "param_name"),)
    param_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"))
    param_name = Column(String(MAX_INDEXED_STRING_LENGTH))
    param_value = Column(Float)
    distribution_json = Column(Text())

    trial = orm.relationship(
        TrialModel, backref=orm.backref("params", cascade="all, delete-orphan")
    )


# revision identifiers, used by Alembic.
revision = "v3.0.0.a"
down_revision = "v2.6.0.a"
branch_labels = None
depends_on = None


def migrate_new_distribution(distribution_json: str) -> str:
    distribution = json_to_distribution(distribution_json)
    new_distribution: BaseDistribution

    # float distributions
    if isinstance(distribution, UniformDistribution):
        new_distribution = FloatDistribution(
            low=distribution.low,
            high=distribution.high,
            log=False,
            step=None,
        )
    elif isinstance(distribution, LogUniformDistribution):
        new_distribution = FloatDistribution(
            low=distribution.low,
            high=distribution.high,
            log=True,
            step=None,
        )
    elif isinstance(distribution, DiscreteUniformDistribution):
        new_distribution = FloatDistribution(
            low=distribution.low,
            high=distribution.high,
            log=False,
            step=distribution.q,
        )

    # int distributions
    elif isinstance(distribution, IntUniformDistribution):
        new_distribution = IntDistribution(
            low=distribution.low,
            high=distribution.high,
            log=False,
            step=distribution.step,
        )
    elif isinstance(distribution, IntLogUniformDistribution):
        new_distribution = IntDistribution(
            low=distribution.low,
            high=distribution.high,
            log=True,
            step=distribution.step,
        )

    # categorical distribution
    else:
        new_distribution = distribution

    return distribution_to_json(new_distribution)


def restore_old_distribution(distribution_json: str) -> str:
    distribution = json_to_distribution(distribution_json)
    old_distribution: BaseDistribution

    # float distributions
    if isinstance(distribution, FloatDistribution):
        if distribution.log:
            old_distribution = LogUniformDistribution(
                low=distribution.low,
                high=distribution.high,
            )
        else:
            if distribution.step is not None:
                old_distribution = DiscreteUniformDistribution(
                    low=distribution.low,
                    high=distribution.high,
                    q=distribution.step,
                )
            else:
                old_distribution = UniformDistribution(
                    low=distribution.low,
                    high=distribution.high,
                )

    # integer distributions
    elif isinstance(distribution, IntDistribution):
        if distribution.log:
            old_distribution = IntLogUniformDistribution(
                low=distribution.low,
                high=distribution.high,
                step=distribution.step,
            )
        else:
            old_distribution = IntUniformDistribution(
                low=distribution.low,
                high=distribution.high,
                step=distribution.step,
            )

    # categorical distribution
    else:
        old_distribution = distribution

    return distribution_to_json(old_distribution)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    tables = inspector.get_table_names()

    assert "trial_params" in tables

    session = orm.Session(bind=bind)
    try:
        distributions = session.query(TrialParamModel).all()
        for distribution in distributions:
            distribution.distribution_json = migrate_new_distribution(
                distribution.distribution_json,
            )
        session.bulk_save_objects(distributions)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def downgrade() -> None:
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)
    tables = inspector.get_table_names()

    assert "trial_params" in tables

    session = orm.Session(bind=bind)
    try:
        distributions = session.query(TrialParamModel).all()
        for distribution in distributions:
            distribution.distribution_json = restore_old_distribution(
                distribution.distribution_json,
            )
        session.bulk_save_objects(distributions)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

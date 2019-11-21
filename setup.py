import os
import pkg_resources
from setuptools import find_packages
from setuptools import setup
import sys

from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA

from pkg_resources import Distribution  # NOQA


def get_version():
    # type: () -> str

    version_filepath = os.path.join(os.path.dirname(__file__), 'optuna', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]
    assert False


def get_long_description():
    # type: () -> str

    readme_filepath = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_filepath) as f:
        return f.read()


def get_install_requires():
    # type: () -> List[str]

    install_requires = [
        'alembic', 'cliff', 'colorlog', 'numpy', 'scipy',
        'sqlalchemy>=1.1.0', 'tqdm', 'typing', 'joblib'
    ]
    if sys.version_info[0] == 2:
        install_requires.extend(['enum34'])
    return install_requires


def get_extras_require():
    # type: () -> Dict[str, List[str]]

    testing_requirements = [
        'bokeh', 'chainer>=5.0.0', 'cma', 'keras', 'lightgbm', 'mock',
        'mpi4py', 'mxnet', 'pandas', 'plotly>=4.0.0', 'pytest', 'scikit-optimize',
        'tensorflow', 'tensorflow-datasets', 'xgboost', 'scikit-learn>=0.19.0',
        'torch', 'torchvision', 'pytorch-ignite', 'pytorch-lightning',
    ]

    example_requirements = [
        'chainer', 'keras', 'catboost', 'lightgbm', 'scikit-learn',
        'mxnet', 'xgboost', 'torch', 'torchvision', 'pytorch-ignite',
        'dask-ml', 'dask[dataframe]', 'pytorch-lightning',

        # TODO(Yanase): Update examples to support TensorFlow 2.0.
        # See https://github.com/optuna/optuna/issues/565 for further details.
        'tensorflow<2.0.0',
    ]

    if sys.version_info[:2] > (3, 5,):
        testing_requirements.append("fastai<2")
        example_requirements.append("fastai<2")

    extras_require = {
        'checking': ['autopep8', 'hacking', 'mypy'],
        'testing': testing_requirements,
        'example': example_requirements,
        'document': ['sphinx', 'sphinx_rtd_theme'],
        'codecov': ['pytest-cov', 'codecov'],
    }
    return extras_require


def find_any_distribution(pkgs):
    # type: (List[str]) -> Optional[Distribution]

    for pkg in pkgs:
        try:
            return pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return None


pfnopt_pkg = find_any_distribution(['pfnopt'])
if pfnopt_pkg is not None:
    msg = 'We detected that PFNOpt is installed in your environment.\n' \
        'PFNOpt has been renamed Optuna. Please uninstall the old\n' \
        'PFNOpt in advance (e.g. by executing `$ pip uninstall pfnopt`).'
    print(msg)
    exit(1)

setup(
    name='optuna',
    version=get_version(),
    description='A hyperparameter optimization framework',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    url='https://optuna.org/',
    packages=find_packages(),
    package_data={
        'optuna': [
            'storages/rdb/alembic.ini',
            'storages/rdb/alembic/*.*',
            'storages/rdb/alembic/versions/*.*'
        ]
    },
    install_requires=get_install_requires(),
    tests_require=get_extras_require()['testing'],
    extras_require=get_extras_require(),
    entry_points={'console_scripts': ['optuna = optuna.cli:main']})

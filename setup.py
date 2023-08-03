from setuptools import setup, find_packages

setup(
    name='titanic-ml-package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'click',
        'pandas',
        'scikit-learn',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'titanic-ml = titanic_ml.cli:cli'
        ],
    },
)

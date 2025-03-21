from setuptools import setup, find_packages

setup(
    name='matsign',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        'scikit-learn',
        'requests',
        'transformers',
        'datasets',
        'accelerate'
    ],
)
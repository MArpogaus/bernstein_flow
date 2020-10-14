from setuptools import setup, find_packages

setup(
    name="bernstein_flow",
    version="0.0.1",
    packages=find_packages('src'),
    package_dir={"": "src"},
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'tensorflow',
        'tensorflow_probability',
    ]
)

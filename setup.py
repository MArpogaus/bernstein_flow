from setuptools import setup, find_packages

setup(
    name="bernstein_flow",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.5.*",
        "tensorflow_probability==0.13.*",
        "scipy==1.7.*",
        "matplotlib",
        "seaborn",
    ],
)

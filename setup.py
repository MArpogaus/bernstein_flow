from setuptools import setup, find_packages

setup(
    name="bernstein_flow",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.4.1",
        "tensorflow_probability==0.12.2",
        "matplotlib==3.3.4",
        "seaborn==0.11.1",
        "numpy==1.19.3",
        "scipy==1.4.1",
    ],
)

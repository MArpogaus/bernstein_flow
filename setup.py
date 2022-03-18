from setuptools import find_packages, setup

setup(
    name="bernstein_flow",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.8.*",
        "tensorflow_probability==0.16.*",
        "scipy==1.7.*",
        "matplotlib",
        "seaborn",
    ],
)

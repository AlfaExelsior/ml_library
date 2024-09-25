from setuptools import setup, find_packages

setup(
    name="ml_library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scikit-learn"],
    author="Alfa Exelsior",
    description="A simple machine learning library",
    license="MIT",
    long_description=open('README.md').read(),
)

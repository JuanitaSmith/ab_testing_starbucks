from setuptools import setup, find_packages

description = """
A demonstration of how to write code for ML development
"""

setup(
    name = "ml_service",
    version = "0.0.1",
    author = "Damien Benveniste",
    author_email = "damien@mail.com",
    description = description,
    license = "BSD",
    url = "http://mywiki",
    packages=find_packages()
)

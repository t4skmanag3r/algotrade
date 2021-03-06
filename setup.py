from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = "0.1.3"
DESCRIPTION = "Algorithmic trading strategy creation and testing"
LONG_DESCRIPTION = (
    "A package that makes creating and testing algorithmic trading strategies simple"
)

# Setting up
setup(
    name="algotrade",
    version=VERSION,
    author="Edvinas Adomaitis",
    author_email="<edvinasad7@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["yahoo_fin", "pandas", "numpy", "matplotlib", "ta"],
    keywords=["python", "stocks", "algorithmic trading", "technical analysis"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

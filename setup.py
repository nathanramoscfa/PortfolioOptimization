from setuptools import setup, find_packages

# Read in requirements.txt for other dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Add the custom dependency for the forked yahooquery
requirements.append("yahooquery @ git+https://github.com/nathanramoscfa/yahooquery.git")

setup(
    name="PortfolioOptimization",
    version="0.0.10",
    author="Nathan Ramos, CFA®",
    author_email="nathan.ramos.github@gmail.com",
    description="A package for portfolio optimization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nathanramoscfa/PortfolioOptimization.git",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],
    python_requires='>=3.8',
)

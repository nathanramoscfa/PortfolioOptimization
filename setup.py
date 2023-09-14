from setuptools import setup, find_packages

# Read in requirements.txt for standard packages only (not git URLs)
with open("requirements.txt") as f:
    all_requirements = f.read().splitlines()
    standard_requirements = [req for req in all_requirements if not req.startswith("git+")]

# Git URL-based packages
git_requirements = [
    "git+https://github.com/nathanramoscfa/bt.git",
    "git+https://github.com/nathanramoscfa/ffn.git"
]

setup(
    name="PortfolioOptimization",
    version="0.0.6",
    author="Nathan Ramos",
    author_email="nathan.ramos.github@gmail.com",
    description="A package for portfolio optimization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nathanramoscfa/PortfolioOptimization.git",
    packages=find_packages(),
    install_requires=standard_requirements + git_requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.8',
)

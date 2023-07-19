from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="PyFunctional",
    description="Package for creating data pipelines with chain functional programming",
    long_description=Path("README.md").read_text(),
    url="https://github.com/EntilZha/PyFunctional",
    author="Pedro Rodriguez",
    author_email="me@pedro.ai",
    maintainer="Pedro Rodriguez",
    maintainer_email="me@pedro.ai",
    license="MIT",
    keywords="functional pipeline data collection chain rdd linq parallel",
    packages=find_packages(exclude=["contrib", "docs", "tests*", "test"]),
    version="1.4.3",
    install_requires=["dill>=0.2.5", "tabulate<=1.0.0"],
    extras_requires={
        "all": ["pandas"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

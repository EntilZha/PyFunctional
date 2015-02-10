from setuptools import setup, find_packages

setup(
    name='ScalaFunctional',
    description="Scala functional programming style in Python",
    url="https://github.com/EntilZha/ScalaFunctional",
    author="Pedro Rodriguez",
    author_email="pedro@snowgeek.org",
    license="MIT",
    keywords="functional scala",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'test']),
    version="0.0.3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7"
    ]
)
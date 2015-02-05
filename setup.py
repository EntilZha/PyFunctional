from setuptools import setup, find_packages

setup(
    name='ScalaFunctional',
    description="Scala functional programming style in Python",
    url="https://github.com/EntilZha/ScalaFunctional",
    author="Pedro Rodriguez",
    author_email="pedro@snowgeek.org",
    license="MIT",
    keywords="functional scala",
    package_data=find_packages(exclude=['contrib', 'docs', 'tests*', 'test']),
    version="0.0.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: 2.7"
    ]
)
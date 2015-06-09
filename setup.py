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
    version="0.3.0",
    install_requires=['enum34', 'future', 'six'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)

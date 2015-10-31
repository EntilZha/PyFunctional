from setuptools import setup, find_packages
import sys

requirements = ['future', 'six']
if not (sys.version_info.major == 3 and sys.version_info.minor >= 4):
    requirements.append('enum34')


setup(
    name='ScalaFunctional',
    description="Scala functional programming style in Python",
    url="https://github.com/EntilZha/ScalaFunctional",
    author="Pedro Rodriguez",
    author_email="ski.rodriguez@gmail.com",
    license="MIT",
    keywords="functional scala",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'test']),
    version="0.4.0",
    install_requires=requirements,
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

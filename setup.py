from setuptools import setup, find_packages

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', extra_args=['--columns=300'])
except (IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='PyFunctional',
    description='Package for creating data pipelines, LINQ, and chain functional programming',
    long_description=long_description,
    url='https://github.com/EntilZha/PyFunctional',
    author='Pedro Rodriguez',
    author_email='ski.rodriguez@gmail.com',
    maintainer='Pedro Rodriguez',
    maintainer_email='ski.rodriguez@gmail.com',
    license='MIT',
    keywords='functional LINQ pipeline data collection rdd scala',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'test']),
    version='0.6.0',
    install_requires=['future', 'six'],
    extras_requires={
        'all': ['pandas']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)

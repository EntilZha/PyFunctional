import sys
from setuptools import setup, find_packages

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', extra_args=['--columns=300'])
except (IOError, ImportError):
    long_description = open('README.md').read()

if sys.version_info.major == 2 or '__pypy__' in sys.builtin_module_names:
    install_requires = ['future==0.15.2', 'six==1.10.0', 'dill==0.2.5', 'bz2file==0.98',
                        'backports.lzma==0.0.3']
else:
    install_requires = ['future==0.15.2', 'six==1.10.0', 'dill==0.2.5']

setup(
    name='PyFunctional',
    description='Package for creating data pipelines with chain functional programming',
    long_description=long_description,
    url='https://github.com/EntilZha/PyFunctional',
    author='Pedro Rodriguez',
    author_email='ski.rodriguez@gmail.com',
    maintainer='Pedro Rodriguez',
    maintainer_email='ski.rodriguez@gmail.com',
    license='MIT',
    keywords='functional pipeline data collection chain rdd linq parallel',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'test']),
    version='0.7.1',
    install_requires=install_requires,
    extras_requires={
        'all': ['pandas']
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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

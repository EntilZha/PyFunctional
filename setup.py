import sys
from setuptools import setup, find_packages

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', extra_args=['--columns=300'])
except (IOError, ImportError):
    long_description = open('README.md').read()

common_install_requires = ['future<=1.0.0', 'six<=2.0.0', 'dill>=0.2.6,<=0.2.7.1', 'tabulate<=1.0.0']
if sys.version_info.major == 2 or '__pypy__' in sys.builtin_module_names:
    compression_requires = ['bz2file==0.98', 'backports.lzma==0.0.6']
    install_requires = common_install_requires
else:
    compression_requires = []
    install_requires = common_install_requires

setup(
    name='PyFunctional',
    description='Package for creating data pipelines with chain functional programming',
    long_description=long_description,
    url='https://github.com/EntilZha/PyFunctional',
    author='Pedro Rodriguez',
    author_email='me@pedro.ai',
    maintainer='Pedro Rodriguez',
    maintainer_email='me@pedro.ai',
    license='MIT',
    keywords='functional pipeline data collection chain rdd linq parallel',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'test']),
    version='1.3.0',
    install_requires=install_requires,
    extras_requires={
        'all': ['pandas'] + compression_requires,
        'compression': compression_requires
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)

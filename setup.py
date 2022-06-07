from setuptools import setup, find_packages
setup(
    name = 'TimeSeriesDataAnalysisLib',
    version = '0.0.1-a1',
    keywords='',
    description = 'A workflow framework for ml time deries fata analysis project',
    license = '',
    url = '',
    author = 'plasticanne',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.fbs","*.md"],"doc": ["../doc/*"]
    },
    install_requires=[
        'ruptures',
        'mlflow==1.7.0',
        'pandas>=1.0.5',
        'aenum',
        'jsonlines',
        'attrs',
        'cattrs'
    ],
    classifiers = [
        # Development Status :: 1 - Planning
        # Development Status :: 2 - Pre-Alpha
        # Development Status :: 3 - Alpha
        # Development Status :: 4 - Beta
        # Development Status :: 5 - Production/State3
        # Development Status :: 6 - Mature
        # Development Status :: 7 - Inactive
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)


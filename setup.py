"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

Built using this blog post as well:
https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
https://tom-christie.github.io/articles/pypi/
http://stackoverflow.com/questions/11848030/how-include-static-files-to-setuptools-python-package
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='auto_ml',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.4',

    description='Automated machine learning for production and analytics',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/ClimbsRocks/auto_ml',

    # Author details
    author='Preston Parry',
    author_email='ClimbsBytes@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords=['machine learning', 'data science', 'automated machine learning', 'regressor', 'regressors', 'regression', 'classification', 'classifiers', 'classifier', 'estimators', 'predictors', 'XGBoost', 'Random Forest', 'sklearn', 'scikit-learn', 'analytics', 'analysts', 'coefficients', 'feature importances' 'analytics', 'artificial intelligence', 'subpredictors', 'ensembling', 'stacking', 'blending', 'feature engineering', 'feature extraction', 'feature selection', 'production', 'pandas', 'dataframes', 'machinejs'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['auto_ml'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['scikit-learn', 'xgboost', 'scipy', 'pandas', 'pathos', 'python-dateutil'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # include_package_data=True,
    # package_data={
    #     'corpora': ['corpora/aggregatedCorpusCleanedAndFiltered.csv']
    # }

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)

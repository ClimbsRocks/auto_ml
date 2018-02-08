from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
try:
    # Try to format our PyPi page as rst so it displays properly
    import pypandoc
    with open ('README.md', 'rb') as read_file:
        readme_text = read_file.readlines()
    # Change our README for pypi so we can get analytics tracking information for that separately
    readme_text = [row.decode() for row in readme_text]
    readme_text[-1] = "[![Analytics](https://ga-beacon.appspot.com/UA-58170643-5/auto_ml/pypi)](https://github.com/igrigorik/ga-beacon)"

    long_description = pypandoc.convert(''.join(readme_text), 'rst', format='md')
except ImportError:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('pypandoc (and possibly pandoc) are not installed. This means the PyPi package info will be formatted as .md instead of .rst. If you are encountering this before uploading a PyPi distribution, please install these')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Get the long description from the README file
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='auto_ml',

    version=open("auto_ml/_version.py").readlines()[-1].split()[-1].strip("\"'"),

    description='Automated machine learning for production and analytics',
    long_description=long_description,

    url='https://github.com/ClimbsRocks/auto_ml',

    author='Preston Parry',
    author_email='ClimbsBytes@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6',
    ],

    keywords=['machine learning', 'data science', 'automated machine learning', 'regressor', 'regressors', 'regression', 'classification', 'classifiers', 'classifier', 'estimators', 'predictors', 'XGBoost', 'Random Forest', 'sklearn', 'scikit-learn', 'analytics', 'analysts', 'coefficients', 'feature importances' 'analytics', 'artificial intelligence', 'subpredictors', 'ensembling', 'stacking', 'blending', 'feature engineering', 'feature extraction', 'feature selection', 'production', 'pandas', 'dataframes', 'machinejs', 'deep learning', 'tensorflow', 'deeplearning', 'lightgbm', 'gradient boosting', 'gbm', 'keras', 'production ready', 'test coverage'],

    packages=['auto_ml'],

    # We will allow the user to install XGBoost themselves. However, since it can be difficult to install, we will not force them to go through that install challenge if they're just checking out the package and want to get running with it quickly.
    install_requires=[
        'dill>=0.2.5, <0.3',
        'h5py>=2.7.0, <3.0',
        'lightgbm>=2.0.11, <2.1',
        'numpy>=1.11.0, <2.0',
        'pandas>=0.18.0, <1.0',
        'pathos>=0.2.1, <0.3.0',
        'python-dateutil>=2.6.1, <3.0',
        'scikit-learn>=0.18.1, <1.0',
        'scipy>=0.14.0, <2.0',
        'sklearn-deap2>=0.2.1, <0.3',
        'tabulate>=0.7.5, <1.0',
    ],

    test_suite='nose.collector',
    tests_require=['nose', 'coveralls']
)

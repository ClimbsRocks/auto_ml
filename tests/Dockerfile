FROM ubuntu

    # Use a fixed apt-get repo to stop intermittent failures due to flaky httpredir connections,
    # as described by Lionel Chan at http://stackoverflow.com/a/37426929/5881346

RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
    apt-get update && apt-get install -y build-essential python-pip apt-utils git cmake && \
    pip install pandas numpy scipy && \
    cd /usr/local/src && \
    pip install tensorflow && \

    cd /usr/local/src && mkdir xgboost && cd xgboost && \
    git clone --depth 1 --recursive https://github.com/dmlc/xgboost.git && cd xgboost && \
    make && cd python-package && python setup.py install && \

    apt-get autoremove -y && apt-get clean

RUN pip install --upgrade pip
RUN pip install --upgrade numpy dill h5py scikit-learn scipy python-dateutil pandas pathos keras coveralls nose lightgbm tabulate imblearn sklearn-deap2 catboost

# To update this image and upload it:
# Build the image (docker build .), and give it two separate tags (latest, and a version number)
# docker build --no-cache -t climbsrocks/auto_ml_tests:0.0.10 -t climbsrocks/auto_ml_tests:latest .
# docker push climbsrocks/auto_ml_tests:latest
# docker push climbsrocks/auto_ml_tests:0.0.10

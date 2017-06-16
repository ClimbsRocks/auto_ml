FROM ubuntu

    # Use a fixed apt-get repo to stop intermittent failures due to flaky httpredir connections,
    # as described by Lionel Chan at http://stackoverflow.com/a/37426929/5881346
RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
    apt-get update && apt-get install -y build-essential python-pip apt-utils git cmake && \
    pip install pandas numpy scipy && \
    cd /usr/local/src && \
    pip install tensorflow && \

    # Microsoft's LightGBM
    cd /usr/local/src && git clone --recursive --depth 1 https://github.com/Microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && cmake .. && make -j $(nproc) && \
    cd /usr/local/src/LightGBM/python-package && python setup.py install && \

    # XGBoost
    cd /usr/local/src && mkdir xgboost && cd xgboost && \
    git clone --depth 1 --recursive https://github.com/dmlc/xgboost.git && cd xgboost && \
    make && cd python-package && python setup.py install && \

    apt-get autoremove -y && apt-get clean

RUN pip install --upgrade pip
RUN pip install --upgrade numpy dill h5py scikit-learn scipy python-dateutil pandas pathos keras coveralls nose tabulate sklearn-deap

# To update this image and upload it:
# docker build -t testdocker .
# docker tag testdocker climbsrocks/auto_ml_tests:0.0.5
# docker push climbsrocks/auto_ml_tests:0.0.5

# Modified version of scikit-learn's DictVectorizer
from array import array
import numbers
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six

from auto_ml.utils import CustomLabelEncoder

bad_vals = set([float('nan'), float('inf'), float('-inf'), None, np.nan, 'None', 'none', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'inf', '-inf'])

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


class DataFrameVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, column_descriptions=None, dtype=np.float32, separator="=", sparse=True, sort=True, keep_cat_features=False):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        if column_descriptions == None:
            column_descriptions = {}
        self.column_descriptions = column_descriptions
        self.vals_to_drop = set(['ignore', 'output', 'regressor', 'classifier'])
        self.has_been_restricted = False
        self.keep_cat_features = keep_cat_features
        self.label_encoders = {}


    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default

    def fit(self, X, y=None):
        print('Fitting DataFrameVectorizer')

        feature_names = []
        vocab = {}

        for col_name in X.columns:
            # Ignore 'ignore', 'output', etc.
            if self.column_descriptions.get(col_name, False) not in self.vals_to_drop:

                if self.column_descriptions.get(col_name, False) == 'categorical' and self.keep_cat_features == True:
                    # All of these values will go in the same column, but they must be turned into ints first
                    self.label_encoders[col_name] = CustomLabelEncoder()
                    # Then, we will use the same flow below to make sure they appear in the vocab correctly
                    self.label_encoders[col_name].fit(X[col_name])


                if self.column_descriptions.get(col_name, False) == 'categorical' and self.keep_cat_features == False:
                    # If this is a categorical column, iterate through each row to get all the possible values that we are one-hot-encoding.
                    for val in X[col_name]:
                        if not isinstance(val, str):
                            if isinstance(val, numbers.Number) or val is None:
                                val = str(val)
                            else:
                                val = val.encode('utf-8').decode('utf-8')

                        feature_name = col_name + self.separator + val

                        if feature_name not in vocab:
                            feature_names.append(feature_name)
                            vocab[feature_name] = len(vocab)
                # Ideally we shouldn't have to check for for duplicate columns, but in case we're passed a DataFrame with duplicate columns, consolidate down to a single column. Maybe not the ideal solution, but solves a class of bugs, and puts the reasonable onus on the user to not pass in two data columns with different meanings but the same column name
                # And of course, if this is a categorical column, do not include the column name itself, just include the feature_names as calculated above
                elif col_name not in vocab:
                    feature_names.append(col_name)
                    vocab[col_name] = len(vocab)

        if self.sort:
            feature_names.sort()
            vocab = dict((f, i) for i, f in enumerate(feature_names))

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab
        # del self.column_descriptions
        return self

    def _transform(self, X):

        dtype = self.dtype
        feature_names = self.feature_names_
        vocab = self.vocabulary_

        # Process everything as sparse regardless of setting

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []


        if isinstance(X, dict):
            for f, val in X.items():
                if self.column_descriptions.get(f, False) == 'categorical':
                    if self.get('keep_cat_features', False) == False:
                        if not isinstance(val, str):
                            if isinstance(val, numbers.Number) or val is None:
                                val = str(val)
                            else:
                                val = val.encode('utf-8').decode('utf-8')
                        f = f + self.separator + val
                        val = 1
                    else:
                        if val in bad_vals:
                            val = '_None'
                        val = self.get('label_encoders')[f].transform([val])

                if f in vocab and val not in bad_vals and (self.get('keep_cat_features', False) or not np.isnan(val)):

                    indices.append(vocab[f])
                    # Convert the val to the correct dtype, then append to our values list
                    values.append(dtype(val))

            indptr.append(len(indices))

            if len(indptr) == 1:
                raise ValueError('The dictionary passed into DataFrameVectorizer is empty')


        else:
            # collect all the possible feature names and build sparse matrix at
            # same time
            X_columns = list(X.columns)
            string_types = six.string_types
            separator = self.separator
            indices_append = indices.append
            values_append = values.append
            keep_cat_features = self.get('keep_cat_features', False)
            is_categorical = [self.column_descriptions.get(f, False) == 'categorical' for f in X_columns]
            X = X.values
            row_len = X.shape[1]
            range_row_len = range(row_len)

            for row_idx in range(X.shape[0]):
                for col_idx in range_row_len:
                    val = X[row_idx, col_idx]
                    f = X_columns[col_idx]

                    if is_categorical[col_idx]:
                        if keep_cat_features:
                            if val in bad_vals:
                                val = '_None'

                            val = self.get('label_encoders')[f].transform([val])

                        else:
                            if not isinstance(val, str):
                                if isinstance(val, numbers.Number) or val is None:
                                    val = str(val)
                                else:
                                    val = val.encode('utf-8').decode('utf-8')
                            f = f + separator + val
                            val = 1

                    # Only include this in our output if it was part of our training data. Silently ignore it otherwise.
                    if f in vocab and val not in bad_vals and (self.get('keep_cat_features', False) or not np.isnan(val)):
                        # Get the index position from vocab, then append that index position to indices
                        indices_append(vocab[f])
                        # Convert the val to the correct dtype, then append to our values list
                        values_append(dtype(val))

                indptr.append(len(indices))

            if len(indptr) == 1:
                raise ValueError('The DataFrame passed into DataFrameVectorizer is empty')

        indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        return result_matrix


    def transform(self, X, y=None):
        return self._transform(X)

    def get_feature_names(self):
        """Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        """
        return self.feature_names_

    def restrict(self, support, indices=False):
        """Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        """
        if self.has_been_restricted == True:
            return self

        if not indices:
            support = np.where(support)[0]

        names = self.feature_names_
        new_vocab = {}
        for i in support:
            new_vocab[names[i]] = len(new_vocab)

        self.vocabulary_ = new_vocab
        self.feature_names_ = [f for f, i in sorted(six.iteritems(new_vocab),
                                                    key=itemgetter(1))]

        self.has_been_restricted = True

        return self

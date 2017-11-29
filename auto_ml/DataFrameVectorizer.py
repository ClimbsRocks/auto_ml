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

    def __init__(self, column_descriptions=None, dtype=np.float32, separator="=", sparse=True, keep_cat_features=False):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        if column_descriptions == None:
            column_descriptions = {}
        self.column_descriptions = column_descriptions
        self.vals_to_drop = set(['ignore', 'output', 'regressor', 'classifier'])
        self.has_been_restricted = False
        self.keep_cat_features = keep_cat_features
        self.label_encoders = {}
        self.numerical_columns = None
        self.num_numerical_cols = None
        self.categorical_columns = None
        self.numeric_col_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.additional_numerical_cols = []



    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default

    def fit(self, X, y=None):
        print('Fitting DataFrameVectorizer')

        feature_names = []
        vocab = {}

        # TODO: rearrange X so that all the categorical columns are first
        # Then we'll have to write a bit of our own custom logic to create feature_names_ and vocabulary_ (the values that thigns like get_feature_names and .restrict depend upon)


        numerical_columns = []
        categorical_columns = []
        for col in X.columns:
            col_desc = self.column_descriptions.get(col, False)
            if col_desc in [False, 'continuous', 'int', 'float', 'numerical']:
                numerical_columns.append(col)
            elif col_desc in self.vals_to_drop:
                continue
            elif col_desc == 'categorical':
                categorical_columns.append(col)
            else:
                print('We are unsure what to do with this column:')
                print(col)
                print(col_desc)

        self.num_numerical_cols = len(numerical_columns)
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        new_cols = numerical_columns + categorical_columns
        X = X[new_cols]

        for col_name in X.columns:

            if self.column_descriptions.get(col_name, False) == 'categorical' and self.keep_cat_features == True:
                # All of these values will go in the same column, but they must be turned into ints first
                self.label_encoders[col_name] = CustomLabelEncoder()
                # Then, we will use the same flow below to make sure they appear in the vocab correctly
                self.label_encoders[col_name].fit(X[col_name])


            # We can't do elif here- it has to be inclusive of the logic above
            if self.column_descriptions.get(col_name, False) == 'categorical' and self.keep_cat_features == False:
                # If this is a categorical column, iterate through each row to get all the possible values that we are one-hot-encoding.
                for val in set(X[col_name]):
                    if not isinstance(val, str):
                        if isinstance(val, numbers.Number) or val is None:
                            val = str(val)
                        else:
                            val = val.encode('utf-8').decode('utf-8')

                    feature_name = col_name + self.separator + val

                    if feature_name not in vocab:
                        feature_names.append(feature_name)
                        vocab[feature_name] = len(vocab)

            # If this is a categorical column, do not include the column name itself, just include the feature_names as calculated above
            elif col_name not in vocab:
                feature_names.append(col_name)
                vocab[col_name] = len(vocab)

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab
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

            indices = np.frombuffer(indices, dtype=np.intc)
            indptr = np.frombuffer(indptr, dtype=np.intc)
            shape = (len(indptr) - 1, len(vocab))

            result_matrix = sp.csr_matrix((values, indices, indptr),
                                          shape=shape, dtype=dtype)

            if self.sparse:
                result_matrix.sort_indices()

            return result_matrix



        else:

            for col in self.numerical_columns:
                if col not in X.columns:
                    X[col] = 0
            for col in self.categorical_columns:
                if col not in X.columns:
                    X[col] = 0
            for col in self.additional_numerical_cols:
                if col not in X.columns:
                    X[col] = 0

            # we might not need to do this. when we get the .values, we're already getting the cols in the right order before .values. and we'll write a function for the categorical transforms that takes proper column names into account

            # NEXT: write a parser for a single categorical column
                # then we'll run those in parallel
                # and eventually

            # X = X.fillna(0)
            # TODO: is there a more efficient way of doing this?
            for idx, col in enumerate(self.numerical_columns):
                if X[col].dtype not in self.numeric_col_types:
                    X[col] = X[col].astype(float)
            numerical_vals = X[self.numerical_columns]
            # print('self.numerical_columns')
            # print(self.numerical_columns)

            numerical_vals = sp.csr_matrix(numerical_vals)


            # for val in self.categorical_columns:
            #     print('val')
            #     print(val)
            #     print('X[val]')
            #     print(X[val])
            #     self.transform_categorical_col(col_vals=X[val], col_name=val)

            categorical_vals = map(lambda col_name: self.transform_categorical_col(col_vals=list(X[col_name]), col_name=col_name), self.categorical_columns)

            final_result = numerical_vals

            for result in categorical_vals:
                # print('final_result.shape')
                # print(final_result.shape)
                # print('result.shape')
                # print(result.shape)
                final_result = sp.hstack((final_result, result), format='csr')
                # print('final_result.shape')
                # print(final_result.shape)

            additional_numerical_vals = X[self.additional_numerical_cols]
            additional_numerical_vals = sp.csr_matrix(additional_numerical_vals)
            final_result = sp.hstack((final_result, additional_numerical_vals), format='csr')
            # add in any missing numerical columns that we had at fitting time that we do not have now
            # TODO: get numerical columns in the exact same order as we had them at fitting time
            # now, our numerical vals are just X[[numerical_cols]].values
            # Then, we just have to transform our categorical cols
            # that, we should be able to do in parallel?
                # what we'll have to be careful of:
                # let's say a categorical col has 10 unique vals at fitting time
                # we need to make sure we've got it handled properly if it has only 9 unique vals at transform time (this will be pretty common)
                # an easy workaround is to create a dense matrix of 0's of the shape we'd expect from this column
                # then, for each value in the column, set the appropriate row/col combo to 1
                # then, at the end, we'll just hstack all the results together
                # we'll probably want to make everything sparse at some point along the way






            # collect all the possible feature names and build sparse matrix at
            # same time
        #     X_columns = list(X.columns)
        #     string_types = six.string_types
        #     separator = self.separator
        #     indices_append = indices.append
        #     values_append = values.append
        #     keep_cat_features = self.get('keep_cat_features', False)
        #     is_categorical = [self.column_descriptions.get(f, False) == 'categorical' for f in X_columns]
        #     X = X.values
        #     row_len = X.shape[1]
        #     range_row_len = range(row_len)

        #     for row_idx in range(X.shape[0]):
        #         for col_idx in range_row_len:
        #             val = X[row_idx, col_idx]
        #             f = X_columns[col_idx]

        #             if is_categorical[col_idx]:
        #                 if keep_cat_features:
        #                     if val in bad_vals:
        #                         val = '_None'

        #                     val = self.get('label_encoders')[f].transform([val])

        #                 else:
        #                     if not isinstance(val, str):
        #                         if isinstance(val, numbers.Number) or val is None:
        #                             val = str(val)
        #                         else:
        #                             val = val.encode('utf-8').decode('utf-8')
        #                     f = f + separator + val
        #                     val = 1

        #             # Only include this in our output if it was part of our training data. Silently ignore it otherwise.
        #             if f in vocab and val not in bad_vals and (self.get('keep_cat_features', False) or not np.isnan(val)):
        #                 # Get the index position from vocab, then append that index position to indices
        #                 indices_append(vocab[f])
        #                 # Convert the val to the correct dtype, then append to our values list
        #                 values_append(dtype(val))

        #         indptr.append(len(indices))

        #     if len(indptr) == 1:
        #         raise ValueError('The DataFrame passed into DataFrameVectorizer is empty')

        # indices = np.frombuffer(indices, dtype=np.intc)
        # indptr = np.frombuffer(indptr, dtype=np.intc)
        # shape = (len(indptr) - 1, len(vocab))

        # result_matrix = sp.csr_matrix((values, indices, indptr),
        #                               shape=shape, dtype=dtype)

        # if self.sparse:
        #     result_matrix.sort_indices()
        # else:
        #     result_matrix = result_matrix.toarray()

        # print('final_result.shape')
        # print(final_result.shape)
        # print('self.vocabulary_')
        # print(self.vocabulary_)
        print('final_result.shape at the end of DataFrameVectorizer _transform')
        print(final_result.shape)

        return final_result


    # We are assuming that each categorical column got a contiguous block of result columns (ie, the 5 categories in City get columns 5-9, not columns 0, 8, 26, 4, and 20)
    def transform_categorical_col(self, col_vals, col_name):
        if self.get('keep_cat_features', False) == True:
            return_vals = self.get('label_encoders')[col_name].transform(col_vals)

            # we will hstack these later with other sparse values
            # scipy.sparse.hstack expects each input to be a matrix, not a vector, so we are making a matrix where each row just has one column
            return_vals = [[val] for val in return_vals]
            return return_vals

        else:

            num_trained_cols = 0
            min_transformed_idx = None
            max_transformed_idx = None
            len_col_name = len(col_name)

            for trained_feature, col_idx in self.vocabulary_.items():
                if trained_feature[:len_col_name] == col_name:
                    num_trained_cols += 1
                    if min_transformed_idx is None:
                        min_transformed_idx = col_idx
                        max_transformed_idx = col_idx
                    elif col_idx > max_transformed_idx:
                        max_transformed_idx = col_idx
                    elif col_idx < min_transformed_idx:
                        min_transformed_idx = col_idx

            # print('col_name')
            # print(col_name)
            # print('num_trained_cols')
            # print(num_trained_cols)
            # print('min_transformed_idx')
            # print(min_transformed_idx)
            # print('max_transformed_idx')
            # print(max_transformed_idx)
            # print('len_col_name')
            # print(len_col_name)
            result = sp.lil_matrix((len(col_vals), num_trained_cols))

            if num_trained_cols == 0:
                return result

            if num_trained_cols != (max_transformed_idx - min_transformed_idx + 1):
                print('We have somehow ended up with categorical column behavior we were not expecting')
                raise(ValueError)


            for row_idx, val in enumerate(col_vals):
                if not isinstance(val, str):
                    if isinstance(val, numbers.Number) or val is None:
                        val = str(val)
                    else:
                        val = val.encode('utf-8').decode('utf-8')

                feature_name = col_name + self.separator + val
                if feature_name in self.vocabulary_:
                    col_idx = self.vocabulary_[feature_name]
                    col_idx = col_idx - min_transformed_idx

                    # TODO: get the appropriate column index/number
                    result[row_idx, col_idx] = 1


            result = sp.csr_matrix(result)
            return result

            # TODO: create a sparse matrix with the right width.
            # it is critical that we have all the same columns, in the same order, and that each of them take up their required amount of space
            # This also means we cannot change our labelencoder later

    def transform(self, X, y=None):
        return self._transform(X)

    def get_feature_names(self):
        """Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        """
        return self.feature_names_


    # This is for cases where we want to add in new features, such as for feature_learning
    def add_new_numerical_cols(self, new_feature_names):
        # add to our vocabulary
        for feature_name in new_feature_names:
            if feature_name not in self.vocabulary_:
                self.feature_names_.append(feature_name)
                self.vocabulary_[feature_name] = len(self.vocabulary_)
                self.additional_numerical_cols.append(feature_name)

        return self

    def restrict(self, support):
        """Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        """
        if self.has_been_restricted == True:
            return self

        print('self.feature_names_ at the start of restrict')
        print(self.feature_names_)
        print('self.numerical_columns at the start of restrict')
        print(self.numerical_columns)
        print('self.categorical_columns at the start of restrict')
        print(self.categorical_columns)
        print('self.vocabulary_ at the start of restrict')
        print(self.vocabulary_)
        print('self.additional_numerical_cols at the start of restrict')
        print(self.additional_numerical_cols)

        new_numerical_cols = []
        new_categorical_cols = []
        new_additional_numerical_cols = []
        new_feature_names = []
        new_vocab = {}

        for idx, val in enumerate(support):
            if val == True:
                feature_name = self.feature_names_[idx]
                if self.separator in feature_name:
                    base_feature_name = feature_name[:feature_name.rfind(self.separator)]
                else:
                    base_feature_name = feature_name
                new_feature_names.append(feature_name)
                new_vocab[feature_name] = len(new_vocab)
                if feature_name in self.numerical_columns:
                    new_numerical_cols.append(feature_name)
                elif base_feature_name in self.categorical_columns and base_feature_name not in new_categorical_cols:
                    new_categorical_cols.append(base_feature_name)
                elif feature_name in self.additional_numerical_cols:
                    new_additional_numerical_cols.append(feature_name)

        self.feature_names_ = new_feature_names
        self.vocabulary_ = new_vocab
        self.numerical_columns = new_numerical_cols
        self.categorical_columns = new_categorical_cols
        self.additional_numerical_cols = new_additional_numerical_cols

        self.has_been_restricted = True

        print('self.feature_names_ at the end of restrict')
        print(self.feature_names_)
        print('self.numerical_columns at the end of restrict')
        print(self.numerical_columns)
        print('self.categorical_columns at the end of restrict')
        print(self.categorical_columns)
        print('self.vocabulary_ at the end of restrict')
        print(self.vocabulary_)
        print('self.additional_numerical_cols at the end of restrict')
        print(self.additional_numerical_cols)
        return self

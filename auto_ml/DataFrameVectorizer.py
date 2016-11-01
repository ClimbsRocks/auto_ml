# Modified version of scikit-learn's DictVectorizer
from array import array
from collections import Mapping
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_array, tosequence
from sklearn.utils.fixes import frombuffer_empty

bad_vals_as_strings = set([str(float('nan')), str(float('inf')), str(float('-inf')), 'None', 'none', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'inf', '-inf'])


class DataFrameVectorizer(BaseEstimator, TransformerMixin):
    """Transforms a DataFrame to vectors.

    Just like scikit-learn's DictVectorizer, but adjusted to take a DataFrame as input, instead of a list of dictionaries.

    This transformer turns a DataFrame into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    However, note that this transformer will only do a binary one-hot encoding
    when feature values are of type string. If categorical features are
    represented as numeric values such as int, the DictVectorizer can be
    followed by OneHotEncoder to complete binary one-hot encoding.

    Features that do not occur in a row will have a zero value
    in the resulting array/matrix.

    Read more in the :ref:`User Guide <dict_feature_extraction>`.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator : string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse : boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort : boolean, optional.
        Whether ``feature_names_`` and ``vocabulary_`` should be sorted when fitting.
        True by default.

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    See also
    --------
    FeatureHasher : performs vectorization using only a hash function.
    sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features
      encoded as columns of integers.
    """

    def __init__(self, column_descriptions=None, dtype=np.float32, separator="=", sparse=True, sort=True):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        if column_descriptions == None:
            column_descriptions = {}
        self.column_descriptions = column_descriptions
        self.vals_to_drop = set(['ignore', 'output', 'regressor', 'classifier'])


    def fit(self, X, y=None):
        """Learn a list of column_name -> indices mappings.

        Parameters
        ----------
        X : DataFrame
        y : (ignored)

        Returns
        -------
        self
        """
        feature_names = []
        vocab = {}

        for col_name in X.columns:
            # Ignore 'ignore', 'output', etc.
            if self.column_descriptions.get(col_name, False) not in self.vals_to_drop:
                if X[col_name].dtype == 'object' or self.column_descriptions.get(col_name, False) == 'categorical':
                    # If this is a categorical column, or the dtype continues to be object, iterate through each row to get all the possible values that we are one-hot-encoding.
                    for val in X[col_name]:
                        feature_name = col_name + self.separator + str(val)
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
        # Sanity check: Python's array has no way of explicitly requesting the
        # signed 32-bit integers that scipy.sparse needs, so we use the next
        # best thing: typecode "i" (int). However, if that gives larger or
        # smaller integers than 32-bit ones, np.frombuffer screws up.
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug report")

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
                if isinstance(val, six.string_types):
                    f = f + self.separator + val
                    val = 1

                if f in vocab and str(val) not in bad_vals_as_strings:
                    # Get the index position from vocab, then append that index position to indices
                    indices.append(vocab[f])
                    # Convert the val to the correct dtype, then append to our values list
                    values.append(dtype(val))

            indptr.append(len(indices))

            if len(indptr) == 1:
                raise ValueError('The dictionary passed into DataFrameVectorizer is empty')


        else:
            # collect all the possible feature names and build sparse matrix at
            # same time
            for row_idx, row in X.iterrows():
                for col_idx, val in enumerate(row):
                    f = X.columns[col_idx]

                    if isinstance(val, six.string_types):
                        f = f + self.separator + val
                        val = 1

                    # Only include this in our output if it was part of our training data. Silently ignore it otherwise.
                    if f in vocab and str(val) not in bad_vals_as_strings:
                        # Get the index position from vocab, then append that index position to indices
                        indices.append(vocab[f])
                        # Convert the val to the correct dtype, then append to our values list
                        values.append(dtype(val))

                indptr.append(len(indices))

            if len(indptr) == 1:
                raise ValueError('The DataFrame passed into DataFrameVectorizer is empty')

        indices = frombuffer_empty(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        # # Sort everything if asked
        # if fitting and self.sort:
        #     feature_names.sort()
        #     map_index = np.empty(len(feature_names), dtype=np.int32)
        #     for new_val, f in enumerate(feature_names):
        #         map_index[new_val] = vocab[f]
        #         vocab[f] = new_val
        #     result_matrix = result_matrix[:, map_index]

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        # if fitting:
        #     self.feature_names_ = feature_names
        #     self.vocabulary_ = vocab
        return result_matrix


    def transform(self, X, y=None):
        """Transform DataFrame to array or sparse matrix.

        Columns (or string values in categorical columns) not encountered during fit will be
        silently ignored.

        Parameters
        ----------
        X : DataFrame where all values are strings or convertible to dtype.
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        return self._transform(X)
        # if self.sparse:
        #     return self._transform(X)

        # else:
        #     dtype = self.dtype
        #     vocab = self.vocabulary_
        #     X = _tosequence(X)
        #     Xa = np.zeros((len(X), len(vocab)), dtype=dtype)

        #     for i, x in enumerate(X):
        #         for f, v in six.iteritems(x):
        #             if isinstance(v, six.string_types):
        #                 f = "%s%s%s" % (f, self.separator, v)
        #                 v = 1
        #             try:
        #                 Xa[i, vocab[f]] = dtype(v)
        #             except KeyError:
        #                 pass

        #     return Xa

    def get_feature_names(self):
        """Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        """
        return self.feature_names_

    def restrict(self, support, indices=False):
        """Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : boolean, optional
            Whether support is a list of indices.

        Returns
        -------
        self

        Examples
        --------
        >>> from sklearn.feature_extraction import DictVectorizer
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> v = DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
        >>> v.get_feature_names()
        ['bar', 'baz', 'foo']
        >>> v.restrict(support.get_support()) # doctest: +ELLIPSIS
        DictVectorizer(dtype=..., separator='=', sort=True,
                sparse=True)
        >>> v.get_feature_names()
        ['bar', 'foo']
        """
        if not indices:
            support = np.where(support)[0]

        names = self.feature_names_
        new_vocab = {}
        for i in support:
            new_vocab[names[i]] = len(new_vocab)

        self.vocabulary_ = new_vocab
        self.feature_names_ = [f for f, i in sorted(six.iteritems(new_vocab),
                                                    key=itemgetter(1))]

        return self

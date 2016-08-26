Formatting Data
===============

WARNING: Lazers ahead.

Err, ok, more reasonably, the warning is to actually run your code first before reading any further. There aren't actually any lazers involved unless you're running on some kind of a crazy quantum computer (hardware geeks, feel free to submit a PR to make this joke technically accurate).


You probably don't need to read this page unless you're getting an error message.

Try running your code first, and come back here only if you're not getting the results you expect. The format of the data is designed to be intuitive.


Training data format
---------------------
#. Must be a list (or other iterable) filled with python dictionaries.
#. The non-header-row objects can be "sparse". You only have to include attributes on each object that actually apply to that row.

Header row information
-----------------------
The ``column_descriptions`` dictionary passed into ``Predictor()`` is essentially the header row. Here you've gotta specify some basic information about each "column" of data in the other dictionaries. This object should essentially have the same attributes as the following objects, except the values stored in each attribute will tell us information about that "column" of data.

#. ``attribute_name: 'output'`` The ``column_descriptions`` dictionary must specify one of your attributes as the output column. This is what the ``auto_ml`` predictor will try to predict. Importantly, the data you pass into ``.train()`` should have the correct values for this column, so we can teach the algorithms what is right and what is wrong.
#. ``attribute_name: 'categorical'`` All attribute names that hold a string in any of the rows after the header row will be encoded as categorical data. If, however, you have any numerical columns that you want encoded as categorical data, you can specify that here.
#. ``attribute_name: 'nlp'`` If any of your data is a text field that you'd like to run some Natural Language Processing on, specify that in the header row. Data stored in this attribute will be encoded using TF-IDF, along with some other feature engineering (count of some aggregations like total capital letters, puncutation characters, smiley faces, etc., as well as a sentiment prediction of that text). SERIOUS WARNING: This isn't built yet. It will be pretty simple to build, so if you want to submit that PR, I'd love to merge it in!

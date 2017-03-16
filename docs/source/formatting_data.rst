Formatting Data
===============

WARNING: Lazers ahead.

Err, ok, more reasonably, the warning is to actually run your code first before reading any further. There aren't actually any lazers involved unless you're running on some kind of a crazy quantum computer (hardware geeks, feel free to submit a PR to make this joke technically accurate).


You probably don't need to read this page unless you're getting an error message.

Try running your code first, and come back here only if you're not getting the results you expect. The format of the data is designed to be intuitive.


Training data format
---------------------
#. Must either be a pandas DataFrame, or a list filled with python dictionaries.
#. The non-header-row objects can be "sparse". You only have to include attributes on each object that actually apply to that row. In fact, we'd recommend passing in None or nan if you have missing values- knowing that a value is missing can itself be signal.

Header row information
-----------------------
The ``column_descriptions`` dictionary passed into ``Predictor()`` is essentially the header row. Here you've gotta specify some basic information about each "column" of data in the DataFrame or list of dictionaries. This column_descriptions object will tell us information about that "column" of data.

NOTE: We assume each column is a numerical column, unless you specify otherwise using one of the types noted below.

#. ``attribute_name: 'output'`` The ``column_descriptions`` dictionary must specify one of your attributes as the output column. This is what the ``auto_ml`` predictor will try to predict. Importantly, the data you pass into ``.train()`` should have the correct values for this column, so we can teach the algorithms what is right and what is wrong.
#. ``attribute_name: 'categorical'`` All attribute names that hold a string in any of the rows after the header row will be encoded as categorical data. If, however, you have any numerical columns that you want encoded as categorical data, you can specify that here.
#. ``attribute_name: 'nlp'`` If any of your data is a text field that you'd like to run some Natural Language Processing on, specify that in the header row. Data stored in this attribute will be encoded using TF-IDF, along with some other feature engineering (count of some aggregations like total capital letters, puncutation characters, smiley faces, etc., as well as a sentiment prediction of that text).
#. ``attribute_name: 'ignore'`` This column of data will be ignored.
#. ``attribute_name: 'date'`` Since ML algorithms don't know how to handle a Python datetime object, we will perform feature engineering on this object, creating new features like day_of_week, or minutes_into_day, etc. Then the original date field will be removed from the training data so the algorithsm don't throw a TypeError.


Passing in your own feature engineering function
=================================================

You can pass in your own function to perform feature engineering on the data. This will be called as the first step in the pipeline that ``auto_ml`` builds out.

You will be passed the entire X dataset (not the y dataset), and are expected to return the entire X dataset.

The advantage of including it in the pipeline is that it will then be applied to any data you want predictions on later. You will also eventually be able to run GridSearchCV over any parameters you include here.

Limitations:
You cannot alter the length or ordering of the X dataset, since you will not have a chance to modify the y dataset. If you want to perform filtering, perform it before you pass in the data to train on.

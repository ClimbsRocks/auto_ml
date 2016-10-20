"""
To get standard out, run nosetests as follows:
  $ nosetests -sv tests
"""
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import with_setup
 
def my_setup_function():
    print("setting up stuff")
 
def my_teardown_function():
    print("tearing down stuff")
 
@with_setup(my_setup_function, my_teardown_function)
def test_basic():
    assert_not_equal("trump", "clinton")
    assert_equal(2, 1+1)

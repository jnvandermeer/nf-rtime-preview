"""Miscellaneous Helper Methods.


"""


import logging
from functools import wraps


logger = logging.getLogger(__name__)


class deprecated(object):
    """Mark a method deprecated

    This method is used internally to mark methods as deprecated.
    Deprecated methods will print a warning when used but will otherwise
    function as usual.

    Parameters
    ----------
    since : str
        The version number that introduced the deprecation of the
        method.
    alternative : str, optional
        The method that should be used instead.

    """
    def __init__(self, since, alternative=None):
        self.since = since
        self.alternative = alternative

    def __call__(self, f):
        msg = "{f} is deprecated since version {v}.".format(f=f.__name__, v=self.since)
        if self.alternative is not None:
            msg2 = " Please use {f} instead.".format(f=self.alternative)
            msg = msg + msg2
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            logger.warn(msg)
            return f(*args, **kwargs)
        return wrapped_f


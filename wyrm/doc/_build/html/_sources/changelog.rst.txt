Changelog
=========

Version 1.1.0
-------------

This version mainly adds the new :func:`wyrm.apply_spatial_filter` method, for
applying spatial filtering used in CSP, CCA, whitening, etc. All manual spatial
filtering has been replaced throughout the toolbox to use the new method
instead.

New Methods
~~~~~~~~~~~

* New decorator :class:`wyrm.misc.deprecated` that is used internally for
  marking methods as deprecated
* New method :func:`wyrm.processing.apply_spatial_filter` for applying spatial
  filters like CSP, CCA, whitening, etc.

Deprecated Methods
~~~~~~~~~~~~~~~~~~

* Deprecated method :func:`wyrm.processing.apply_csp`. One should use
  :func:`wyrm.processing.apply_spatial_filter` instead

Bugfixes
~~~~~~~~

* Fixed bug in :func:`wyrm.processing.calculate_whitening_matrix`, that
  incorrectly calculated the whitening matrix due to a missing transpose


Version 1.0.0
-------------

We bumped the version up to 1 without backwards-incompatible changes since the
last version.

New Methods
~~~~~~~~~~~

* New method :meth:`wyrm.processing.rereference` for rereferencing channels
* New method :meth:`wyrm.processing.calculate_whitening_matrix`

Improvements
~~~~~~~~~~~~

* :meth:`wyrm.plot.plot_channels` is now able to plot continuous and epoched
  data
* :meth:`wyrm.plot.plot_channels` allows for configuring the number of columns
  of the grid

Misc
~~~~

* Upgraded to Sphinx 1.3.1
* We use napoleon instead of the numpydoc plugin
* Several fixes for various docstring issues

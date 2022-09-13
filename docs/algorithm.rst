========================
Algorithm Implementation
========================

This page provides a description of the correlation calibration algorithm.

Setup
=====

The calibration routine begins by setting up a :class:`~.CorrCal` object from
an appropriate set of files. The :class:`~.CorrCal` object contains the data
to be calibrated, the model covariance, the calibration solution, and various
metadata necessary for running the calibration routine. The data is first read
into a :class:`pyuvdata.UVData` object, then reshaped and sorted into redundant
blocks, with relevant metadata pulled from the :class:`pyuvdata.UVData` object
as needed. The model covariance can either be built from a sky model or loaded
directly into a :class:`~.Sparse2Level` object from a ``.covh5`` file. The
calibration solution is loaded into a :class:`pyuvdata.UVCal` object.

Attribute Descriptions
======================
- ``CorrCal.data``
   - Visibility data arranged into an array with shape
   ``(Nbls, Ntimes, Nfreqs, Npols)``
- ``CorrCal.cov``
   - Model covariance loaded into a 

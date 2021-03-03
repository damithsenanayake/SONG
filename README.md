# SONG
This Repository Contains the Source Code for the Python Implementation of Self Organizing Nebulous Growths.

*NOTE: This implementation has several improvements not described in the publication* 

## Installation
SONG is now available in the Python Packaging Index (PyPI). You can simply install the stable version (current release 1.0.0) via ``pip`` using the following command. 

``$ pip install song-vis``

If you would like the bleeding edge version, you can simply download it via the github source code. To install this package, download the zip file, extract it, and run the following command on the commandline

``$ cd <extracted_location>``

``$ python setup.py install``

Alternatively, you can use pip to install this package as follows

``$ cd <extracted_location>``

``$ pip install .``

## Running The Algorithm

In python, after installation, run the following. 

``from song.song import SONG``

This will import the libraries. To execute the algorithm run the following

`` Y = SONG().fit_transform(X)``

To retain a model after training it, use a variable to store the model

``model = SONG()``<br>
``model.fit(X)``

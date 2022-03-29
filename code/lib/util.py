#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================

import csv
import scipy.io as spio


# ==================================================================================================
# CLASSES
# ==================================================================================================

class Writer:
    def __init__(self, *args):
        self.writers = list(args)

    def add(self, writer):
        self.writers += [writer]

    def remove(self, writer):
        self.writers.remove(writer)

    def write(self, text):
        for writer in self.writers:
            writer.write(text)

    def flush(self):
        pass


# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def csv2dict(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)

        # Get the keys from the csv file
        keys = reader.__next__()

        # Get the values from the csv file
        values = []
        for values_ in reader:
            values.append(values_)
        values = list(map(list, zip(*values)))

    # Merge keys and values into the dictionary
    dictionary = dict(zip(keys, values))
    return dictionary


def dict2csv(dict, filename, sort=False):
    # Convert the dictionary to a sorted list
    dict_list = list(dict.items())
    if sort:
        dict_list = sorted(dict_list)
    keys, values = zip(*dict_list)

    with open(filename, 'x') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(keys)
        if isinstance(values[0], list):
            values = zip(*values)
            for row in values:
                writer.writerow(row)
        else:
            writer.writerow(values)


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


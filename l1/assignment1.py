import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle

url = 'http://yaroslavvb.com/upload/notMNIST/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print 'Found and verified', filename
    else:
        raise Exception(
                'Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10


def extract(filename):
    tar = tarfile.open(filename)
    tar.extractall()
    tar.close()
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    if len(data_folders) != num_classes:
        raise Exception(
                'Expected %d folders, one per class. Found %d instead.' % (
                    num_classes, len(data_folders)))
    print data_folders
    return data_folders


train_folders = extract(train_filename)
test_folders = extract(test_filename)

"""Problem 1
---------

Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character
A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the
package IPython.display."""


def show_an_image(folders):
    for i in folders:
        for f in os.listdir(i):
            img = os.path.join(i, f)
            if os.path.isfile(img):
                break
        print(img)
        display(Image(filename=img))


show_an_image(train_folders)
show_an_image(test_folders)

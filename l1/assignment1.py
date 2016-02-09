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

"""" End of Problem 1 """""

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load(data_folders, min_num_images, max_num_images):
    dataset = np.ndarray(
            shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0
    image_index = 0
    for folder in data_folders:
        print folder
        for image in os.listdir(folder):
            if image_index >= max_num_images:
                raise Exception('More images than expected: %d >= %d' % (
                    image_index, max_num_images))
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) -
                              pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1
            except IOError as e:
                print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'
        label_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (
            num_images, min_num_images))
    print 'Full dataset tensor:', dataset.shape
    print 'Mean:', np.mean(dataset)
    print 'Standard deviation:', np.std(dataset)
    print 'Labels:', labels.shape
    return dataset, labels


train_dataset, train_labels = load(train_folders, 450000, 550000)
test_dataset, test_labels = load(test_folders, 18000, 20000)

"""Problem 2

Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray.
Hint: you can use matplotlib.pyplot."""


def plot_an_image(dataset_img, label):
    plt.imshow(dataset_img)
    print(str(label))
    plt.title(str(label))
    display(plt.show())


plot_an_image(test_dataset[19], test_labels[19])
plot_an_image(test_dataset[15000], test_labels[15000])
plot_an_image(test_dataset[8900], test_labels[8900])
plot_an_image(train_dataset[19], train_labels[19])
plot_an_image(train_dataset[135000], train_labels[135000])
plot_an_image(train_dataset[28900], train_labels[28900])

"""" End of Problem 2 """""

np.random.seed(133)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

"""Problem 3

Convince yourself that the data is still good after shuffling!"""

plot_an_image(test_dataset[19], test_labels[19])
plot_an_image(test_dataset[15000], test_labels[15000])
plot_an_image(test_dataset[8900], test_labels[8900])
plot_an_image(train_dataset[19], train_labels[19])
plot_an_image(train_dataset[135000], train_labels[135000])
plot_an_image(train_dataset[28900], train_labels[28900])

"""Problem 4

Another check: we expect the data to be balanced across classes. Verify that."""
plt.hist(test_labels)
plt.title("test classes")
display(plt.show())
plt.hist(train_labels)
plt.title("train classes")
display(plt.show())

train_size = 200000
valid_size = 10000

valid_dataset = train_dataset[:valid_size, :, :]
valid_labels = train_labels[:valid_size]
train_dataset = train_dataset[valid_size:valid_size + train_size, :, :]
train_labels = train_labels[valid_size:valid_size + train_size]
print 'Training', train_dataset.shape, train_labels.shape
print 'Validation', valid_dataset.shape, valid_labels.shape

pickle_file = 'notMNIST.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print 'Unable to save data to', pickle_file, ':', e
    raise
statinfo = os.stat(pickle_file)
print 'Compressed pickle size:', statinfo.st_size

"""Problem 5

By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained
in the validation and test set! Overlap between training and test can skew the results if you expect to use your model
in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when
you use it. Measure how much overlap there is between training, validation and test samples. Optional questions:

    What about near duplicates between datasets? (images that are almost identical)
    Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
"""

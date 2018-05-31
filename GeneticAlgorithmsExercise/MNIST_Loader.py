import mnist
from NetworkBackPropagation import *
from random import seed

def get_train_images():
    mndata = mnist.MNIST('resources')
    mndata.gz = True
    images, labels = mndata.load_training()
    return images, labels


def create_csv_from_data(images ,labels):
    lines = []
    for i in xrange(len(labels)):
        to_add = list(images[i])
        to_add.extend([labels[i]])
        lines.append(','.join(str(x) for x in to_add))
    with open('resources/mnist_w_labels.csv', 'wb') as csvfile:
        for line in lines:
            csvfile.write(line + '\n')


if __name__ == "__main__":
    images, labels = get_train_images()
    # create_csv_from_data(images, labels)

    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    filename = 'resources/seeds_dataset.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    n_folds = 5
    l_rate = 0.5
    n_epoch = 500
    n_hidden = 5
    scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

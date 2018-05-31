from mnist import MNIST
from NetworkBackPropagation import *
from MNIST_Loader import *
from random import seed


if __name__ == "__main__":
    # Create csv from MNIST train images
    images, labels = get_train_images()
    # create_csv_from_data(images, labels, 'mnist_w_labels_2500.csv', 2500)

    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    filename = 'resources/mnist_w_labels_2500.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    n_folds = 5 # remove this n_folds! we have train and test datasets
    l_rate = 0.5
    n_epoch = 50
    n_hidden = 5
    scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    send_results_via_email('Scores: %s, Mean Accuracy: %.3f%%' % (scores, (sum(scores)/float(len(scores)))) , "Genetic Run 128 64")

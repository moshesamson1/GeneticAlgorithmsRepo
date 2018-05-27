import mnist
from mnist import MNIST

def get_train_images():
    mndata = MNIST('resources')
    mndata.gz = True
    images, labels = mndata.load_training()
    return images, labels

if __name__ == "__main__":
    images, labels = get_train_images()
    print images[1]
import cv2
import glob
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def summarize_data(ds, show_classes=False):
    print("Number of training examples = ", ds.get_size_of_train())
    print("Number of validation examples = ", ds.get_size_of_valid())
    print("Number of testing examples = ", ds.get_size_of_test())
    print("Image data shape = ", ds.get_size_of_image())
    print("Number of classes (labels) = ", ds.get_num_of_classes())
    if(show_classes):
        print("The class (label) names are listed as below")
        for cls in ds.get_classes():
            print(ds.get_class_name(cls))

def plot_sign_images(ds):
    """
    Plot one sign image for each class
    """
    images, labels = ds.get_train()
    visited = set()
    demos = []
    num_of_classes = ds.get_num_of_classes()
    for i, class_id in enumerate(labels):
        if class_id not in visited:
            visited.add(class_id)
            #print(ds.get_class_name(class_id))
            demos.append(images[i])
            if len(visited) == num_of_classes:
                break
    show_images(demos)
    
def show_images(images, cols=8, cmap=None):
    """
    print images
    """
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(20, 20)) #
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

def generate_noisy_image(img):
    
    def random_rotate(image):
        if random.randrange(2) == 0:
            angle = random.randrange(10)
        else:
            angle = random.randrange(10) * -1
        M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
        return cv2.warpAffine(image, M, (32, 32))

    def random_blur(img):
        """
        Blur the image
        """
        # kernel size options: 1, 3
        kernel_size = 2 * random.randrange(2) + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def random_zoom(img):
        """
        Zoom out a grayscale image
        """
        inc = random.randrange(5)
        return cv2.resize(img, (32 + 2*inc, ) * 2)[inc:inc+32, inc:inc+32]
    
    return random_zoom(
        random_blur(
            random_rotate(img)
        )
    )

def get_new_data(image_size):
    labels = []
    images = []
    names = []
    for fname in glob.glob('new_images/*'):
        labels.append(int(fname.split('/')[-1].split('-')[0]))
        names.append(fname.split('/')[-1].split('-', 1)[1][:-4])
        img = cv2.resize(mpimg.imread(fname), image_size, interpolation=cv2.INTER_CUBIC)
        images.append(img)
    labels = np.array(labels)
    # apply our preprocessing: grayscale conversion and normalization
    images = np.sum(np.array(images) / 3.0, axis=3, keepdims=True)
    images = (images - 128.0) / 128.0
    return images, labels

def show_new_data():
    images = []
    for fname in glob.glob('new_images/*'):
        images.append(mpimg.imread(fname))
    show_images(images, cols=5)
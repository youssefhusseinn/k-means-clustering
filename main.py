import time
from shutil import rmtree
import matplotlib.pyplot as plt
import pickle
import numpy
from math import *
import os
import random

classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

MAX_ITERATIONS = 15
FIRST_RED = 0
FIRST_GREEN = 1024
FIRST_BLUE = 2048
DIMENSION = 32 * 32
K = 10
NUMBER_OF_CLASSES = 10


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_image(img):
    im_r = img[FIRST_RED:FIRST_GREEN].reshape(32, 32)
    im_g = img[FIRST_GREEN:FIRST_BLUE].reshape(32, 32)
    im_b = img[FIRST_BLUE:].reshape(32, 32)
    img = numpy.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.show()


def find_max_label(images):
    freq = [0 for x in range(NUMBER_OF_CLASSES)]
    if len(images) == 0:
        return freq
    for img in images:
        freq[img[1]] += 1
    for i in range(NUMBER_OF_CLASSES):
        freq[i] = freq[i] * 100 / len(images)
    return freq


def construct_image(img):
    im_r = img[FIRST_RED:FIRST_GREEN].reshape(32, 32)
    im_g = img[FIRST_GREEN:FIRST_BLUE].reshape(32, 32)
    im_b = img[FIRST_BLUE:].reshape(32, 32)
    return numpy.dstack((im_r, im_g, im_b))


def save_data(centroid, cluster, centroid_id):
    freq = find_max_label(cluster)
    dominant_class = freq.index(max(freq))
    cluster_path = f'./samples/CLUSTER={centroid_id} SIZE={len(cluster)} DOM={dominant_class}{classes[dominant_class]}'
    os.makedirs(cluster_path, 0o666, True)
    centroid = construct_image(centroid)
    plt.imsave(cluster_path + f'/centroid{centroid_id}.png', centroid)
    for i in range(min(20, len(cluster))):
        image_to_save = random.choice(cluster)
        plt.imsave(cluster_path + f'/img-{classes[image_to_save[1]]}-{i}.png',
                   construct_image(numpy.array(image_to_save[0], dtype="uint8")))
    classes_percentage = []
    for i in range(NUMBER_OF_CLASSES):
        classes_percentage.append((classes[i], str(round(freq[i], 2)) + '%'))
    with open(cluster_path + "/classes_frequency.txt", "w") as output:
        output.write(str(classes_percentage))


def plot_graph(distortions):
    x = []
    y = []
    for i in range(len(distortions)):
        x.append(distortions[i][0])
        y.append(distortions[i][1])
    plt.plot(x, y)
    plt.xlabel('iterations')
    plt.ylabel('distortions')
    plt.title('distortion_graph')
    plt.savefig('./samples/distortion_graph.png')
    plt.show()


def update_centroid(id, cluster):
    # # ROW MAJOR
    new_array = [0.0 for i in range(3 * DIMENSION)]
    rows = len(cluster)
    cols = 3 * DIMENSION
    for i in range(rows):
        for j in range(cols):
            new_array[j] += cluster[i][0][j]
    if rows > 0:
        for i in range(cols):
            new_array[i] /= rows
    return id, new_array

    # # COLUMN MAJOR
    # new_array = []
    # size = len(cluster)
    # for i in range(3 * 1024):
    #     sum = 0
    #     for j in range(size):
    #         sum += cluster[j][0][i]
    #     if size > 0:
    #         new_array.append(int(sum / size))
    # return id, new_array


def euclidean_distance(rgb_1, rgb_2):
    r = float(rgb_1[0]) - float(rgb_2[0])
    g = float(rgb_1[1]) - float(rgb_2[1])
    b = float(rgb_1[2]) - float(rgb_2[2])
    return sqrt(r * r + g * g + b * b)


def manhattan_distance(rgb_1, rgb_2):
    r = float(rgb_1[0]) - float(rgb_2[0])
    g = float(rgb_1[1]) - float(rgb_2[1])
    b = float(rgb_1[2]) - float(rgb_2[2])
    return abs(r) + abs(g) + abs(b)


def examples_distance(image1, image2):
    dist = 0
    for i in range(DIMENSION):
        img1_rgb = (image1[i + FIRST_RED], image1[i + FIRST_GREEN], image1[i + FIRST_BLUE])
        img2_rgb = (image2[i + FIRST_RED], image2[i + FIRST_GREEN], image2[i + FIRST_BLUE])
        dist += euclidean_distance(img1_rgb, img2_rgb)
    return dist


def distortion(cluster, centroid):
    #J(c,μ)=∑i=1m||x(i)−μc(i)||2
    sum = 0
    for img in cluster:
        d0 = examples_distance(img[0], centroid)
        sum += d0*d0
    return sum


def start_clustering(all_images):
    #initialize centroids randomly
    centroids = [(i, random.choice(all_images)) for i in range(K)]  # centroid_id, represented_image
    clusters = [[] for i in range(K)]
    start_time = time.time()
    distortions = []
    # K means for loop
    for iteration in range(MAX_ITERATIONS):
        img_idx = 0
        # assign images to clusters
        for image in all_images:
            nearest_centroid_id = None
            nearest_centroid_distance = 2e18

            # find best centroid
            for centroid in centroids:
                centroid_id, centroid_arr = centroid
                distance = examples_distance(image, centroid_arr)
                # improve distance if you can
                if distance < nearest_centroid_distance:
                    nearest_centroid_id = centroid_id
                    nearest_centroid_distance = distance
            #append image to the nearest centroid's cluster
            clusters[nearest_centroid_id].append((image, all_labels[img_idx]))
            img_idx += 1
        print(f'clustering done {iteration+1}   {time.time() - start_time}   {round((time.time() - start_time) / 60, 3)}')
        total_distortion = 0.0
        centroid_idx = 0
        #measure distortion for each cluster J(c,μ)=∑i=1m||x(i)−μc(i)||2
        for cluster in clusters:
            cluster_distortion = distortion(cluster, centroids[centroid_idx][1])
            total_distortion += cluster_distortion
            centroid_idx += 1
        distortions.append((iteration, total_distortion))
        #update centroids to be the mean on their clusters
        for i in range(K):
            new_centroid = update_centroid(i, clusters[i])
            if len(new_centroid[1]) > 0:
                centroids[i] = new_centroid
        if iteration < MAX_ITERATIONS - 1:
            clusters = [[] for i in range(K)]

    # save output
    for i in range(K):
        np = numpy.array(centroids[i][1], dtype="uint8")
        save_data(np, clusters[i], str(i + 1))
    print("clusters saved")
    plot_graph(distortions)


def convert(x):
    #converting numpy arrays to normal lists
    ret = []
    for i in x:
        ret.append(int(i))
    return ret


if __name__ == '__main__':
    """
    1 - DOWNLOAD CIFAR-10 PYTHON IMAGE DATASET FROM https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    2 - EXTRACT cifar-10-python.tar.gz IN PROJECT DIRECTORY
    3 - RUN
    """
    try:
        rmtree(f'./samples')
    except:
        pass
    global all_images
    global all_labels

    dict1 = unpickle("./cifar-10-batches-py/data_batch_1")
    dict2 = unpickle("./cifar-10-batches-py/data_batch_1")
    dict3 = unpickle("./cifar-10-batches-py/data_batch_1")
    dict4 = unpickle("./cifar-10-batches-py/data_batch_1")
    dict5 = unpickle("./cifar-10-batches-py/data_batch_1")
    all_imagess = []
    all_labels = []

    for i in range(10000):
        all_imagess.append(convert(dict1[b'data'][i]))
        all_labels.append(dict1[b'labels'][i])
    for i in range(10000):
        all_imagess.append(convert(dict2[b'data'][i]))
        all_labels.append(dict2[b'labels'][i])
    for i in range(10000):
        all_imagess.append(convert(dict3[b'data'][i]))
        all_labels.append(dict3[b'labels'][i])
    for i in range(10000):
        all_imagess.append(convert(dict4[b'data'][i]))
        all_labels.append(dict4[b'labels'][i])
    for i in range(10000):
        all_imagess.append(convert(dict5[b'data'][i]))
        all_labels.append(dict5[b'labels'][i])

    start_clustering(all_imagess)

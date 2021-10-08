from sklearn import datasets
import heapq
from numpy.linalg import norm


def main():
    digits = datasets.fetch_openml('mnist_784')
    data = digits.data
    target = digits.target
    data_list = list(digits.data)
    images_and_labels = list(zip(digits.data, digits.target))

    split(images_and_labels)
    prediction = knn(images_and_labels[:20], list(digits.data[21:23]), 3)
    test_model()
    print(prediction)


def test_model(k=3):
    """
    Uses the Scikit defined KNN implementation to offer predictions on the open ML dataset
    :param k: the k value to be used in the KNN algorithim
    :return: a dicitonary of f1 scores for each digit
    """
    from sklearn.neighbors import KNeighborsClassifier
    digits = datasets.fetch_openml('mnist_784')
    n_samples = len(digits.data)
    data = digits.data.reshape((n_samples, -1))
    # define Model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(split(data)['train'], split(digits.target)['train'])
    y_hat = model.predict(data[int(n_samples * .8):int(n_samples * .95)])
    y = digits.target[int(n_samples * .8):]
    f_1_dict = find_f_1(y_hat, y)
    print('f1 scores for digits 0-9 from a', k, 'nearest neighbor model are')
    for number in list(f_1_dict.values()):
        print('{:.2f} & '.format(number), end='')
    return f_1_dict


def knn(train, pred, k):
    """
    Uses the train data to make predictions on the test data
    :param train: zipped list of tupules, 1st element picture array, second element classification eg "1"
    :param pred: just the picture array, we want to predict the classification for this array/picture
    :param k: number of near neighbors to consider
    """
    inf = 10 ** 6  # ~inf
    nn_heap = []
    prediction_list = []
    for _ in range(k):
        # use negatives to turn the implemented min heap into a max heap
        heapq.heappush(nn_heap, (-1 * inf, '1'))  # initialize the set of NN to five 1's at infinite distance
    for i in range(len(pred)):
        for j in range(len(train)):
            dist = euclid_dist(pred[i], train[j][0])
            if dist < -1 * nn_heap[0][0]:
                heapq.heappushpop(nn_heap, (-1 * dist, train[j][1]))  # add distance and classification
            assert len(nn_heap) == k
        prediction_list.append(classify(nn_heap))
    return prediction_list


def classify(nn_tup_list):
    """
    Classifies a given datapoint based on its neighbors and their distance from the datapoint

    :param nn_tup_list: a list of tuple of neighbors and their distance from the data point
    :return: A classification of the unknown datapoint
    """
    count_list = [0] * 10
    neighbors = list(zip(*nn_tup_list))[1]
    distances = list(zip(*nn_tup_list))[0]
    for i in range(10):  # count the instances of each prediction
        for neighbor in neighbors:
            if neighbor == str(i):
                count_list[i] += 1

    maximum = max(count_list)  # most common prediction, (min to account for negative values in the max heap)
    if maximum == 1:
        return neighbors(neighbors.index(max(distances)))
    index = count_list.index(maximum)
    prediction = neighbors[index]
    return prediction


def euclid_dist(v1, v2):
    """
    Finds the Euclidean distance between two vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: euclidean distance between them
    """
    return norm(v1.flatten() - v2.flatten())


def split(data):
    """
    returns a train, validate, and test dictionary
    """
    split_dict = {}
    split_dict.update({'train': data[0:int(len(data) * 0.8)]})
    split_dict.update({'validate': data[int(len(data) * 0.8):int(len(data) * 0.95)]})
    split_dict.update({'test': data[int(len(data) * 0.95)::]})
    return split_dict


def find_f_1(predicted, actual):
    """
    Returns f_1 metrics, hardcoded specific to the MNIST 1 to ten classification
    :param predicted: list of string integers
    :param actual: list of string integers
    :return: f_1_dict dictionary of the f_1 scores for each number
    """
    f_1_dict = {}
    for i in range(10):
        correct_positive = 0
        predicted_positive = 0
        total_positive = 0
        for j in range(len(predicted)):
            if predicted[j] == str(i):
                predicted_positive += 1
                if actual[j] == str(i):
                    correct_positive += 1
                    total_positive += 1
            elif actual[j] == str(i):
                total_positive += 1
        precision = correct_positive / predicted_positive
        recall = correct_positive / total_positive
        f_1_dict[str(i)] = 2 * (precision * recall) / (precision + recall)
    return f_1_dict


if __name__ == "__main__":
    test_model()

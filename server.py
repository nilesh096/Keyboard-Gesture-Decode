'''
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.
'''

from flask import Flask, request
from flask import render_template
import time
import json
import sys
from scipy.interpolate import interp1d
import numpy as np
import math
import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

def get_diff(points):
    diff = [0]
    for i in range(len(points)-1):
        diff.append(points[i+1]-points[i])
    return np.square(diff)

def cumsum(l: list):
    if not l:
        return []
    sums = []
    sums.append(l[0])
    return helper_cumsum(sums, l[1:])

def helper_cumsum(sums: list, xs: list):
    if not xs:
        return sums
    s = sums[len(sums) - 1]
    sums.append(s + xs[0])
    if len(xs) > 1:
        return helper_cumsum(sums, xs[1:])
    return sums

def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling (10 points)
    #sample_points_X = list()
    #sample_points_Y = list()

    X = points_X
    Y = points_Y

    alpha_value = np.linspace(0, 1, 100)
    dist = np.cumsum(np.sqrt(np.ediff1d(X, to_begin=0)**2  + np.ediff1d(Y, to_begin=0)**2))

    dist /= dist[len(dist)-1]

    dist_x, dist_y = interp1d(dist, X), interp1d(dist, Y)
    sample_points_X, sample_points_Y = dist_x(alpha_value), dist_y(alpha_value)
    return sample_points_X, sample_points_Y

template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def calculate_distance(x,y):
    return math.sqrt(pow(x,2) + pow(y,2))

def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 18
    # TODO: Do pruning (10 points)
    len_temp_points = len(template_sample_points_X)
    start_x = [gesture_points_X[0][0]]*len_temp_points
    start_y = [gesture_points_Y[0][0]]*len_temp_points

    end_x = [gesture_points_X[0][-1]]*len_temp_points
    end_y = [gesture_points_Y[0][-1]]*len_temp_points

    #Start Pruning
    for idx, val in enumerate(template_sample_points_X):
        d1 = calculate_distance(start_x[idx] - val[0], start_y[idx] - template_sample_points_Y[idx][0])
        d2 = calculate_distance(end_x[idx] - val[-1], end_y[idx] - template_sample_points_Y[idx][-1])

        if d1 > threshold or d2 > threshold:
            continue
        valid_template_sample_points_X.append(template_sample_points_X[idx])
        valid_template_sample_points_Y.append(template_sample_points_Y[idx])
        valid_words += [(words[idx],probabilities[words[idx]])]

    print("No. of valid words = %d" %(len(valid_words)))
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W)

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)
    return scaled_X, scaled_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    # TODO: Set your own L
    L = 1
    # TODO: Calculate shape scores (10 points)
    shape_scores = []
    if len(gesture_sample_points_X) == 0 or len(gesture_sample_points_Y) == 0:
        return shape_scores

    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return shape_scores

    #gesture_sample_points_X, gesture_sample_points_Y = get_scaled_points(gesture_sample_points_X, gesture_sample_points_Y, L)
    for idx, val in enumerate(valid_template_sample_points_X):
        _s = 0
        j = 0
        while j < 100:
            euc_distance = math.hypot(gesture_sample_points_X[0][j] - val[j],
                                              gesture_sample_points_Y[0][j] - valid_template_sample_points_Y[idx][j])
            _s += np.nan_to_num(euc_distance)
            j += 1
        shape_scores += [_s / 100]
    try:
        print(" Shape score of 1 = ", shape_scores[0])
    except Exception as e:
        print(str(e))
    return shape_scores

#Helper functions Original
def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance-r , 0)
        final_max += local_max
    return final_max

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

#Helper functions 2
def helper(x,y,points_X,points_Y):
    #nums = list()
    min_val = sys.maxsize
    for p in zip(points_X,points_Y):
        val = calculate_distance(p[1]-y, p[0]-x)
        if val < min_val:
            min_val = val
    return min_val

def location_score_helper1(gesture_sample_points_X,gesture_sample_points_Y,template_X,template_Y,r):
    _s = 0
    for idx,k in enumerate(zip(gesture_sample_points_X, gesture_sample_points_Y)):
        distance = helper(gesture_sample_points_X[0][idx],gesture_sample_points_Y[0][idx],template_X,template_Y)-r
        _s += max(distance,0)
    return _s

def location_score_helper2(x1,y1,x2,y2,D):
    if D != 0:
        return math.hypot(y2-y1,x2-x1)
    return 0

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores (10 points)
    values = [0.01]*100
    # Calculating location_scores.
    for idx, val in enumerate(valid_template_sample_points_X):
        _s = 0
        #D = get_delta(gesture_sample_points_X,gesture_sample_points_Y,valid_template_sample_points_X,valid_template_sample_points_Y,radius,i)
        D = location_score_helper1(gesture_sample_points_X, gesture_sample_points_Y, val,
                                    valid_template_sample_points_Y[idx], radius)
        for j in range(len(gesture_sample_points_X)-1,-1,-1):
            _s = _s + values[j] * location_score_helper2(gesture_sample_points_X[0][j], gesture_sample_points_Y[0][j],
                                                           valid_template_sample_points_X[idx][j],
                                                           valid_template_sample_points_Y[idx][j], D)
        location_scores += [_s]

    try:
        print(" Location score of 1 = ", location_scores[0])
    except Exception as e:
        print("ERROR: " + str(e))
    return location_scores[::-1]


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.85
    # TODO: Set your own location weight
    location_coef = 0.15
    for i in range(len(shape_scores)):
        integration_scores += [shape_coef * shape_scores[i] + location_coef * location_scores[i]]
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = len(valid_words)
    if n > 5:
        n = min(n,7)
    suggestion = ""
    # TODO: Get the best word (10 points)

    if len(integration_scores) == 0:
        return "No Word Found"
    sortedIndex = np.argsort(np.array(integration_scores))

    for i in range(0, n):
        print(" Words = ", valid_words[i][0], " ", integration_scores[i])

    return valid_words[sortedIndex[0]][0]


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()
    
    print('{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}')

    return '{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'


if __name__ == "__main__":
    app.run()

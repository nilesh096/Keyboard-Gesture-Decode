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
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
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
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)
        

    # Setting the list of x-axis and y-axis values of the gesture
    x_points = points_X
    y_points = points_Y
        # Here I am calculating the Euclidean distance and storing them in a variable so that I can later divide this distance into equidistant points
    # Reference: https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
    eu_distance = np.cumsum(np.sqrt( np.ediff1d(x_points, to_begin=0)**2 + np.ediff1d(y_points, to_begin=0)**2 ))
    eu_distance = eu_distance/eu_distance[-1]
    # Now, I am carrying out interpolation
    fx, fy = interp1d( eu_distance, x_points ), interp1d( eu_distance, y_points )
    # Final step is to divide the distance into 100 equidistant points, that would complete the process of sampling.
    alpha = np.linspace(0, 1, 100)
    sample_points_X, sample_points_Y = fx(alpha), fy(alpha)

    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

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
    threshold = 20
    # TODO: Do pruning (12 points)
    # Here I am computing the start-to-start and end-to-end distances between a template and the unknown gesture entered by the user.
    # Once I do that I have a predetermined threshold. Words satisfying this threshold will be returned as valid words.
    for index, (sample_point_x, sample_point_y) in enumerate(zip(template_sample_points_X, template_sample_points_Y)):
        start_dist = math.sqrt(((gesture_points_X[0][0] - sample_point_x[0])**2)+((gesture_points_Y[0][0] - sample_point_y[0])**2))
        end_dist = math.sqrt(((gesture_points_X[0][-1] - sample_point_x[-1])**2)+((gesture_points_Y[0][-1] - sample_point_y[-1])**2))
        if start_dist <= threshold:
            if end_dist <= threshold:
                valid_words.append(content[index].split('\t')[0])
                valid_template_sample_points_X.append(sample_point_x)
                valid_template_sample_points_Y.append(sample_point_y)
    print(valid_words)
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

# This is a custom helper function that carries out the task of scaling both the unknown gesture and the template points, by a scaling factor
def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_max = max(sample_points_X)
    x_min = min(sample_points_X)
    W = x_max - x_min
    y_max = max(sample_points_Y)
    y_min = min(sample_points_Y)
    H = y_max - y_min
    # Computing the scale factor
    s = L/max(H, W)
    
    gesture_X, gesture_Y = [], []
    # Normalizing every point with the scale factor
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(s * point_x)
        gesture_Y.append(s * point_y)
    
    # Final step is to move the centroid of all the points to the origin
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
    shape_scores = []
    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return shape_scores
    # TODO: Set your own L
    L = 1
    # Scaled the gesture and template points, by calling the get_scaled_points function
    gesture_sample_points_X, gesture_sample_points_Y = get_scaled_points(gesture_sample_points_X[0], gesture_sample_points_Y[0], L)
    scaled_template_points_X, scaled_template_points_Y = [] , []
    for template_points_X, template_points_Y in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        points_X, points_Y = get_scaled_points(template_points_X, template_points_Y, L)
        scaled_template_points_X.append(points_X)
        scaled_template_points_Y.append(points_Y)
    
    # Finally, compute the Euclidean Norm and return the shape scores
    for template_points_X, template_points_Y in zip(scaled_template_points_X, scaled_template_points_Y):
        d = 0
        for i in range(0, 100):
            dist = math.sqrt((gesture_sample_points_X[i] - template_points_X[i])**2 + (gesture_sample_points_Y[i] - template_points_Y[i])**2)
            d += dist
        shape_scores.append(d/100)
    
    # TODO: Calculate shape scores (12 points)
    
    print(shape_scores)
    return shape_scores

def get_small_d(p_X, p_Y, q_X, q_Y):
    min_d = []
    for i in range(0, 100):
        d = math.sqrt((p_X - q_X[i])**2 + (p_Y - q_Y[i])**2)
        min_d.append(d)
    return (sorted(min_d)[0])

def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for i in range(0, 100):
        local_max = 0
        d = get_small_d(p_X[i], p_Y[i], q_X, q_Y)
        local_max = max(d-r , 0)
        final_max += local_max
    return final_max

def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)

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
    # TODO: Calculate location scores (12 points)
    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return location_scores
    alpha = []
    for i in range(100):
        t=0
        if (i<50):
            t=50-i
        else:
            t =i-49
        alpha.append(t/(51*50))
    
    for template_points_X, template_points_Y in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        sum_score = 0
        for i in range(0, 100):
            delta = get_delta(gesture_sample_points_X[0], gesture_sample_points_Y[0], template_points_X, template_points_Y, radius, i)
            prod = delta * alpha[i]
            sum_score += prod
        location_scores.append(sum_score)
    print(location_scores)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.5
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
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
    n = 3
    # TODO: Get the best word (12 points)
     
    idx = integration_scores.index(min(integration_scores))
    best_word = valid_words[idx]
    return best_word


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
    print('{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}')

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()

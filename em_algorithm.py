# MIT License
#
# Copyright (c) 2018 Eugene Kolyvoshko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import norm

DEBUG = True


def generate_test_data(theta: dict, num_clusters: int) -> pd.DataFrame:
    """
    This function generates normal distributions
    :param theta: dictionary with parameters of distributions
    :param num_clusters: number of distributions
    :return: DataFrame with points
    """
    data = []
    labels = []
    for n in range(num_clusters):
        data.append(np.random.multivariate_normal(theta['mu'][n], theta['sig'][n], theta['size'][n]))
        labels += [n] * theta['size'][n]

    data = np.concatenate(data)
    data = pd.DataFrame(data=data, columns=['x', 'y'])
    data['label'] = labels

    if DEBUG:
        visualization(data, "debug_info/true_values.png", theta=theta, num_clusters=num_clusters)

    return data


def expectation(data: pd.DataFrame, theta: dict, num_clusters: int):
    points_clusters = []
    for i in range(num_clusters):
        points_clusters.append(theta['lambda'][i] *
                               norm.pdf(data['x'], theta['mu'][i][0], theta['sig'][i][0][0]) *
                               norm.pdf(data['y'], theta['mu'][i][1], theta['sig'][i][1][1]))
    data['label'] = np.argmax(points_clusters, axis=0)


def maximization(data: pd.DataFrame, theta: dict, num_clusters: int) -> dict:
    """
    update estimates of lambda, mu and sigma
    """
    percent_clusters = []
    data_size = len(data)
    for i in range(num_clusters):
        points_cluster = data[data['label'] == i]
        percent_clusters.append(len(points_cluster) / data_size)
        theta['mu'][i] = [points_cluster['x'].mean(), points_cluster['y'].mean()]
        theta['sig'][i] = [[points_cluster['x'].std(), 0], [0, points_cluster['y'].std()]]
    theta['lambda'] = percent_clusters
    return theta


def compute_l2_loss(old_params: dict, new_params: dict, num_clusters: int) -> float:
    dist = 0
    for p in range(num_clusters):
        for i in range(2):
            dist += (old_params['mu'][p][i] - new_params['mu'][p][i]) ** 2
    return dist ** 0.5


def draw_ellipse(position: List[float], covariance: List[float], ax: plt.Axes = None):
    """Draw an ellipse with a given position and covariance"""
    angle = 0
    width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for n in range(1, 4):
        ax.add_patch(Ellipse(position, n * width, n * height, angle, alpha=1 / (width + height), color='red'))


def visualization(data: pd.DataFrame, save_path: str, theta: dict = None, num_clusters: int = -1):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax = ax or plt.gca()

    ax.scatter(data['x'], data['y'], 24, c=data['label'])
    ax.axis('equal')

    if theta is not None and num_clusters > 0:
        for n in range(num_clusters):
            draw_ellipse(theta['mu'][n], [theta['sig'][n][0][0], theta['sig'][n][1][1]], ax)

    fig.savefig(save_path)


def em_algorithm(data: pd.DataFrame, theta: dict, num_clusters: int = 2) -> (pd.DataFrame, dict):
    """
    Implementation of Expectation Maximizatio (EM) Algorithm
    :param data: input data
    :param theta: starting parameters
    :param num_clusters: number of clusters
    :return: fitted data, hidden parameters of distribution
    """
    loss = 1e6
    epsilon = 0.01
    i = 0

    data['label'] = map(lambda x: x, np.random.choice(num_clusters, len(data)))

    while loss > epsilon:
        i += 1
        expectation(data, theta, num_clusters)
        old_theta = copy.deepcopy(theta)
        theta = maximization(data, theta, num_clusters)
        loss = compute_l2_loss(theta, old_theta, num_clusters=num_clusters)

        # logging
        print("iteration {}, loss {}".format(i, loss))

        if DEBUG:
            visualization(data, "debug_info/iteration{}.png".format(i), theta=theta, num_clusters=num_clusters)

    print('Final distance', compute_l2_loss(theta, p_theta, num_clusters=num_clusters))

    return data, theta


if __name__ == '__main__':
    if not os.path.exists('./debug_info'):
        os.makedirs('./debug_info')

    os.system('rm -rf ./debug_info/*')

    clusters = 2
    p_theta = {
        'mu': [[0, 5], [3, 0], [6, 6]],
        'sig': [[[2, 0], [0, 3]], [[200, 0], [0, 100]], [[1, 0], [0, 1]]],
        'size': [200, 200, 200],
    }

    theta_0 = {
        'mu': [[1, 1], [2, 2], [5, 5]],
        'sig': [[[1, 0], [0, 1]]] * clusters,
        'lambda': [1 / clusters] * clusters
    }

    p_data = generate_test_data(theta=p_theta, num_clusters=2)

    print('Start Expectation Maximizatio (EM) Algorithm')
    t_start = datetime.now()
    fit_data, approx_theta = em_algorithm(data=p_data, theta=theta_0, num_clusters=2)
    print('Eval time', datetime.now() - t_start)

    print('Hidden parameters:', approx_theta)

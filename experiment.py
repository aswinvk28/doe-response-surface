import pyDOE2
import numpy as np
import skvideo.io
from dataset import *
import argparse
import cv2
from time import sleep
import matplotlib.pyplot as plt
import imageio

MNIST_DIMENSION = 28

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='', type=str, default="", required=True)
    parser.add_argument('-fs', '--from_state', help='', type=str, default="", required=True)
    parser.add_argument('-fi', '--from_index', help='', type=int, default="", required=True)
    parser.add_argument('-ts', '--to_state', help='', type=str, default="", required=True)
    parser.add_argument('-ti', '--to_index', help='', type=int, default="", required=True)
    parser.add_argument('-n', '--num_experiments', help='', type=int, default="", required=True)

    return parser.parse_args()

def generate_random_experiments(num, criterion='center'):
    experiments = [pyDOE2.lhs(MNIST_DIMENSION, samples=MNIST_DIMENSION, criterion=criterion) for i in range(num)]
    return experiments

def generate_flow(experiments, ground_truth, num_experiments):
    return [experiments[i]*(experiments[i] - ground_truth) for i in range(num_experiments)]

def create_video_data(from_ground_truth, to_ground_truth, num_experiments=40):
    experiments = generate_random_experiments(num_experiments)
    from_intervals = generate_flow(experiments, from_ground_truth, num_experiments)
    to_intervals = generate_flow(experiments, to_ground_truth, num_experiments)
    from_norms = [np.linalg.norm(from_intervals[i]/np.max(from_intervals[i])) for i in range(num_experiments)]
    to_norms = [np.linalg.norm(to_intervals[i]/np.max(to_intervals[i])) for i in range(num_experiments)]

    from_sort = np.argsort(from_norms)[:int(num_experiments)]
    to_sort = np.argsort(to_norms)[::-1][:int(num_experiments)]
    from_norms = np.array(from_norms)/np.max(from_norms)
    to_norms = np.array(to_norms)/np.max(to_norms)

    alpha = np.linspace(0,1,num_experiments)
    beta = np.linspace(1,0,num_experiments)

    # im = [to_ground_truth]
    # out.write(to_ground_truth)

    for i in range(int(num_experiments)):
        from_idx = from_sort[i]
        to_idx = to_sort[i]
        img = np.matmul(
            alpha[i] * from_norms[from_idx] * from_intervals[from_idx]/np.max(from_intervals[from_idx]) + 
            beta[i] * to_norms[to_idx] * to_intervals[to_idx]/np.max(to_intervals[to_idx])
        , from_ground_truth)
        img = (img / np.max(img))*255
        imageio.imwrite("images/digits-"+i.__str__()+".png", cv2.resize(img, (50,50)))
    
    # out.write(from_ground_truth)

    return

if __name__ == "__main__":

    args = parse_args()
    
    if args.dataset == "mnist":
        X, y = get_XY(images_path='./train-images.idx3-ubyte', 
        labels_path='./train-labels.idx1-ubyte')
        from_ground_truth = get_mnist_digits(X, y, int(args.from_state), args.from_index)
        to_ground_truth = get_mnist_digits(X, y, int(args.to_state), args.to_index)
    elif args.dataset == "emnist":
        X, y = get_XY(images_path='./emnist-letters-train-images-idx3-ubyte', 
        labels_path='./emnist-letters-train-labels-idx1-ubyte')
        from_ground_truth = get_emnist_digits(X, y, args.from_index)
        to_ground_truth = get_emnist_digits(X, y, args.to_index)

    create_video_data(from_ground_truth, to_ground_truth, args.num_experiments)

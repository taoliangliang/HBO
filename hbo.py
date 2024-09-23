import math
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from scipy.spatial import distance_matrix



__author__ = 'taoll'

class HBO(object):
    def __init__(self,
                 k=5,
                 alpha=1,
                 step=0.001,
                 n_steps=1000,
                 approximate_heat=True,
                 stop_probability=0.01,
                 verbose=False):

        self.k = k
        self.alpha = alpha
        self.step = step
        self.n_steps =n_steps
        self.approximate_heat = approximate_heat
        self.stop_probability = stop_probability
        self.verbose = verbose

    def fit(self,X,y):
       
        self.X = check_array(X)
        self.y = np.array(y)

        classes = np.unique(y)
      
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        self.unique_classes_ = classes[indices]
        self.observation_dict = {c: X[y == c] for c in classes}
        self.maj_class_ = self.unique_classes_[0]
        n_max = max(sizes)
        if self.verbose:
            print(
              
                % (self.maj_class_, max(sizes)))


    def fit_sample(self, X, y):
      
        self.fit(X, y)

        for i in range(1, len(self.observation_dict)):
            current_class = self.unique_classes_[i]

         
            reshape_points, reshape_labels = self.reshape_observations_dict()
            oversampled_points, oversampled_labels = self.generate_samples(reshape_points, reshape_labels,
                                                                           current_class)
            self.observation_dict = {cls: oversampled_points[oversampled_labels == cls] for cls in self.unique_classes_}
     
        reshape_points, reshape_labels = self.reshape_observations_dict()
        return reshape_points, reshape_labels

    def generate_samples(self, X, y, minority_class=None):

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()
        self.n = len(majority_points) - len(minority_points)
      
        considered_minority_points_indices = range(len(minority_points))
        n_synthetic_points_per_minority_object = {i: 0 for i in considered_minority_points_indices}

       
        for _ in range(self.n):
            idx = np.random.choice(considered_minority_points_indices)
            n_synthetic_points_per_minority_object[idx] += 1
        appended = []
       
        for i in considered_minority_points_indices:
           
            if n_synthetic_points_per_minority_object[i] == 0:
                continue
          
            point = minority_points[i]

            if self.approximate_heat:
                distance_vector = [self.distance(point, x) for x in X]
                distance_vector[i] = -np.inf
                indices = np.argsort(distance_vector)[:(self.k + 1)]

            closest_points = X[indices]
            closest_labels = y[indices]
            closest_minority_points = closest_points[closest_labels == minority_class]
            closest_majority_points = closest_points[closest_labels != minority_class]

          
            for _ in range(n_synthetic_points_per_minority_object[i]):
               
                translation = [0 for _ in range(len(point))]
                translation_history = [translation]
              
                mutual_heat = self.heat_diference(point, closest_majority_points, closest_minority_points)
               
                possible_directions = self.generate_possible_directions(len(point))

                for i in range(self.n_steps):
                    if len(possible_directions) == 0:
                        break
                    if self.stop_probability is not None and self.stop_probability > np.random.rand():
                        break
                   
                    dimension, sign = possible_directions.pop()
                 
                    modified_translation = translation.copy()
                  
                    modified_translation[dimension] += sign * self.step
                  
                    modified_heat = self.heat_diference(point + modified_translation, closest_majority_points,
                                                                closest_minority_points)
                  
                    if np.abs(modified_heat) < np.abs(mutual_heat):
                        translation = modified_translation
                        translation_history.append(translation)
                        mutual_heat = modified_heat
                       
                        possible_directions = self.generate_possible_directions(len(point), (dimension, -sign))

                appended.append(point + translation)

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])
          
        return points, labels

    def nc(self,point1,point2):
        if self.alpha == 0.0:
            return 0.0
        else:
            return np.exp(- self.alpha * self.distance(point1, point2))

    def heat_diference(self, point, majority_points, minority_points):

        majority_center = np.mean(majority_points, axis=0)
        minority_center = np.mean(minority_points, axis=0)
        pos_heat = 0.0
        neg_heat = 0.0
        if len(minority_points) != 0:
            for majority_point in majority_points:
                pos_heat += 1/(self.distance(point, majority_point)+0.001)
        if len(minority_points) != 0:
            for minority_point in minority_points:
                neg_heat += 1/(self.distance(point, minority_point)+0.001)
        mutual_heat = pos_heat * self.nc(majority_center,point) - pos_heat * self.nc(minority_center,point)

        return mutual_heat

    def distance(self,x, y, p_norm=2):
        return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)

    def generate_possible_directions(self,n_dimensions, excluded_direction=None):
        possible_directions = []

        for dimension in range(n_dimensions):
            for sign in [-1, 1]:
                if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction[1] != sign):
                    possible_directions.append((dimension, sign))

        np.random.shuffle(possible_directions)

        return possible_directions

    def reshape_observations_dict(self):
      
        reshape_points = []
        reshape_labels = []

        for cls in self.observation_dict.keys():
            if len(self.observation_dict[cls]) > 0:
              
                reshape_points.append(self.observation_dict[cls])
               
                reshape_labels.append(np.tile([cls], len(self.observation_dict[cls])))

        reshape_points = np.concatenate(reshape_points)
        reshape_labels = np.concatenate(reshape_labels)

        return reshape_points, reshape_labels


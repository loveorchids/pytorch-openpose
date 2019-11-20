import os, math, random
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from joblib import dump, load
import pickle

class AngleEstimator():
    def __init__(self, phase="head", verbose=False):
        self.phase = phase
        self.verbose = verbose
        self.model_name = "./angle_estimator_model.sav"
        self.model = svm.SVC(kernel="poly", degree=10, gamma='scale', decision_function_shape='ovr')

        self.key_to_idx = {"0": 0, "00": 0, "15": 1, "30": 2, "45": 3, "60": 4, "90": 5}
        self.bad_sample = -1

        self.head_dis_pairs = [[15, 17], [14, 16], [1, 2], [1, 5], [8, 11]]
        self.head_angle_pairs = [[0, 1, 2], [0, 1, 5], [16, 14, 0], [14, 0, 15], [17, 15, 0]]
        self.low_conf_pairs = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]


    def cal_body_length(self, p_info, body_coord=([1, 2, 5], [4, 7, 8, 11])):
        if sum(abs(p_info[body_coord[0]])) == 0:
            return 0
        else:
            up_candidate = np.asarray([c for c in p_info[body_coord[0]] if c != 0])
            low_candidate = np.asarray([c for c in p_info[body_coord[1]] if c != 0])
            upper = np.min(up_candidate)
            lower = np.max(low_candidate)
            # because the image corrd is top-down manner
            # so the length = lower - upper
            # and in case the person has son strange pose, like upside down
            return abs(lower - upper)

    def cal_distance(self, p_info, coord_pairs, length):
        # coord_pairs = np.asarray(coord_pairs)
        dist = []
        for coord in coord_pairs:
            if sum(abs(p_info[coord[0]])) == 0 or sum(abs(p_info[coord[1]])) == 0 or length == 0:
                dist.append(self.bad_sample)
            else:
                dist.append(np.sqrt(sum((p_info[coord[0]] - p_info[coord[1]]) ** 2)) / length)
        return np.asarray(dist)

    def cal_angle(self, p_info, coord_pairs, bad_sample=-1):
        # coord_pairs = np.asarray(coord_pairs)
        angles = []
        for coord in coord_pairs:
            if sum(abs(p_info[coord[0]])) == 0 or \
                    sum(abs(p_info[coord[1]])) == 0 or \
                    sum(abs(p_info[coord[2]])) == 0:
                angles.append(bad_sample)
            else:
                v1 = p_info[coord[0]] - p_info[coord[1]]
                v2 = p_info[coord[2]] - p_info[coord[1]]
                angle = np.arccos(
                    np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                ) / math.pi
                angles.append(angle)
        return np.asarray(angles)

    def lowest_conf(self, p_info, coord_pairs):
        coord_pairs = np.asarray(coord_pairs)
        return np.min(p_info[coord_pairs], axis=1)

    def prepare_data(self, path, prefix=None):
        X = []
        Y = []
        if prefix is None:
            prefix = ""
        else:
            prefix = prefix + "_"
        raw_candidate = os.path.expanduser(os.path.join(path, "{}candidate.mat".format(prefix)))
        raw_subset = os.path.expanduser(os.path.join(path, "{}subset.mat".format(prefix)))
        candidates = sio.loadmat(raw_candidate)
        subsets = sio.loadmat(raw_subset)
        #p_count = {}
        if self.verbose:
            print("%d samples are loaded."%(len(candidates.keys()) - 3))
        for name in candidates.keys():
            if name in ['__header__', '__version__', '__globals__']:
                continue
            #p_id = int(name[:name.rfind("_")])
            angle = name[name.rfind("_")+1:]
            subset = subsets[name]
            if len(subset) > 1:
                continue
            candidate = candidates[name]

            index = subset[0, :18].astype(int)
            p_info = candidate[index][:, :-1]

            # Calculate feature
            h_wid = self.cal_body_length(p_info[:, 0], body_coord=([0, 14, 15, 16, 17], [0, 14, 15, 16, 17]))
            h_dis = self.cal_distance(p_info[:, :2], self.head_dis_pairs, h_wid)
            h_ang = self.cal_angle(p_info[:, :2], self.head_angle_pairs)
            #joint_conf = p_info[:, 2][np.asarray(range(2, 18))]
            joint_conf = p_info[:, 2][np.asarray(range(2, 18))] # increase 1.3% for 6-class
            low_conf = self.lowest_conf(p_info[:, 2], self.low_conf_pairs)
            feature = np.concatenate([h_dis, h_ang, low_conf, joint_conf])
            X.append(feature)
            Y.append(self.key_to_idx[angle])
        X = np.stack(X, axis=0)
        Y = np.asarray(Y)
        if self.verbose:
            print("data set shape: %s"%str(X.shape))
            print("ground truth shape: %s" % str(Y.shape))
        return X, Y

    def load_model(self):
        #self.clf = load(self.model_name)
        self.model = pickle.load(open(self.model_name, 'rb'))

    def __call__(self, X, Y=None):
        if Y is None:
            return self.model.predict(X)
        else:
            return self.model.score(X, Y)

    def train(self, X, Y, cv=1, test_split=0.1, test_set=None):
        if test_set is not None:
            self.model.fit(X, Y)
            score = self.model.score(test_set[0], test_set[1])
        else:
            _cv = ShuffleSplit(n_splits=cv, test_size=test_split)
            scores = cross_val_score(self.model, X, Y, cv=_cv)
            score = sum(scores) / len(scores)
        #dump(self.clf, self.model_name)
        #pickle.dump(self.model, open(self.model_name, 'wb'))
        return score


if __name__ == '__main__':
    path = os.path.expanduser("~/Pictures/dataset/reid/")
    angle_estimator = AngleEstimator()
    #X, Y = angle_estimator.prepare_data(prefix="raw")
    test_set = angle_estimator.prepare_data(path)
    #scores = angle_estimator.train(X, Y, cv=10, test_split=0.1)
    #print("Avg score is: %.2f"%(scores * 100))
    angle_estimator.load_model()
    pred = angle_estimator(test_set[0])
    print(pred)



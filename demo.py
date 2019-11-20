import sys, glob, os, random, math, time
sys.path.insert(0, 'python')
import cv2
import python.model as model
import torch
import torch.nn as nn
import python.util as util
from python.hand import Hand
from python.body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.io as sio

body_estimation = Body('model/body_pose_model.pth')
#hand_estimation = Hand('model/hand_pose_model.pth')

img_h = 800
crop_size = 60

cam_folder = os.path.expanduser("~/Pictures/dataset/reid/RPIfield_v0/Data/Cam_*")
eval_folder = os.path.expanduser("~/Pictures/dataset/reid/EVAL_DS/*")
destiny = os.path.expanduser("~/Pictures/dataset/reid/OP_local_patch")
eval_destiny = os.path.expanduser("~/Pictures/dataset/reid/eval_lp")
mat_dir = os.path.expanduser("~/Pictures/dataset/reid")


def crop_from_img(img, coords, crop_size, save_path):
    cps = int(crop_size / 2)
    for joint_id in coords.keys():
        x, y = coords[joint_id]
        x = int(x)
        y = int(y)
        patch = img[y - cps: y + cps, x - cps: x + cps, :]
        cv2.imwrite(os.path.join(save_path, "%s.jpg"%str(joint_id).zfill(2)), patch)

def create_local_patch():
    for i, camera in enumerate(sorted(glob.glob(cam_folder))):
        camera_id = camera[camera.rfind("/") + 1:]
        if not os.path.exists(os.path.join(destiny, camera_id)):
            os.mkdir(os.path.join(destiny, camera_id))
        for j, person in enumerate(sorted(glob.glob(camera + "/*"))):
            print("camera %d, person %d"%(i, j))
            person_id = person[person.rfind("/") + 1:]
            if not os.path.exists(os.path.join(destiny, camera_id, person_id)):
                os.mkdir(os.path.join(destiny, camera_id, person_id))
            person_imgs = sorted(glob.glob(person + "/*.png"))
            person_imgs = random.sample(person_imgs, int(len(person_imgs) / 2))
            person_imgs.sort()
            for k, person_img in enumerate(person_imgs):
                pic_id = person_img[person_img.rfind("/")+1:-4]
                save_path = os.path.join(destiny, camera_id, person_id, pic_id)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                oriImg = cv2.imread(person_img)  # B,G,R order
                oriImg = cv2.resize(oriImg, (int(oriImg.shape[1] / oriImg.shape[0] * img_h / 8) * 8, img_h))
                try:
                    candidate, subset = body_estimation(oriImg)
                except ZeroDivisionError:
                    continue
                #canvas = copy.deepcopy(oriImg)
                oriImg, coords = util.draw_bodypose(oriImg, candidate, subset, stickwidth=2, crop_size=crop_size)
                if coords is None:
                    os.system("rm -r %s"%save_path)
                    continue
                else:
                    crop_from_img(oriImg, coords, crop_size, save_path)

def Eval_local_patch(source, destiny):
    for i, camera in enumerate(sorted(glob.glob(source))):
        camera_id = camera[camera.rfind("/") + 1:]
        if not os.path.exists(os.path.join(destiny, camera_id)):
            os.mkdir(os.path.join(destiny, camera_id))
        for j, person in enumerate(sorted(glob.glob(camera + "/*"))):
            print("camera %d, person %d"%(i, j))
            person_id = person[person.rfind("/") + 1:]
            if not os.path.exists(os.path.join(destiny, camera_id, person_id)):
                os.mkdir(os.path.join(destiny, camera_id, person_id))
            person_imgs = sorted(glob.glob(person + "/*.png"))
            #person_imgs = random.sample(person_imgs, len(person_imgs))
            #person_imgs.sort()
            for k, person_img in enumerate(person_imgs):
                pic_id = person_img[person_img.rfind("/")+1:-4]
                save_path = os.path.join(destiny, camera_id, person_id, pic_id)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                oriImg = cv2.imread(person_img)  # B,G,R order
                oriImg = cv2.resize(oriImg, (int(oriImg.shape[1] / oriImg.shape[0] * img_h / 8) * 8, img_h))
                try:
                    candidate, subset = body_estimation(oriImg)
                except ZeroDivisionError:
                    continue
                #canvas = copy.deepcopy(oriImg)
                oriImg, coords = util.draw_bodypose(oriImg, candidate, subset, stickwidth=2, crop_size=crop_size)
                if coords is None:
                    os.system("rm -r %s"%save_path)
                    continue
                else:
                    crop_from_img(oriImg, coords, crop_size, save_path)

def demo():
    for i, person_img in enumerate(sorted(glob.glob("./images/*.jpg"))):
        print(i)
        # Skip some frames
        img = cv2.imread(person_img)  # B,G,R order
        img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * img_h / 8) * 8, img_h))
        try:
            candidate, subset = body_estimation(img)
        except ZeroDivisionError:
            continue
        canvas = copy.deepcopy(img)
        canvas, coords = util.draw_bodypose(canvas, candidate, subset, stickwidth=2, crop_size=crop_size, draw=True)
        cv2.imwrite(os.path.expanduser("~/Pictures/tmp_%s.jpg"%str(i).zfill(2)), canvas)

def create_facing_mat(mat_dir):
    head_train = {}
    body_train = {}
    body_dis_pairs = [[1, 2], [1, 5], [8, 11]]
    head_dis_pairs = [[15, 17], [14, 16]]
    body_angle_pairs = [[0, 1, 2], [0, 1, 5]]
    head_angle_pairs = [[0, 1, 2], [0, 1, 5], [16, 14, 0], [14, 0, 15], [17, 15, 0]]
    body_low_conf_pairs = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]

    for i, person_img in enumerate(sorted(glob.glob("./images/*.jpg"))):
        name = person_img[person_img.rfind("/")+1:-4]
        p_id = int(name[:2])
        angle = name[3:]
        print(p_id, angle)
        img = cv2.imread(person_img)  # B,G,R order
        img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * img_h / 8) * 8, img_h))
        try:
            candidate, subset = body_estimation(img)
        except ZeroDivisionError:
            continue
        # The first dim of subset represent the person in the pictures
        # The second dim of sebset represent the joint of that person
        assert len(subset) == 1
        p_info = []
        for i in range(18):
            index = int(subset[0][i])
            # print(index)
            if index == -1:
                p_info.append(np.asarray([0, 0, 0], dtype=float))
            else:
                p_info.append(candidate[index][0:3])
        p_info = np.stack(p_info, axis=0)

        _index = subset[0, :18].astype(int)
        _candidate = np.concatenate([candidate, np.zeros([1, 4])], axis=0)
        residual = np.sum(p_info - _candidate[_index][:, :-1])
        if residual != 0:
            xxx=0

        # Calculate body feature
        b_length = cal_body_length(p_info[:, 1])
        b_dis = cal_distance(p_info[:, :2], body_dis_pairs, b_length)
        b_ang = cal_angle(p_info[:, :2], body_angle_pairs)
        low_conf = lowest_conf(p_info[:, 2], body_low_conf_pairs)
        body_feature = np.concatenate([b_dis, b_ang, low_conf])
        print(body_feature)
        if angle in body_train:
            body_train[angle].append(body_feature)
        else:
            body_train.update({angle: [body_feature]})

        # Calculate head feature
        h_wid = cal_body_length(p_info[:, 0], body_coord=([0, 14, 15, 16, 17], [0, 14, 15, 16, 17]))
        h_dis = cal_distance(p_info[:, :2], head_dis_pairs, h_wid)
        h_ang = cal_angle(p_info[:, :2], head_angle_pairs)
        joint_conf = p_info[:, 2][np.asarray([14, 15, 16, 17])]
        head_feature = np.concatenate([h_dis, h_ang, joint_conf])
        print(head_feature)
        print("-------------------")

        if angle in head_train:
            head_train[angle].append(head_feature)
        else:
            head_train.update({angle: [head_feature]})

    # Tranform the list into numpy array
    body_train_mat = to_array(body_train)
    head_train_mat = to_array(head_train)
    sio.savemat(mat_dir + "/body_facing.mat", body_train_mat)
    sio.savemat(mat_dir + "/head_facing.mat", head_train_mat)

def create_raw_mat(mat_dir, prefix):
    if prefix is not None:
        prefix = ""
    else:
        prefix = prefix + "_"
    raw_candidate = {}
    raw_subset = {}
    for i, person_img in enumerate(sorted(glob.glob("./images/*.jpg"))):
        if i % 20 == 0:
            print(i)
        name = person_img[person_img.rfind("/")+1:-4]
        #p_id = int(name[:2])
        #angle = name[3:]
        #print(p_id, angle)
        img = cv2.imread(person_img)  # B,G,R order
        img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * img_h / 8) * 8, img_h))
        try:
            candidate, subset = body_estimation(img)
        except ZeroDivisionError:
            continue
        # The first dim of subset represent the person in the pictures
        # The second dim of sebset represent the joint of that person
        if len(subset) > 1:
            print("We happen to find 2 or more persons")
            continue
        candidate = np.concatenate([candidate, np.zeros([1, 4])], axis=0)
        raw_candidate.update({name: candidate})
        raw_subset.update({name: subset})

    # Tranform the list into numpy array
    #raw_candidate = to_array(raw_candidate)
    #raw_subset = to_array(raw_subset)
    sio.savemat(mat_dir + "/{}candidate.mat".format(prefix), raw_candidate)
    sio.savemat(mat_dir + "/{}subset.mat".format(prefix), raw_subset)

def to_array(mat_dict):
    new_dict = {}
    for key in mat_dict.keys():
        new_dict.update({key: np.stack(mat_dict[key], axis=0)})
    return new_dict

def cal_body_length(p_info, body_coord=([1, 2, 5], [4, 7, 8, 11])):
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

def cal_distance(p_info, coord_pairs, length, bad_sample=-1):
    #coord_pairs = np.asarray(coord_pairs)
    dist = []
    for coord in coord_pairs:
        if sum(abs(p_info[coord[0]])) == 0 or sum(abs(p_info[coord[1]])) == 0 or length == 0:
            dist.append(bad_sample)
        else:
            dist.append(np.sqrt(sum((p_info[coord[0]] - p_info[coord[1]]) ** 2)) / length)
    return np.asarray(dist)

def cal_angle(p_info, coord_pairs, bad_sample=-1):
    #coord_pairs = np.asarray(coord_pairs)
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

def lowest_conf(p_info, coord_pairs):
    coord_pairs = np.asarray(coord_pairs)
    return np.min(p_info[coord_pairs], axis=1)

def test_parallel():
    img = cv2.imread("./images/0002_0.jpg")
    img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * img_h / 8) * 8, img_h))
    img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).repeat(4, 1, 1, 1)
    for i in range(100):
        start = time.time()
        body_estimation(img)
        print(time.time() - start)




if __name__ == '__main__':
    #Eval_local_patch(eval_folder, eval_destiny)
    #create_facing_mat(mat_dir)
    demo()
    #create_raw_mat(mat_dir, prefix="test")
    #test_parallel()


"""
hands_list = util.handDetect(candidate, subset, oriImg)

all_hand_peaks = []
for x, y, w, is_left in hands_list:
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
"""



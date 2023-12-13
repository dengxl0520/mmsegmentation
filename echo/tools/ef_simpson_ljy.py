import os
import cv2 
import csv
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations


def NearestVertex(trg, points):
    point_set = []
    dis0 = np.reshape(trg,(1,-1,2)).repeat(points.shape[0], axis=0)
    dis1 = np.reshape(points,(-1,1,2)).repeat(3, axis=1)
    dis = np.linalg.norm(dis0 - dis1, axis=-1, keepdims=False)
    for i in range(dis.shape[1]):
        point_set.append(points[dis[:,i].argmin()])
    return point_set


def ExtensionLine(line1, line2, points):
    # if line1[1] > line2[1]:
    #     line1, line2 = line2, line1

    line_vec = line1 - line2
    line_vec = np.reshape(line_vec,(1,1,2)).repeat(points.shape[0],axis=0)

    line2_s = np.reshape(line2,(1,1,2)).repeat(points.shape[0],axis=0)
    pline_vec = line2_s - points
    
    prod = line_vec[:,:,0] * pline_vec[:,:,1] - line_vec[:,:,1] * pline_vec[:,:,0]
    prod = np.abs(prod)

    inc = np.argsort(prod, axis=0)
    point_set = [line1]
    for i in inc[1:]:
        if (line1[1] - line2[1]) * (points[i[0],0,1] - line2[1]) < 0:
            point_set.append(points[i[0],0])
            break
    if len(point_set) == 1:
        point_set.append(line2)
        
    return point_set


def get_line(line1, line2, p1, points):
    if line1[1] > line2[1]:
        line1, line2 = line2, line1

    line_vec = line1 - line2
    line_vec = np.reshape(line_vec,(1,1,2)).repeat(points.shape[0],axis=0)

    p1_s = np.reshape(p1,(1,1,2)).repeat(points.shape[0],axis=0)
    vert_vec = p1_s - points
    
    prod = vert_vec * line_vec
    prod = prod[:,:,0] + prod[:,:,1]
    prod = np.abs(prod)

    inc = np.argsort(prod, axis=0)
    point1 = points[inc[0,0]][0]
    point_set = [point1]
    for i in inc[1:]:
        if (point1[0] - p1[0]) * (points[i[0],0,0] - p1[0]) < 0:
            point_set.append(points[i[0],0])
            break
    
    return point_set


def get_info(img_arr, save_path):
    img_result = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours_, _ = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    if len(contours) > 1:
        idx = 0
        nmax = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > nmax:
                idx = i
                nmax = contours[i].shape[0]
        contours = contours[idx:idx+1] 

    if len(contours_) > 1:
        idx = 0
        nmax = 0
        for i in range(len(contours_)):
            if contours_[i].shape[0] > nmax:
                idx = i
                nmax = contours_[i].shape[0]
        contours_ = contours_[idx:idx+1]  

    # Inscribed and circumscribed triangle
    _, out_trg = cv2.minEnclosingTriangle(contours_[0])
    in_trg = NearestVertex(out_trg, contours[0])

    for c in combinations(out_trg, 2):
        start_p = [int(x) for x in c[0][0]]
        end_p = [int(x) for x in c[1][0]]
        cv2.line(img_result, start_p, end_p,color=(255,0,0))

    for c in combinations(in_trg, 2):
        start_p = [int(x) for x in c[0][0]]
        end_p = [int(x) for x in c[1][0]]
        cv2.line(img_result, start_p, end_p,color=(0,255,0))

    # Midpoint of circumscribed triangle
    out_mid = [np.mean(x, axis=0) for x in combinations(out_trg, 2)]
    
    # Euclidean distance from each midpoint to each inscribed triangle vertex
    dis0 = np.reshape(in_trg,(1,3,2)).repeat(3,axis=0)
    dis1 = np.reshape(out_mid,(3,1,2)).repeat(3,axis=1)
    dis = np.linalg.norm(dis0-dis1,axis=-1,keepdims=False)

    # Find the longest distance 
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(dis)
    L_p = [in_trg[max_indx[0]][0],out_mid[max_indx[1]][0]]
    # L_p = ExtensionLine(L_p[0], L_p[1], contours[0])

    start_p = [int(x) for x in L_p[0]]
    end_p = [int(x) for x in L_p[1]]
    cv2.line(img_result, start_p, end_p,color=(0,0,255))

    # Trisection point
    div_p1 = (2*L_p[0] + L_p[1])/3.0
    div_p2 = (L_p[0] + 2*L_p[1])/3.0

    # Endpoints of vertical line through trisection point
    div_l1 = get_line(L_p[0], L_p[1], div_p1, contours[0])
    div_l2 = get_line(L_p[0], L_p[1], div_p2, contours[0])

    start_p = [int(x) for x in div_l1[0]]
    end_p = [int(x) for x in div_l1[1]]
    cv2.line(img_result, start_p, end_p,color=(0,0,255))

    start_p = [int(x) for x in div_l2[0]]
    end_p = [int(x) for x in div_l2[1]]
    cv2.line(img_result, start_p, end_p,color=(0,0,255))

    cv2.imwrite(save_path, img_result)
    # plt.imshow(img_result)
    # plt.show()

    return out_trg, in_trg, contours, L_p, [div_l1, div_l2]


def get_volume(img_arr, save_path):
    info = get_info(img_arr, save_path)
    div_line1, div_line2 = info[-1]
    L = info[-2][0] - info[-2][1]
    dis1 = div_line1[0] - div_line1[1]
    dis2 = div_line2[0] - div_line2[1]

    # Trisecting surface radius
    dis1 = np.linalg.norm(dis1,axis=-1,keepdims=False) / 2.0
    dis2 = np.linalg.norm(dis2,axis=-1,keepdims=False) / 2.0
    if dis1 > dis2:
        dis1, dis2 = dis2, dis1

    # LV length 
    L = np.linalg.norm(L,axis=-1,keepdims=False)
    L = L / 10.

    dis1 = dis1 / 10.
    dis2 = dis2 / 10.

    ap = np.pi * dis1 * dis1
    am = np.pi * dis2 * dis2


    v1 = am * L / 3
    v2 = (am + ap) / 2 * L / 3
    v3 = ap * L / 3 / 3
    v = v1 + v2 + v3

    return v


def get_LVEF(result_dir, resolutions=None):
    save_dir = result_dir + "_plt"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    csv_path = result_dir + ".csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    csvfile = open(csv_path, 'a', newline='')
    csv_write = csv.writer(csvfile, dialect='excel')
    csv_write.writerow(["patient", "Predict LVEDV", "Predict LVESV", "Predict LVEF"])

    patient_list = os.listdir(result_dir)
    with tqdm(patient_list) as pbar:
        for patient_name in patient_list:
            # if patient_name == "0X33EAE0F44B7618C1.avi":
            #     csv_write.writerow([patient_name, None, None, None])
            #     pbar.update()
            #     continue
            # print(patient_name)
            patient_dir = os.path.join(result_dir, patient_name)
            img_ed = cv2.imread(os.path.join(patient_dir, "0.png"), 0)
            img_es = cv2.imread(os.path.join(patient_dir, "9.png"), 0)

            if resolutions is not None:
                img_ed = cv2.resize(img_ed, resolutions[patient_name])
                img_es = cv2.resize(img_es, resolutions[patient_name])

            # img_ed[img_ed == 1] = 255
            # img_es[img_es == 1] = 255

            # img_size = (int(size_dict[patient_name][0]), int(size_dict[patient_name][1]))

            # img_ed = cv2.resize(img_ed, img_size, interpolation=cv2.INTER_NEAREST)
            # img_es = cv2.resize(img_es, img_size, interpolation=cv2.INTER_NEAREST)

            # print(patient_name + "-ED")
            save_ed = os.path.join(save_dir, patient_name + "_0.png")
            # print(patient_name + "-ES")
            save_es = os.path.join(save_dir, patient_name + "_9.png")

            edv = get_volume(img_ed, save_ed)
            esv = get_volume(img_es, save_es)
            ef = (edv - esv) / edv * 100

            csv_write.writerow([patient_name, edv, esv, ef])
            pbar.update()
    
    csvfile.close()


if __name__ == "__main__":
    # img_ed = cv2.imread("D:\\Workspace\\AAAI2023\\result\\echo\\echo\\0X10A5FC19152B50A5.avi\\0.png", 0)
    # img_es = cv2.imread("D:\\Workspace\\AAAI2023\\result\\echo\\echo\\0X10A5FC19152B50A5.avi\\1.png", 0)
    
    # img_ed[img_ed == 1] = 255
    # img_es[img_es == 1] = 255

    # img_ed = cv2.resize(img_ed,(169,206), interpolation=cv2.INTER_NEAREST)
    # img_es = cv2.resize(img_es,(169,206), interpolation=cv2.INTER_NEAREST)

    # edv = get_volume(img_ed, "D:\\Workspace\\AAAI2023\\result\\echo\\echo_plt\\0X10A5FC19152B50A5.avi_0.png")
    # esv = get_volume(img_es, "D:\\Workspace\\AAAI2023\\result\\echo\\echo_plt\\0X10A5FC19152B50A5.avi_1.png")
    # ef = (edv - esv) / edv * 100
    # print(edv, esv, ef)

    # dataset = "camus"
    dataset = "echo"

    # dir_name = "label"
    # dir_name = "CLAS"
    # dir_name = "HCPN"
    # dir_name = "MBEchoNet"
    # dir_name = "PKEchoNet"
    # dir_name = "PLANet"
    # dir_name = "SOCOF"
    # dir_name = "SSCFNet"
    dir_name = "STCN"

    resolutions = None
    # resolutions = {}
    # with open("D:\\Workspace\\TMI2023\\LVEF\\"+dataset+"\\resolution_1x1.csv", "r") as csvfile:
    #     csvreader = csv.reader(csvfile)
    #     for i, row in enumerate(csvreader):
    #         if i > 0:
    #             resolutions[row[0]] = (int(row[1]), int(row[2]))

    get_LVEF("D:\\Workspace\\TMI2023\\visualization\\LVEF\\"+dataset+"\\"+dir_name, resolutions)
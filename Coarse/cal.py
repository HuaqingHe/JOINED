import csv
from numpy import mean, std
import os
import pandas as pd


save_path = './results/paper_results/seg_best/dataset2/'

# labels=[]
# preds=[]
dice_1os = []
dice_2os = []
jaccard_1os = []
jaccard_2os = []
HD_os = []
ASSD_os = []
# probs = []

for i in range(1, 6):

    file = save_path + str(i) + '_' + 'class&seg.csv'
    print(file)
    # with open(file, 'rt') as csvfile:
    #     reader = csv.reader(csvfile, skipinitialspace=True)
    #     label = ([row[2] for row in reader][1:])
    #     label = list(map(int, label))
    # with open(file, 'rt') as csvfile:
    #     reader = csv.reader(csvfile, skipinitialspace=True)
    #     pred = ([row[3] for row in reader][1:])
    #     pred = list(map(int, pred))
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        dice_1 = ([row[4] for row in reader][1:])
        dice_1 = list(map(float, dice_1))
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        dice_2 = ([row[5] for row in reader][1:])
        dice_2 = list(map(float, dice_2))
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        jaccard_1 = ([row[6] for row in reader][1:])
        jaccard_1 = list(map(float, jaccard_1))
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        jaccard_2 = ([row[7] for row in reader][1:])
        jaccard_2 = list(map(float, jaccard_2))
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        HD_o = ([row[8] for row in reader][1:])
        HD_o = list(map(float, HD_o))
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        ASSD_o = ([row[9] for row in reader][1:])
        ASSD_o = list(map(float, ASSD_o))
    # with open(file, 'rt') as csvfile:
    #     reader = csv.reader(csvfile, skipinitialspace=True)
    #     prob = ([row[1] for row in reader][1:])
    #     prob = list(map(float, prob))

    # labels += label
    # preds += pred
    dice_1os += dice_1
    dice_2os += dice_2
    jaccard_1os += jaccard_1
    jaccard_2os += jaccard_2
    HD_os += HD_o
    ASSD_os += ASSD_o
    # probs += prob

# print(labels)
# print(preds)
# a = len(np.unique(labels))

# kappa = cohen_kappa_score(labels, preds)
# acc = accuracy_score(labels, preds)
# recall = recall_score(labels, preds, average='micro')
# f1 = f1_score(labels, preds, average='weighted')
# c_matrix = confusion_matrix(labels, preds)
# if a == 3:
#     target_names = ['N', 'D', 'M']
#     clas_report = classification_report(labels, preds, target_names=target_names, digits=5)
# elif a ==2:
#     target_names = ['N', 'D']
#     clas_report = classification_report(labels, preds, target_names=target_names, digits=5)

resultframe = pd.DataFrame({'seg_dice1': mean(dice_1os), 'seg_dice2': mean(dice_2os), 'jaccard1': mean(jaccard_1os), 'jaccard2': mean(jaccard_2os), 'HD': mean(HD_os), 'ASSD': mean(ASSD_os)}, index=[1])
resultframe.to_csv(save_path + "/" + "all_acc_kappa.csv", index=0)

# with open(save_path + "/" + "all_cmatrix.txt","w") as f:
#     f.write(str(c_matrix))
# with open(save_path + "/" + "all_clas_report.txt","w") as f:
#     f.write(str(clas_report))

with open(os.path.join(save_path, "all_seg_report.txt"), "w") as f:
    f.write('Dice 1: ' + str(mean(dice_1os)) + ' + ' + str(std(dice_1os)) + '\r\n')
    f.write('Jaccard 1 : ' + str(mean(jaccard_1os)) + ' + ' + str(std(jaccard_1os)) + '\r\n')
    f.write('Dice 2 : ' + str(mean(dice_2os)) + ' + ' + str(std(dice_2os)) + '\r\n')
    f.write('Jaccard 2 : ' + str(mean(jaccard_2os)) + ' + ' + str(std(jaccard_2os)) + '\r\n')
    f.write('HD : ' + str(mean(HD_os)) + ' + ' + str(std(HD_os)) + '\r\n')
    f.write('ASSD : ' + str(mean(ASSD_os)) + ' + ' + str(std(ASSD_os)) + '\r\n')

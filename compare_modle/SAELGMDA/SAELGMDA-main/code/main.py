# -*- coding: utf-8 -*-
import numpy as np
from utils import *
from model import *
import sys
import codecs
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

parser = argparse.ArgumentParser(description='my model')

parser.add_argument("--cvs", default=1, type=int, choices=[1,2,3])
parser.add_argument("--dataset_name", default="HMDAD", type=str, choices=["HMDAD", "Disbiome"])
options, other_args = parser.parse_known_args()

# 一般情况下采用该代码能够解决问题，可以发现函数返回结果已经转化成tensorflow.python.framework.ops.Tensor类了
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# cv1 is disease and cv2 is microbe
if __name__ == "__main__":
    all_performance_list = []
    params = [0.1, 0.0005, 0.05]

    for i in range(1):
        start = datetime.datetime.now()
        data, label = Sparse_Autoencoder(dataset_name=options.dataset_name,params=params)
        all_performance = Cross_Validation(data, label,dataset_name=options.dataset_name, cv=options.cvs)
        all_performance_list.append(all_performance)
        end = datetime.datetime.now()
        print("Iterate",i+1,"times is",end - start,"")

        Mean_Result = np.around(np.mean(all_performance_list, axis=(0, 1)), 4)
        Stan_Result = np.around(np.std(all_performance_list, axis=(0, 1)), 4)

        print('Mean-Acc=', Mean_Result[0], '±', format(Stan_Result[0], '.4f'), '\nMean-pre=', Mean_Result[1], '±', format(Stan_Result[1], '.4f'))
        print('Mean-Rec=', Mean_Result[2], '±', format(Stan_Result[2], '.4f'), '\nMean-Spe=', Mean_Result[3], '±', format(Stan_Result[3], '.4f'))
        print('Mean_F1=', Mean_Result[7], '±', format(Stan_Result[7], '.4f'), '\nMean-MCC=', Mean_Result[4], '±', format(Stan_Result[4], '.4f'))
        print('Mean-auc=', Mean_Result[5], '±', format(Stan_Result[5], '.4f'), '\nMean-Aupr=', Mean_Result[6], '±', format(Stan_Result[6], '.4f'))
        print('---' * 20)
        print('\n \n')

    print("end")
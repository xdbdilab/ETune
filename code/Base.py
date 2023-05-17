import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from sklearn.preprocessing import StandardScaler
import json
from collections import *
from tool import *
import operator
from tensorboardX import SummaryWriter
from scipy.stats import norm
# from tqdm import trange
from sklearn.metrics import mean_absolute_error
from DeepPerf.mlp_sparse_model import *
import matplotlib.pyplot as plt
from tqdm import *


class random_forest(object):
    def __init__(self,task_name):

        self.confs = json.load(open('paras.json'), strict=False)

        self.selected_params = self.confs['common_params']['all_params']
        self.system = self.confs['common_params']['systerm']


         self.task_name = task_name

        self.is_EI=self.confs['is_EI']

        self.performance = self.confs['common_params']['performance'].split('_')[0]

        self.min_or_max = self.confs['common_params']['performance'].split('_')[1]

        self.model = None

        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_output')):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_output'))
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'target_output/random_forest.pickle')

        self.import_feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'target_output/random_forest_feature_importance.txt')

        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'data/target_data/{}.csv'.format(self.task_name))

        self.save_filesname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'target_output/')

        self.columns = None
        self.fake_columns = None

        self.scalar = None

        self.data_type={"executorCores": int,
        "executorMemory": int,
        "executorInstances": int,
        "defaultParallelism": int,
        "memoryOffHeapEnabled": str,
        "memoryOffHeapSize": int,
        "memoryFraction": float,
        "memoryStorageFraction": float,
        "shuffleFileBuffer":int,
        "speculation": str,
        "reducerMaxSizeInFlight": int,
        "shuffleSortBypassMerageThreshold": int,
        "speculationInterval": int,
        "speculationMultiplier": float,
        "speculationQuantile": float,
        "broadcastBlockSize": int,
        "ioCompressionCodec": str,
        "ioCompressionLz4BlockSize": int,
        "ioCompressionSnappyBlockSize": int,
        "kryoRederenceTracking": str,
        "kryoserializerBufferMax": int,
        "kryoserializerBuffer": int,
        "storageMemoryMapThreshold": int,
        "networkTimeout": int,
        "localityWait": int,
        "shuffleCompress": str,
        "shuffleSpillCompress": str,
        "broadcastCompress": str,
        "rddCompress": str,
        "serializer": str}


    def data_in(self, data):
        self.columns = list(data.columns)
        columns = self.columns

        char = [] 
        enum_index = []
        for name in columns:
            if self.selected_params[name][0] == 'string' or self.selected_params[name][0] == 'enum':
                char.append(name)
                enum_index.append(columns.index(name))

        enum_number = []  
        enum_book = {}

        m = 0
        for c in char:
            i = enum_index[m]

            new_data = pd.DataFrame({c: self.selected_params[c][1]})  
            data = data.append(new_data, ignore_index=True)
            enum_data = pd.get_dummies(data[c], prefix=c) 
            enum_book[c] = list(enum_data.columns)
            # print(enum_data.columns)
            data = data.drop(c, 1)

            enum_list = list(enum_data.columns)
            enum_number.append(len(enum_list))

            for k in range(len(enum_list)):
                data.insert(i + k, enum_list[k], enum_data[enum_list[k]]) 
            m = m + 1
            enum_index = [j + len(enum_data.columns) - 1 for j in enum_index] 

            data.drop(data.index[-len(self.selected_params[c][1]):], inplace=True) 

        self.fake_columns = list(data.columns)

        return data

    def data_out(self, data, enum_number, enum_book):
        """
        :param enum_number:
        :param enum_book:
        :return:
        """
        if data.empty:
            return pd.DataFrame(columns=self.columns[:-1])
        data.columns = [i for i in range(len(data.columns))]
        hang = data.iloc[:, 0].size

        m = 0  
        index_enum_number = 0 
        enum_list = []  
        for name in self.columns[:-1]:
            if self.selected_params[name][0] == 'int':
                data[m] = data[m].astype(np.int64)
                m = m + 1
            elif self.selected_params[name][0] == 'float' or self.selected_params[name][0] == 'double':
                m = m + 1
                continue
            else:
                enum_list.append(name)
                for i in range(enum_number[index_enum_number]):
                    data[m + i] = data[m + i].round().astype(np.int64)
                m = m + enum_number[index_enum_number]
                index_enum_number = index_enum_number + 1
        # print(data)

        for i in enum_list:
            data_temp = pd.DataFrame(columns=[i])
            data = pd.concat([data, data_temp], 1)

        for index in range(hang):
            for k in range(len(enum_list)):

                name_index = self.columns.index(enum_list[k])  
                if k != 0:
                    for a in range(k):
                        name_index = name_index + enum_number[a] - 1 
                # print(name_index)

                flag = True
                true_index = False
                first_index = -1
                for i in range(enum_number[k]):

                    for j in range(i, enum_number[k]):
                        if i == j:

                            if round(data.loc[[index]].values[0][name_index + j]) == 1:
                                first_index = j
                                continue
                            else:
                                if j == enum_number[k] - 1:
                                    flag = False
                                break
                        else:
                            if round(data.loc[[index]].values[0][name_index + j]) == 0:

                                if j == enum_number[k] - 1:
                                    true_index = True
                                    break
                                else:
                                    continue
                            else:
                                flag = False
                                break

                    if flag == False:
                        break
                    if true_index == True:
                        break
                if flag == False:
                    data.drop([index], inplace=True)
                    break
                if (first_index + 1) != 0 and flag == True:
                    data.loc[index, enum_list[k]] = enum_book[enum_list[k]][first_index]
        # print(data)

        number = 0
        for i in range(len(enum_list)):
            name_index = number + self.columns.index(enum_list[i])

            for j in range(enum_number[i]):
                data = data.drop([name_index + j], 1)

            number = number + enum_number[i] - 1
        # print(data)

        for i in range(len(enum_list)):
            orgin_index = self.columns.index(enum_list[i])
            data.insert(orgin_index, enum_list[i], data.pop(enum_list[i]))
        # print(data)

        col = {}
        for (key, value) in zip(list(data.columns), self.columns[:-1]):
            col[key] = value
        # print(col)
        data.rename(columns=col, inplace=True)
        # print(data)
        return data

    def data_range(self, data, enum_number):
        data = data.values
        MAX_BOUND = []
        MIN_BOUND = []
        m = 0
        for name in self.columns[:-1]:
            if self.selected_params[name][0] == 'enum' or self.selected_params[name][0] == 'string':
                for i in range(enum_number[m]):
                    MIN_BOUND.append(0)
                    MAX_BOUND.append(1)
                m = m + 1

            else:
                MIN_BOUND.append(self.selected_params[name][1][0])
                MAX_BOUND.append(self.selected_params[name][1][1])
        MIN_BOUND = [i - 1 for i in MIN_BOUND]
        MAX_BOUND = [i + 1 for i in MAX_BOUND]

        result = []
        for each in data:
            flag = True
            for j in range(len(MIN_BOUND)):
                if each[j] <= MIN_BOUND[j] or each[j] >= MAX_BOUND[j]:
                    flag = False
                    break
            if flag:
                result.append(each)

        return pd.DataFrame(np.array(result))

    def one_step_train(self, data):
        data_y = data.iloc[:, -1].values
        data_x= self.data_in(data.iloc[:,:-1])
        self.scalar = StandardScaler()
        data_x = self.scalar.fit_transform(data_x.astype(float))

        self.model = RandomForestRegressor()
        self.model.fit(data_x, data_y)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def target_train(self):

        origin_data = pd.read_csv(self.filename,dtype=self.data_type)
        feature_sample_num = self.confs['sample_params']['feature_sample_num']
        temp_data_real_perf_min = pd.DataFrame(columns=origin_data.columns) 
        eta = max(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else min(origin_data.iloc[:, -1].values) 
        eta_index=-np.argmax(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else -np.argmin(origin_data.iloc[:, -1].values)

        writer = SummaryWriter(self.save_filesname+"runs/")

        for i in trange(self.confs['sample_params']['sample_epoch']):
            self.one_step_train(origin_data)
            sample_data = random_sample_1(self.selected_params,feature_sample_num)
            sample_data = sample_data.reset_index(drop=True)

            ei = self.get_ei(self.model,eta,sample_data)
            max_index = np.argmax(ei)

            top_one = dict(sample_data.iloc[max_index, :])
            perf = get_performance(top_one)

            if self.min_or_max == 'max':
                if eta < perf:
                    eta_index = i + 1
                eta = max(eta, perf)
            else:
                if eta > perf:
                    eta_index = i + 1
                eta = min(eta, perf)

            top_one[self.performance] = perf

            one_data = pd.DataFrame([top_one])
            origin_data = pd.concat([origin_data, one_data[list(origin_data.columns)]], axis=0)
            origin_data = origin_data.reset_index(drop=True)

            temp_data_real_perf_min = pd.concat([temp_data_real_perf_min, one_data], axis=0)

            writer.add_scalar('perf_min', perf, global_step=i)
        with open(self.save_filesname + 'optimal_value.json', "w") as f:
            json.dump({"index":str(eta_index),"opt_perf":float(eta)}, f)
        temp_data_real_perf_min.to_csv(self.save_filesname+"temp_data_real_perf_min.csv", index=False)

    def get_ei(self, model, eta, sample_data):
        data = self.data_in(sample_data)
        self.scalar=StandardScaler()
        data = self.scalar.fit_transform(data.astype(float))

        pred = []
        for e in model.estimators_:
            pred.append(e.predict(data))
        pred = np.array(pred).transpose(1, 0)
        m = np.mean(pred, axis=1)
        s = np.std(pred, axis=1)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = self.calculate_f(eta,m,s)
            f[s_copy == 0.0] = 0.0
        else:
            f = self.calculate_f(eta,m,s)
        return f

    def calculate_f(self,eta,m,s):
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)
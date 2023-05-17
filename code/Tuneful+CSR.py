import math
from scipy.stats import norm
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from sklearn.preprocessing import StandardScaler
import json
from collections import *
from utils import *
import operator
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor
from sklearn import svm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from DeepPerf.mlp_sparse_model import *
import copy

class gaussian_process(object):
    def __init__(self,task_name):

        self.confs = json.load(open('paras.json'), strict=False)
        self.arg = json.load(open('paras.json'), strict=False)

        self.selected_params = self.confs['common_params']['all_params']

        self.task_name = task_name

        self.system = self.confs['common_params']['systerm']


        self.candidate_supplement=self.confs['candidate_supplement']

        self.candidate_supplement_threshold = self.confs['candidate_completion_threshold']

        self.sourcetasks_get_method=self.confs['sourcetasks_get_method']

        self.slected_k_source=self.confs['slected_k_source']

        self.sourcetask_num=self.confs['sourcetask_num']

        self.sourcetask_models = [None] * self.sourcetask_num

        self.alpha_max=self.confs['alpha_max']
        self.alpha_min=self.confs['alpha_min']

        self.source_M_list=[]

        self.similarity_list=[]

        self.performance = self.confs['common_params']['performance'].split('_')[0]

        self.min_or_max = self.confs['common_params']['performance'].split('_')[1]

        self.model = None

        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_output/target')):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_output/target/'))
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_output/target/gp.pickle')


        self.import_feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'target_output/target/random_forest_feature_importance.txt')

        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'data/target_data/{}.csv'.format(self.task_name))

        self.save_filesname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'target_output/target/')


        self.columns = None
        self.fake_columns = None

        self.scalar = None

        self.data_type = {"executorCores": int,
                          "executorMemory": int,
                          "executorInstances": int,
                          "defaultParallelism": int,
                          "memoryOffHeapEnabled": str,
                          "memoryOffHeapSize": int,
                          "memoryFraction": float,
                          "memoryStorageFraction": float,
                          "shuffleFileBuffer": int,
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
            # print(data.values.shape)
            # enum_book[c] = list(pd.get_dummies(data[c]).columns)
            # print(data[c])
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

        for i in enum_list:
            data_temp = pd.DataFrame(columns=[i])
            data = pd.concat([data, data_temp], 1)

        for index in range(hang):
            for k in range(len(enum_list)):

                name_index = self.columns.index(enum_list[k])  
                if k != 0:
                    for a in range(k):
                        name_index = name_index + enum_number[a] - 1  

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

        number = 0
        for i in range(len(enum_list)):
            name_index = number + self.columns.index(enum_list[i])

            for j in range(enum_number[i]):
                data = data.drop([name_index + j], 1)

            number = number + enum_number[i] - 1

        for i in range(len(enum_list)):
            orgin_index = self.columns.index(enum_list[i])
            data.insert(orgin_index, enum_list[i], data.pop(enum_list[i]))

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
        data_x = self.data_in(data.iloc[:,:-1])

        self.scalar = StandardScaler()
        data_x = self.scalar.fit_transform(data_x.astype(float))

        kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5,
                                                                                             length_scale_bounds=(
                                                                                             1e-4, 1e4))
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        self.model.fit(data_x, data_y)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def target_train(self):


        targettask_data = pd.read_csv(self.filename, dtype=self.data_type)

        self.arg["start_sample_num"]=len(targettask_data.values)
        with open(self.save_filesname+"arg.json","w") as f:
            json.dump(self.arg,f,indent=2)

        feature_sample_num = self.confs['sample_params']['feature_sample_num'] 
        important_feature_nums=self.confs['sample_params']['important_feature_nums'] 
        sample_step_change = self.confs['sample_params']['sample_step_change']  
        temp_data_real_perf_min = pd.DataFrame(columns=targettask_data.columns)
        temp_data_forest_perf_min = pd.DataFrame(columns=targettask_data.columns)  
        eta = max(targettask_data.iloc[:, -1].values) if self.min_or_max == 'max'else min(targettask_data.iloc[:, -1].values) 
        eta_index=-np.argmax(targettask_data.iloc[:, -1].values) if self.min_or_max == 'max'else -np.argmin(targettask_data.iloc[:, -1].values)
        min_distance=pd.DataFrame(columns=["min_l2_distance"])

        every_epoch_remain_sample_num = []
        every_epoch_remain_sample_ratio=[]  
        sourcetask_name_list=os.listdir("./target_output/source/")
        similarity_save = pd.DataFrame(columns=sourcetask_name_list)  

        condenseed_sample_data_iszero_iter=[]

        self.important_feature = self.get_important_feature(targettask_data, important_feature_nums)
        self.selected_params_temp = {key: self.selected_params[key] for key in self.important_feature}

        writer = SummaryWriter(self.save_filesname+"runs/")
        self.get_sourcetask_models(sourcetask_name_list)

        self.source_M_list=self.cal_M_list(self.sourcetask_models,(self.data_in(targettask_data.iloc[:,:-1])).values)

        for i in trange(self.confs['sample_params']['sample_epoch']):
            self.similarity_list=self.cal_Similarity(self.source_M_list,targettask_data.iloc[:, -1].values)

            similarity_one_iter = pd.DataFrame(data=[self.similarity_list], columns=similarity_save.columns)
            similarity_save=pd.concat([similarity_save,similarity_one_iter],axis=0)


            self.selected_svm_index=self.similarity_list.index(max(self.similarity_list))

            sourcetask_name=sourcetask_name_list[self.selected_svm_index]
            start_data = pd.read_csv("./data/source_data/" + sourcetask_name + ".csv", dtype=self.data_type)
            train_data = pd.read_csv("./target_output/source/" + sourcetask_name + "/temp_data_real_perf_min.csv", dtype=self.data_type)
            data=pd.concat([start_data, train_data], axis=0)
            data = data.reset_index(drop=True)
            data = pd.concat([data, temp_data_real_perf_min], axis=0)
            data = data.reset_index(drop=True)
            data=data.loc[:, self.important_feature+["PERF"]]

            temp=self.selected_params
            self.selected_params=self.selected_params_temp
            self.one_step_train(data)
            self.selected_params=temp
            self.selected_svm_classifier=self.get_selected_sourcetask_svm_clf(self.task_name,temp_data_real_perf_min)

            sample_data = random_sample_1(self.selected_params_temp,feature_sample_num)
            sample_data = sample_data.reset_index(drop=True)


            condenseed_sample_data,discard_data=self.condense_sampledata(self.selected_svm_classifier,sample_data)

            every_epoch_remain_sample_num.append(len(condenseed_sample_data.values))
            every_epoch_remain_sample_ratio.append(len(condenseed_sample_data.values) / len(sample_data.values))
            if self.candidate_supplement==0:
                if len(condenseed_sample_data.values) == 0:
                    condenseed_sample_data = sample_data.iloc[:, :]
            elif self.candidate_supplement==1:
                m=0
                while len(condenseed_sample_data.values)<self.candidate_supplement_threshold and m<50:
                    candidate_sample_data=random_sample_1(self.selected_params_temp,feature_sample_num)
                    candidate_sample_data,_=self.condense_sampledata(self.selected_svm_classifier,candidate_sample_data,self.slected_k_source)
                    condenseed_sample_data=pd.concat([condenseed_sample_data, candidate_sample_data], axis=0)
                    condenseed_sample_data = condenseed_sample_data.reset_index(drop=True)
                    m += 1
                if len(condenseed_sample_data.values)==0:
                    condenseed_sample_data_iszero_iter.append(i)
                    condenseed_sample_data = sample_data.iloc[:, :]
            elif self.candidate_supplement==2:
                if len(condenseed_sample_data.values) <= 1:
                    condenseed_sample_data = sample_data.iloc[:, :]
                candidate_data_temp=list(condenseed_sample_data.values)
                while len(candidate_data_temp)<self.candidate_supplement_threshold:
                    point_1_index,point_2_index=np.random.choice(len(candidate_data_temp),2,replace=False)
                    point_1,point_2=np.array(candidate_data_temp[point_1_index]),np.array(candidate_data_temp[point_2_index])
                    in_point=point_1+random.uniform(0,1)*(point_2-point_1)
                    candidate_data_temp.append(in_point)
                condenseed_sample_data=pd.DataFrame(candidate_data_temp,columns=list(self.selected_params_temp.keys()))
            elif self.candidate_supplement == 3:
                if len(condenseed_sample_data.values) <= 1:
                    condenseed_sample_data = sample_data.iloc[:, :]
                candidate_data_temp = list(condenseed_sample_data.values)
                if len(candidate_data_temp) < self.candidate_supplement_threshold:
                    print(len(candidate_data_temp))
                    print("______________")
                    start = time.time()
                    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
                    min_l2_distance = self.get_two_point_min_distance(candidate_data_temp, self.selected_params_temp,
                                                                      sample_data.columns)

                    if min_l2_distance == 0:
                        min_l2_distance = 0.01
                    distance = pd.DataFrame([[min_l2_distance]], columns=min_distance.columns)

                    end = time.time()
                    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))

                else:
                    min_l2_distance = 0
                    distance = pd.DataFrame([[min_l2_distance]], columns=min_distance.columns)

                min_distance = pd.concat([min_distance, distance], axis=0)

                start = time.time()
                start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))

                while len(candidate_data_temp) < self.candidate_supplement_threshold:
                    n = len(candidate_data_temp)
                    center = candidate_data_temp[np.random.randint(n)]
                    sample_temp = self.box_of_point_sample(center, min_l2_distance, sample_data.columns)
                    condese_sample_k, _ = self.condense_sampledata(self.selected_svm_classifier, sample_temp)
                    candidate_data_temp.extend(condese_sample_k.values)

                end = time.time()
                end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))

                condenseed_sample_data = pd.DataFrame(candidate_data_temp, columns=list(self.selected_params_temp.keys()))


            ei = self.get_ei(self.selected_svm_index,self.model, eta, condenseed_sample_data)
            max_index = np.argmax(ei)

            top_one = dict(condenseed_sample_data.iloc[max_index, :])
            # print(top_one)
            perf = get_performance(top_one)

            self.updata_M_list(self.source_M_list,self.sourcetask_models,self.data_in(pd.DataFrame([top_one_temp])).values)

            if self.min_or_max == 'max':
                if eta < perf:
                    eta_index = i + 1
                eta = max(eta, perf)
            else:
                if eta > perf:
                    eta_index = i + 1
                eta = min(eta, perf)

            top_one[self.performance] = perf
            top_one_temp[self.performance] = perf
            targettask_data = pd.concat([targettask_data, (pd.DataFrame([top_one_temp]))[list(targettask_data.columns)]], axis=0)
            targettask_data = targettask_data.reset_index(drop=True)


            one_data = pd.DataFrame([top_one])

            temp_data_real_perf_min = pd.concat([temp_data_real_perf_min, one_data], axis=0)

            feature_sample_num = feature_sample_num + sample_step_change

            writer.add_scalar('perf_min_forest', self.model.predict(
                self.data_in(condenseed_sample_data.iloc[max_index:max_index + 1, :]).values)[0], global_step=i)
            writer.add_scalar('perf_min', perf, global_step=i)
            np.savetxt(self.save_filesname + 'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)
            min_distance.to_csv(self.save_filesname + "min_distance.csv", index=False)
            similarity_save.to_csv(self.save_filesname + "similarity.csv", index=False)
            np.savetxt(self.save_filesname + 'every_epoch_remain_sample_num.txt', every_epoch_remain_sample_num)
            np.savetxt(self.save_filesname + 'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)
            temp_data_real_perf_min.to_csv(self.save_filesname + "temp_data_real_perf_min.csv", index=False)
        np.savetxt(self.save_filesname + 'condenseed_sample_data_iszero_iter.txt', condenseed_sample_data_iszero_iter)
        with open(self.save_filesname + 'optimal_value.json', "w") as f:
            json.dump({"index": str(eta_index), "opt_perf": float(eta)}, f)

    def get_sort_feature(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_weight_dict = defaultdict(int)
        for i in range(len(self.fake_columns)):
            if "_" in self.fake_columns[indices[i]]:
                feature_weight_dict[self.fake_columns[indices[i]].split('_')[0]] = max(importances[indices[i]],
                                                                                       feature_weight_dict[
                                                                                           self.fake_columns[
                                                                                               indices[i]].split(
                                                                                               '_')[0]])
            else:
                feature_weight_dict[self.fake_columns[indices[i]]] = importances[indices[i]]

        feature_weight_dict = dict(sorted(feature_weight_dict.items(), key=operator.itemgetter(1), reverse=True))
        with open(self.import_feature_path, 'a') as f:
            f.write(str(feature_weight_dict)+"\n")

        return list(feature_weight_dict.keys())

    def test_(self, data):
        data[self.performance] = 0
        data, _, _ = self.data_in(data)
        data = self.scalar.transform(data)
        return self.model.predict(data.values)

    def choose_source(self,slected_k_source,similarity_list,sourcetask_num,sourcetasks_get_method):
        if sourcetasks_get_method==0:
            similarity_list_sum=sum(similarity_list)
            p_list=[v/similarity_list_sum for v in similarity_list]
            if sum(p_list)!= 1:
                p_list[-1]=1-sum(p_list[:-1])
            selected_svm_index=np.random.choice(sourcetask_num,slected_k_source,replace=False,p=p_list)
        elif sourcetasks_get_method==1:
            similarity_list_temp=similarity_list[:]
            selected_svm_index = []
            Inf = 0
            for i in range(slected_k_source):
                selected_svm_index.append(similarity_list_temp.index(max(similarity_list_temp)))
                similarity_list_temp[similarity_list_temp.index(max(similarity_list_temp))] = Inf
        return selected_svm_index

    def condense_sampledata(self, clff, sample_data):
        temp=self.selected_params
        self.selected_params=self.selected_params_temp
        data_x=self.data_in(sample_data)
        self.selected_params=temp
        condenseed_sample_data=[]
        discard_data=[]
        for i,one_data in enumerate(data_x.values):
            if clff.predict([one_data])[0]==1:
                condenseed_sample_data.append(sample_data.values[i])
            else:
                discard_data.append(sample_data.values[i])
        return pd.DataFrame(condenseed_sample_data, columns=sample_data.columns),pd.DataFrame(discard_data, columns=sample_data.columns)

    def get_selected_sourcetask_svm_clf(self, task_name,temp_data_real_perf_min):
        sample_data_1 = pd.read_csv("./data/target_data/{}.csv".format(task_name),dtype=self.data_type)
        sample_data_2 = temp_data_real_perf_min
        sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
        sample_data = sample_data.reset_index(drop=True)
        sample_data = sample_data.loc[:,self.important_feature]
        temp=self.selected_params
        self.selected_params=self.selected_params_temp
        X=self.data_in(sample_data)
        self.selected_params=temp

        y = sample_data.iloc[:, -1].values
        y_min, y_max = min(y), max(y)
        alpha_i = self.alpha_min + (1 - 2 * max(max(self.similarity_list) - 0.5, 0)) * (
                    self.alpha_max - self.alpha_min)
        print("alpha_i")
        print(alpha_i)
        perf_list=sample_data.iloc[:,-1].values
        temp=sorted(perf_list)
        y_plus_i = temp[int(len(perf_list)*alpha_i)]

        print("y_plus_i")
        print(y_plus_i)
        y = [1 if v < y_plus_i else 0 for v in y]
        clff = svm.SVC(gamma='scale')
        clff.fit(X, y)
        return clff

    def get_ei(self,selected_svm_index, model, eta, sample_data):
        temp=self.selected_params
        self.selected_params=self.selected_params_temp
        data = self.data_in(sample_data)
        self.selected_params=temp

        self.scalar=StandardScaler()
        data = self.scalar.fit_transform(data.astype(float))
        m, s=model.predict(data,return_cov=True)

        m, s = m.ravel(),np.sqrt(np.diag(s))

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = self.calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = self.calculate_f(eta, m, s)
        return f

    def calculate_f(self,eta,m,s):

        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)


    def get_sourcetask_models(self, sourcetask_name_list):
        for i,sourcetask_name in enumerate(sourcetask_name_list):
            # print(sourcetask_name)
            sample_data_1 = pd.read_csv("./data/source_data/{}.csv".format(sourcetask_name),dtype=self.data_type)
            sample_data_2 = pd.read_csv("./target_output/source/{}/temp_data_real_perf_min.csv".format(sourcetask_name),dtype=self.data_type)
            sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
            sample_data = sample_data.reset_index(drop=True)
            X = (self.data_in(sample_data.iloc[:, :-1])).values
            y = sample_data.iloc[:, -1].values

            self.scalar = StandardScaler()
            X = self.scalar.fit_transform(X.astype(float))
            kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5,
                                                                                                 length_scale_bounds=(
                                                                                                     1e-4, 1e4))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
            model.fit(X, y)
            self.sourcetask_models[i] = model

    def cal_M_list(self, sourcetask_models, target_X):
        return list(map(self.cal_one_M,sourcetask_models,[target_X]*len(self.sourcetask_models)))

    def cal_one_M(self,sourcetask_model,target_X):
        return list(sourcetask_model.predict(target_X).ravel())

    def cal_Similarity(self, source_M_list, target_Y):
        similarity_list = list(map(self.cal_one_Similarity, source_M_list, [target_Y] * len(self.sourcetask_models)))

        return similarity_list

    def cal_one_Similarity(self, one_M, target_Y):

        one_M_col = np.array(one_M).reshape(len(target_Y), 1)
        one_M_row = one_M_col.T
        target_Y_col = np.array(target_Y).reshape(len(target_Y), 1)
        target_Y_row = target_Y_col.T
        temp = np.logical_not(np.logical_xor(one_M_col < one_M_row, target_Y_col < target_Y_row))
        rank_loss = np.sum(temp[np.triu_indices(len(target_Y), 1)])
        return 2 * rank_loss / (len(target_Y) * (len(target_Y) - 1))

    def updata_M_list(self, source_M_list, sourcetask_models, values):

        for i in range(len(sourcetask_models)):
            source_M_list[i].append(sourcetask_models[i].predict(values)[0])

    def get_two_point_min_distance(self, candidate_data_temp,selected_params,sample_data_columns):
        candidate_data_temp=np.array(candidate_data_temp)
        left,right=[],[]
        length=candidate_data_temp.shape[1]
        print("length",length)
        for k in range(length-1,-1,-1):
            key = sample_data_columns[k]
            if selected_params[key][0] == 'enum':
                candidate_data_temp=np.delete(candidate_data_temp,k,axis=1)
            else:
                left=[selected_params[key][1][0]]+left
                right=[selected_params[key][1][1]]+right
        if len(left)>5:
            temp=list(np.random.choice(len(left),len(left)-5,replace=False))
            temp.sort(reverse=True)
            left, right = np.array(left).reshape(1, -1), np.array(right).reshape(1, -1)

            for k in temp:
                candidate_data_temp=np.delete(candidate_data_temp,k,axis=1)
                left=np.delete(left,k,axis=1)
                right = np.delete(right, k, axis=1)

        print("left",left)
        print("right",right)

        candidate_data_temp=(candidate_data_temp-left)/right
        axis_1_expand=np.expand_dims(candidate_data_temp, 1)
        axis_0_expand = np.expand_dims(candidate_data_temp, 0)
        Difference=axis_1_expand-axis_0_expand
        del axis_1_expand
        del axis_0_expand

        Difference=Difference[(np.triu_indices(Difference.shape[0], k=1))]
        Difference=Difference.astype('float64')
        l2=np.linalg.norm(Difference, ord=2,axis=1, keepdims=False)
        return np.min(l2)


    def box_of_point_sample(self,center,min_distance,sample_data_columns):
        all_params={key: copy.deepcopy(self.selected_params_temp[key]) for key in self.important_feature}
        for j in range(len(center)):
            key=sample_data_columns[j]
            if all_params[key][0]=="int" or all_params[key][0]=="float":
                a,b=all_params[key][1][0],all_params[key][1][1]
                distance=min_distance*(b-a)
            if all_params[key][0]=="int":
                if distance < 1:
                    left=a if center[j]-1<a else center[j]-1
                    right=b if center[j]+1>b else center[j]+1
                else:
                    left=a if center[j]-distance<a else center[j]-distance
                    right=b if center[j]+distance>b else center[j]+distance

                all_params[key][1][0], all_params[key][1][1] = int(left), int(right)

            elif  all_params[key][0]=="float":
                left = a if center[j] - distance < a else center[j] - distance
                right = b if center[j] + distance > b else center[j] + distance

                all_params[key][1][0], all_params[key][1][1] = left,right
        return random_sample_1(all_params,10)



    def get_important_feature(self, data, important_feature_nums):
        data_y = data.iloc[:, -1].values
        data_x = self.data_in(data.iloc[:,:-1])
        self.scalar = StandardScaler()
        data_x = self.scalar.fit_transform(data_x.astype(float))

        self.model = RandomForestRegressor()
        self.model.fit(data_x, data_y)

        sort_features = self.get_sort_feature()

        return sort_features[:important_feature_nums]




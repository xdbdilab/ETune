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
# from DeepPerf.mlp_sparse_model import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from DeepPerf.mlp_sparse_model import *

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

        self.MODEL = MLPSparseModel(systerm=self.system, TEST_flag=True)
        self.MODEL.restore_session()

        deepperf_scalar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepPerf/result')
        deepperf_scalar_name = 'result_{}_AutoML_veryrandom.txt'.format(self.system)
        deepperf_scalar_path = os.path.join(deepperf_scalar_dir, deepperf_scalar_name)
        file_read = open(deepperf_scalar_path, 'r', encoding='utf-8')
        Params = file_read.read()
        Params = dict(eval(Params))
        self.max_X = Params['max_x']
        self.max_Y = Params['max_y']
        self.SYSTERM_COLUMNS = Params['SYSTERM_COLUMNS']

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
        data.rename(columns=col, inplace=True)
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

        origin_data = pd.read_csv(self.filename,dtype=self.data_type)
        self.arg["start_sample_num"]=len(origin_data.values)
        with open(self.save_filesname+"arg.json","w") as f:
            json.dump(self.arg,f,indent=2)

        feature_sample_num = self.confs['sample_params']['feature_sample_num']
        sample_step_change = self.confs['sample_params']['sample_step_change']
        temp_data_real_perf_min = pd.DataFrame(columns=origin_data.columns) 
        min_distance=pd.DataFrame(columns=["min_l2_distance"])

        eta = max(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else min(origin_data.iloc[:, -1].values)
        eta_index=-np.argmax(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else -np.argmin(origin_data.iloc[:, -1].values)

        every_epoch_remain_sample_num = [] 
        every_epoch_remain_sample_ratio=[]

        self.sourcetask_name_list=os.listdir("./target_output/source/")

        similarity_save = pd.DataFrame(columns=self.sourcetask_name_list)  

        condenseed_sample_data_iszero_iter=[]

        writer = SummaryWriter(self.save_filesname+"runs/")

        self.get_sourcetask_models(self.sourcetask_name_list)

        self.source_M_list=self.cal_M_list(self.sourcetask_models,(self.data_in(origin_data.iloc[:,:-1])).values)

        for i in trange(self.confs['sample_params']['sample_epoch']):
            self.one_step_train(origin_data)
            self.similarity_list=self.cal_Similarity(self.source_M_list,origin_data.iloc[:, -1].values)
            similarity_one_iter = pd.DataFrame(data=[self.similarity_list], columns=similarity_save.columns)
            similarity_save=pd.concat([similarity_save,similarity_one_iter],axis=0)

            self.selected_svm_index=self.choose_source(self.slected_k_source,self.similarity_list,self.sourcetask_num,self.sourcetasks_get_method)

            self.selected_svm_classifier=self.get_selected_sourcetask_svm_clf(np.array(self.selected_svm_index))

            sample_data = random_sample_1(self.selected_params,feature_sample_num)
            sample_data = sample_data.reset_index(drop=True)

            condenseed_sample_data=sample_data
            ei = self.get_ei(self.selected_svm_index,self.model, eta, condenseed_sample_data)
            max_index = np.argmax(ei)

            top_one = dict(condenseed_sample_data.iloc[max_index, :])
            perf = get_performance(top_one)

            self.updata_M_list(self.source_M_list,self.sourcetask_models,(self.data_in(condenseed_sample_data.iloc[max_index:max_index+1, :])).values)

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

            feature_sample_num = feature_sample_num + sample_step_change

            writer.add_scalar('perf_min', perf, global_step=i)
            np.savetxt(self.save_filesname + 'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)
            min_distance.to_csv(self.save_filesname+"min_distance.csv", index=False)
            similarity_save.to_csv(self.save_filesname+"similarity.csv", index=False)
            np.savetxt(self.save_filesname+'every_epoch_remain_sample_num.txt', every_epoch_remain_sample_num)
            np.savetxt(self.save_filesname+'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)
            temp_data_real_perf_min.to_csv(self.save_filesname + "temp_data_real_perf_min.csv", index=False)
        np.savetxt(self.save_filesname + 'condenseed_sample_data_iszero_iter.txt', condenseed_sample_data_iszero_iter)
        with open(self.save_filesname + 'optimal_value.json', "w") as f:
            json.dump({"index":str(eta_index),"opt_perf":float(eta)}, f)


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

    def condense_sampledata(self, selected_svm_classifier, sample_data,slected_k_source):
        condenseed_sample_data=[]
        discard_data=[]
        sample_data_temp=self.data_in(sample_data)
        for i,one_data in enumerate(sample_data_temp.values):
            count=0
            for clff in selected_svm_classifier:
                if clff.predict([one_data])[0]==1:
                    count += 1
            if count>=slected_k_source//2+1:
                condenseed_sample_data.append(sample_data.iloc[i,:].values)
            else:
                discard_data.append(sample_data.iloc[i,:].values)
        return pd.DataFrame(condenseed_sample_data, columns=sample_data.columns),pd.DataFrame(discard_data, columns=sample_data.columns)

    def get_selected_sourcetask_svm_clf(self, selected_svm_index):
        selected_sourcetask_svm_classifier=list(map(self.get_one_clf,selected_svm_index))
        return selected_sourcetask_svm_classifier

    def get_one_clf(self,index):
        sample_data_1 = pd.read_csv("./data/source_data/{}.csv".format(self.sourcetask_name_list[index]),dtype=self.data_type)

        sample_data_2 = pd.read_csv("./target_output/source/{}/temp_data_real_perf_min.csv".format(self.sourcetask_name_list[index]),dtype=self.data_type)
        sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
        sample_data = sample_data.reset_index(drop=True)
        X = (self.data_in(sample_data.iloc[:, :-1])).values

        y = sample_data.iloc[:, -1].values
        alpha_i=self.alpha_min+(1-2*max(self.similarity_list[index]-0.5,0))*(self.alpha_max-self.alpha_min)
        print("alpha_i")
        print(alpha_i)
        perf_list=sample_data.iloc[:,-1].values
        temp=sorted(perf_list)
        y_plus_i = temp[int(len(perf_list)*alpha_i)]

        print(self.sourcetask_name_list[index])
        print("y_plus_i")
        print(y_plus_i)
        y = [1 if v < y_plus_i else 0 for v in y]
        clff = svm.SVC(gamma='scale')
        clff.fit(X, y)
        return clff

    def get_ei(self,selected_svm_index, model, eta, sample_data):
        data = self.data_in(sample_data)
        self.scalar=StandardScaler()
        data = self.scalar.fit_transform(data.astype(float))

        mu=np.zeros(len(sample_data.values))
        g_sum=1
        for index in selected_svm_index:
            g_sum+=self.similarity_list[index]
            muu=self.sourcetask_models[index].predict(data)
            mu+=self.similarity_list[index]*muu.ravel()

        m, s=model.predict(data,return_cov=True)

        m, s = m.ravel(),np.sqrt(np.diag(s))
        m=(m+mu)/g_sum

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
        # l1=np.fabs(Difference.reshape(-1))
        return np.min(l2)



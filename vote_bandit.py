# coding: utf-8
# Author: crb
# Date: 2021/7/17 20:53
import csv
import datetime
import logging
import pickle
import random
import sys
import os
import subprocess
import time
from collections import Counter
import numpy as np
from numpy.linalg import inv

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# print(rootPath)
from get_arm_qos import update_core_arm_context, arm_cor_numapp, get_llc_bandwith_config
from run_and_getconfig_qos import gen_config, get_now_ipc, run_bg_benchmark, gen_init_config,LC_APP_NAMES
from store_data import save_file
# Info: arm分几层 各种各存一个矩阵 一个个决定后给出最后决策 reward是一样的所有是同时决定
# cache:55 arms/per app
# app band :10 arms/per app
# cpu arm 扩容方便
# 1_st: 只用上部分历史数据, 下一个colocation没出现的就删掉




class LinUCB():
    '''
    train_success调用情况：mab_1 = LinUCB(nof_counters=22个性能指标, alpha=0.01, colocation_list[0]=app_id, core_arm_orders=所有core的组合排列)
    '''
    def __init__(self, ndims, th_alpha, fair_alpha, app_id, core_arm_orders, llc_narms=55, band_namrms=10):
        self.num_app = len(app_id)
        assert 2 <= self.num_app <= 7, 'the num of colocation size is wrong'
        self.all_core_narms_p = {2: 8, 3: 34, 4: 92, 5: 130, 6: 86, 7: 28}
        #根据app的个数确定对应core_narms的大小，也就是每个app的臂的个数(可供选择的cpu组合的决策数)
        self.core_narms = self.all_core_narms_p[self.num_app]  
        self.app_id = app_id
        self.llc_narms = llc_narms
        self.band_namrms = band_namrms
        # number of context features
        self.ndims = ndims #性能个数，22
        # explore-exploit parameter
        self.th_alpha = th_alpha
        self.core_arm_orders = core_arm_orders

        self.th_A_c = {}
        self.th_b_c = {}
        self.th_p_c_t = {}

        self.th_A_l = {}
        self.th_b_l = {}
        self.th_p_l_t = {}

        self.th_A_b = {}
        self.th_b_b = {}
        self.th_p_b_t = {}

        self.fair_A_c = {}
        self.fair_b_c = {}
        self.fair_p_c_t = {}

        self.fair_A_l = {}
        self.fair_b_l = {}
        self.fair_p_l_t = {}

        self.fair_A_b = {}
        self.fair_b_b = {}
        self.fair_p_b_t = {}
        # a,b,p参数：
        # a:{app1:[arms1[44*44矩阵],arms2[44*44矩阵],...],  app2:...}
        # b:{app1:[arms1[44*1矩阵],arms2[44*1矩阵],...],    app2:...}
        # p:{app1:[arms1[数值],arms2[数值],...],    app2:...}
        for i in app_id:
            #throughput
            self.th_A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims* 2))
            self.th_b_c[i] = np.zeros((self.core_narms, self.ndims* 2, 1))
            self.th_p_c_t[i] = np.zeros(self.core_narms)

            self.th_A_l[i] = np.zeros((self.llc_narms, self.ndims* 2, self.ndims* 2))
            self.th_b_l[i] = np.zeros((self.llc_narms, self.ndims* 2, 1))
            self.th_p_l_t[i] = np.zeros(self.llc_narms)

            self.th_A_b[i] = np.zeros((self.band_namrms, self.ndims* 2, self.ndims* 2))
            self.th_b_b[i] = np.zeros((self.band_namrms, self.ndims* 2, 1))
            self.th_p_b_t[i] = np.zeros(self.band_namrms)
            for arm in range(self.core_narms):
                self.th_A_c[i][arm] = np.eye(self.ndims* 2)  #self.th_A_c[i][arm]代表第i个app的cpu策略的第arm个臂，每个臂就是一个44*44的单位矩阵

            for arm in range(self.llc_narms):
                self.th_A_l[i][arm] = np.eye(self.ndims* 2)

            for arm in range(self.band_namrms):
                self.th_A_b[i][arm] = np.eye(self.ndims* 2)

            #fairness
            self.fair_A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims* 2))
            self.fair_b_c[i] = np.zeros((self.core_narms, self.ndims* 2, 1))
            self.fair_p_c_t[i] = np.zeros(self.core_narms)

            self.fair_A_l[i] = np.zeros((self.llc_narms, self.ndims* 2, self.ndims* 2))
            self.fair_b_l[i] = np.zeros((self.llc_narms, self.ndims* 2, 1))
            self.fair_p_l_t[i] = np.zeros(self.llc_narms)

            self.fair_A_b[i] = np.zeros((self.band_namrms, self.ndims* 2, self.ndims* 2))
            self.fair_b_b[i] = np.zeros((self.band_namrms, self.ndims* 2, 1))
            self.fair_p_b_t[i] = np.zeros(self.band_namrms)
            for arm in range(self.core_narms):
                self.fair_A_c[i][arm] = np.eye(self.ndims* 2)  #self.th_A_c[i][arm]代表第i个app的cpu策略的第arm个臂，每个臂就是一个44*44的单位矩阵

            for arm in range(self.llc_narms):
                self.fair_A_l[i][arm] = np.eye(self.ndims* 2)

            for arm in range(self.band_namrms):
                self.fair_A_b[i][arm] = np.eye(self.ndims* 2)

        super().__init__()
        return

    def add_del_app(self, app_id):
        '''
        根据新一轮的app_id，对self的各项参数进行更新（貌似只需要在一开始时更新一下原始参数就行，因为只运行一轮app_id,没有新的app_id）
        '''
        assert 2 <= self.num_app <= 7, 'the num of colocation size is wrong'
        th_A_c, th_A_l, th_A_b, fair_A_c, fair_A_l, fair_A_b = 0, 0, 0, 0, 0, 0  #th_A_c, th_A_l, A_b把所有app的core_arm，llc_arm,band_arm矩阵都加起来
        for i in self.th_A_c.keys():
            th_A_c += self.th_A_c[i]
            th_A_l += self.th_A_l[i]
            th_A_b += self.th_A_b[i]
            fair_A_c += self.fair_A_c[i]
            fair_A_l += self.fair_A_l[i]
            fair_A_b += self.fair_A_b[i]

        self.core_narms = self.all_core_narms_p[len(app_id)]

        for i in app_id:
            if i not in self.th_A_c.keys():
                self.th_A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
                self.th_b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
                self.th_p_c_t[i] = np.zeros((self.core_narms))
                self.fair_A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
                self.fair_b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
                self.fair_p_c_t[i] = np.zeros((self.core_narms))
                for arm in range(self.core_narms):
                    self.th_A_c[i][arm] = th_A_c[arm] / self.num_app  #新加入的app参数矩阵取旧的app的core arm矩阵的平均值
                    self.fair_A_c[i][arm] = fair_A_c[arm] / self.num_app

                self.th_A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
                self.th_b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
                self.th_p_l_t[i] = np.zeros((self.llc_narms))
                self.fair_A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
                self.fair_b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
                self.fair_p_l_t[i] = np.zeros((self.llc_narms))
                for arm in range(self.llc_narms):
                    self.th_A_l[i][arm] = th_A_l[arm] / self.num_app  #新加入的app参数矩阵取旧的app的llc arm矩阵的平均值
                    self.fair_A_l[i][arm] = fair_A_l[arm] / self.num_app

                self.th_A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
                self.th_b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
                self.th_p_b_t[i] = np.zeros((self.band_namrms))
                self.fair_A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
                self.fair_b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
                self.fair_p_b_t[i] = np.zeros((self.band_namrms))
                for arm in range(self.band_namrms):
                    self.th_A_b[i][arm] = th_A_b[arm] / self.num_app  #新加入的app参数矩阵取旧的app的band arm矩阵的平均值
                    self.fair_A_b[i][arm] = fair_A_b[arm] / self.num_app
            # if i not in self.th_A_c.keys():
            #     self.th_A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
            #     self.th_b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
            #     self.th_p_c_t[i] = np.zeros((self.core_narms))
            #     for arm in range(self.core_narms):
            #         self.th_A_c[i][arm] = np.eye(self.ndims * 2)
            #
            #     self.th_A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
            #     self.th_b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
            #     self.th_p_l_t[i] = np.zeros((self.llc_narms))
            #     for arm in range(self.llc_narms):
            #         self.th_A_l[i][arm] = np.eye(self.ndims * 2)
            #
            #     self.th_A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
            #     self.th_b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
            #     self.th_p_b_t[i] = np.zeros((self.band_namrms))
            #     for arm in range(self.band_namrms):
            #         self.th_A_b[i][arm] = np.eye(self.ndims * 2)

        self.th_A_c, self.th_b_c, self.th_p_c_t, self.fair_A_c, self.fair_b_c, self.fair_p_c_t = update_core_arm_context(self.app_id, app_id, self.core_arm_orders, 
                                                                            self.th_A_c,self.th_b_c, self.th_p_c_t,
                                                                            self.fair_A_c,self.fair_b_c, self.fair_p_c_t,
                                                                            self.ndims * 2)
        self.num_app = len(app_id)
        self.app_id = app_id

    def play(self, context, other_context, alg):
        '''
        根据上一轮运行获得的性能指标，以及不断更新的A,b两类参数，计算p_*_t，
        选择p_*_t最大的那一个arm_*,从而得到每个app的下一步的core_action, llc_action, band_action，
        {app1:int,app2:int ,...},   {app1:[int,int],app2:[int,int]...},   {app1:int,app2:int,...}

        :param context:记录每个app的22个性能数据，{app1:[feature1,feature2,...,feature22],app2:[...],...}
        :param other_context:记录除该app之外其他所有app的性能数据{app1:[other_features],app2:[other_features],...}
        :return: 该函数处理了三个字典：core_action，llc_action，band_action，
        这三个字典的key为app_id，value是每个app选择的动作，这三个字典组成了chosen_arms
        '''
        assert len(context[self.app_id[0]]) == self.ndims, 'the shape of context size is wrong'
        core_action = {}
        llc_action = {}
        band_action = {}
        contexts = {}
        # gains per each arm
        # only calculate the app in this colocation
        if alg == "throughput":
            for key in self.app_id:  #对app逐个操作，获取每个app的core_action, llc_action, band_action
                th_A = self.th_A_c[key]
                th_b = self.th_b_c[key]

                contexts[key] = np.hstack((context[key],other_context[key]))  #行不变，列组在一起

                for i in range(self.core_narms):
                    # initialize theta hat
                    th_theta = inv(th_A[i]).dot(th_b[i])
                    # 获取每个app对应的：[自身所有性能指标,其它app的性能指标]，get context of each arm from flattened vector of length 100
                    cntx = np.array(contexts[key])
                    # 获取每个臂的收益，get gain th_reward of each arm
                    self.th_p_c_t[key][i] = th_theta.T.dot(cntx) + self.th_alpha * np.sqrt(cntx.dot(inv(th_A[i]).dot(cntx)))

                
                th_A = self.th_A_l[key]
                th_b = self.th_b_l[key]
                for i in range(self.llc_narms):
                    th_theta = inv(th_A[i]).dot(th_b[i])
                    cntx = np.array(contexts[key])
                    self.th_p_l_t[key][i] = th_theta.T.dot(cntx) + self.th_alpha * np.sqrt(cntx.dot(inv(th_A[i]).dot(cntx)))
                

                th_A = self.th_A_b[key]
                th_b = self.th_b_b[key]
                for i in range(self.band_namrms):
                    th_theta = inv(th_A[i]).dot(th_b[i])
                    cntx = np.array(contexts[key])
                    self.th_p_b_t[key][i] = th_theta.T.dot(cntx) + self.th_alpha * np.sqrt(cntx.dot(inv(th_A[i]).dot(cntx)))
                
                core_action[key] = np.random.choice(np.where(self.th_p_c_t[key] == max(self.th_p_c_t[key]))[0])#选择p_c_t最大的那个动作
                llc_action[key] = np.random.choice(np.where(self.th_p_l_t[key] == max(self.th_p_l_t[key]))[0])
                band_action[key] = np.random.choice(np.where(self.th_p_b_t[key] == max(self.th_p_b_t[key]))[0])
        else:
            for key in self.app_id:  #对app逐个操作，获取每个app的core_action, llc_action, band_action
                fair_A = self.fair_A_c[key]
                fair_b = self.fair_b_c[key]

                contexts[key] = np.hstack((context[key],other_context[key]))  #行不变，列组在一起

                for i in range(self.core_narms):
                    # initialize theta hat
                    fair_theta = inv(fair_A[i]).dot(fair_b[i])
                    # 获取每个app对应的：[自身所有性能指标,其它app的性能指标]，get context of each arm from flattened vector of length 100
                    cntx = np.array(contexts[key])
                    # 获取每个臂的收益，get gain fair_reward of each arm
                    self.fair_p_c_t[key][i] = fair_theta.T.dot(cntx) + self.fair_alpha * np.sqrt(cntx.dot(inv(fair_A[i]).dot(cntx)))

                
                fair_A = self.fair_A_l[key]
                fair_b = self.fair_b_l[key]
                for i in range(self.llc_narms):
                    fair_theta = inv(fair_A[i]).dot(fair_b[i])
                    cntx = np.array(contexts[key])
                    self.fair_p_l_t[key][i] = fair_theta.T.dot(cntx) + self.fair_alpha * np.sqrt(cntx.dot(inv(fair_A[i]).dot(cntx)))
                

                fair_A = self.fair_A_b[key]
                fair_b = self.fair_b_b[key]
                for i in range(self.band_namrms):
                    fair_theta = inv(fair_A[i]).dot(fair_b[i])
                    cntx = np.array(contexts[key])
                    self.fair_p_b_t[key][i] = fair_theta.T.dot(cntx) + self.fair_alpha * np.sqrt(cntx.dot(inv(fair_A[i]).dot(cntx)))
                
                core_action[key] = np.random.choice(np.where(self.fair_p_c_t[key] == max(self.fair_p_c_t[key]))[0])#选择p_c_t最大的那个动作
                llc_action[key] = np.random.choice(np.where(self.fair_p_l_t[key] == max(self.fair_p_l_t[key]))[0])
                band_action[key] = np.random.choice(np.where(self.fair_p_b_t[key] == max(self.fair_p_b_t[key]))[0])

        return core_action, llc_action, band_action

    def update(self, core_arms, llc_arms, band_arms, th_reward, fair_reward, context,other_context,alg):
        '''
        根据环境交互得到的reward, context, other_context，对A_c，b_c，A_l，b_l，A_b，b_b这些参数进行更新
        '''
        contexts ={}
        print(core_arms)
        if alg == "throughput":
            for key in self.app_id:
                contexts[key] = np.hstack((context[key],other_context[key]))

                arm = core_arms[key]
                self.th_A_c[key][arm] += np.outer(np.array(contexts[key]),
                                            np.array(contexts[key]))
                self.th_b_c[key][arm] = np.add(self.th_b_c[key][arm].T,
                                            np.array(contexts[key]) * th_reward).reshape(
                    self.ndims*2, 1)

                arm = llc_arms[key]
                self.th_A_l[key][arm] += np.outer(np.array(contexts[key]),
                                            np.array(contexts[key]))
                self.th_b_l[key][arm] = np.add(self.th_b_l[key][arm].T,
                                            np.array(contexts[key]) * th_reward).reshape(
                    self.ndims*2, 1)

                arm = band_arms[key]
                self.th_A_b[key][arm] += np.outer(np.array(contexts[key]),
                                            np.array(contexts[key]))
                self.th_b_b[key][arm] = np.add(self.th_b_b[key][arm].T,
                                            np.array(contexts[key]) * th_reward).reshape(
                    self.ndims*2, 1)

        else:
            for key in self.app_id:
                contexts[key] = np.hstack((context[key],other_context[key]))

                arm = core_arms[key]
                self.fair_A_c[key][arm] += np.outer(np.array(contexts[key]),
                                            np.array(contexts[key]))
                self.fair_b_c[key][arm] = np.add(self.fair_b_c[key][arm].T,
                                            np.array(contexts[key]) * fair_reward).reshape(
                    self.ndims*2, 1)

                arm = llc_arms[key]
                self.fair_A_l[key][arm] += np.outer(np.array(contexts[key]),
                                            np.array(contexts[key]))
                self.fair_b_l[key][arm] = np.add(self.fair_b_l[key][arm].T,
                                            np.array(contexts[key]) * fair_reward).reshape(
                    self.ndims*2, 1)

                arm = band_arms[key]
                self.fair_A_b[key][arm] += np.outer(np.array(contexts[key]),
                                            np.array(contexts[key]))
                self.fair_b_b[key][arm] = np.add(self.fair_b_b[key][arm].T,
                                            np.array(contexts[key]) * fair_reward).reshape(
                    self.ndims*2, 1)

'''
主函数引用情况：train_success(load_list_i=0, load_list_m=0)
'''
def train_success(load_list_i,load_list_m,pre_rounds=100,rounds=100):  #load_list_i=load_list_m=0
    nof_counters = 22
    nof_colocation = len(colocation_list)  #nof_colocation=1，此外colocation_list[0]中记录的是app_id

    init_alg = "fair"

    th_alpha = 0.01
    fair_alpha = 0.01
    th_reward_baseline = 100
    fair_reward_baseline = 10
    mab_1 = LinUCB(nof_counters, th_alpha, fair_alpha, colocation_list[0], core_arm_orders)
    #fromkeys() 函数用于创建一个新字典，以序列 seq 中元素做字典的键，value 为字典所有键对应的初始值：dict.fromkeys(seq[, value])
    #store_dict用于存储一系列信息
    store_dict = {}.fromkeys(["app_id", "load_list", "th_reward", "core_config", "llc_config", "mb_config", "counters","other_counters"])

    for col_items in range(nof_colocation):
        now = datetime.datetime.now()

        app_id = colocation_list[col_items]  #colocation_list[list[]],每一个list[]中都记录着一组需要共定位的app

        lc_app, bg_app = [], []
        for i in app_id:
            if i in LC_APP_NAMES:  #LC_APP_NAMES是一批专指的app，这里将colocation_list中的app分为lc_app和bg_app两类
                lc_app.append(i)
            else:
                bg_app.append(i)

        print(f"Start run {col_items}th")

        chose_arm_storage = []
        reward_arms = []
        cumulative_reward = []
        G = [0,0]
        history = []

        #拿到一组app后，首先获取三种资源的初始化情况，以及选择的初始化动作
        core_list, llc_config, mb_config, chosen_arms = gen_init_config(app_id, core_arm_orders, llc_arm_orders,
                                                                        alg=init_alg)

        if bg_app != []:
            run_bg_benchmark(bg_app, core_list[len(lc_app):])

    
        if col_items!= nof_colocation-1:
            nrounds = pre_rounds  
        else:
            nrounds = rounds

        #load_list记录着每个app的某种情况——load_list的作用：load_list控制着每个app在运行过程中的qps（每秒查询率）级别
        load_list = [load_list_i] *(len(colocation_list[col_items])-1)
        load_list.append(load_list_m)
        '''
        app_id组和load_list一同被存入vote_sample_times_1.csv
        '''
        f_w.writerow([colocation_list[col_items],load_list])
        logging.error("colocation,load_list, {} {}".format(colocation_list[col_items],load_list))
        for i in range(nrounds):
            if nrounds % 50 == 0:
                #建立mab123，另外每过50轮交互，就将mab1重建一次，清除历史记录，来记录更新的交互历史
                if "mab_2" in locals().keys():  #locals().keys()代表当前所有局部变量名称，如果不存在指定的变量"mab_2"就创建一个
                    if "mab_3" in locals().keys():
                        mab_1 = LinUCB(nof_counters, th_alpha, fair_alpha, colocation_list[col_items], core_arm_orders)
                    else:
                        mab_3 = LinUCB(nof_counters, th_alpha, fair_alpha, colocation_list[col_items], core_arm_orders)
                else:
                    mab_2 = LinUCB(nof_counters, th_alpha, fair_alpha, colocation_list[col_items], core_arm_orders)
            
            if "mab_2" in locals().keys():
                if "mab_3" in locals().keys():
                    mab_1.add_del_app(app_id)
                    mab_2.add_del_app(app_id)
                    mab_3.add_del_app(app_id)
                else:
                    mab_1.add_del_app(app_id)
                    mab_2.add_del_app(app_id)
            else:
                mab_1.add_del_app(app_id)

            if i == 0:
                #计算得到该资源配置方案的reward
                context, another_context, th_reward, fair_reward, p95_list = get_now_ipc(lc_app, bg_app, load_list, core_list, evne)
                #save_file函数负责将当前所有信息整合为一个int型list，也就是store_file列表
                #关于chosen_arms：chosen_arms = [core_arms,llc_arms,mb_arms]
                store_file = save_file(app_id, load_list, th_reward, fair_reward, context, chosen_arms[0][app_id[0]], list(chosen_arms[1].values()),
                                              list(chosen_arms[2].values()))
                last_fair_reward = fair_reward
                last_th_reward = th_reward
                '''
                store_file列表被存入vote_sample_times_data_1.csv
                '''
                f_d_w.writerow(store_file)
                #将当前的资源配置方案存储进入chose_arm_storage列表中
                chose_arm_storage.append([core_list, llc_config, mb_config])
                #onlineEvaluate函数分析该步动作的reward等信息，并给出各个动作的reward等信息
                reward_arms, chosen_arms, cumulative_reward, G = onlineEvaluate(mab_1, th_reward, fair_reward, reward_arms,chosen_arms,
                                                                                         cumulative_reward,
                                                                                         context, another_context, G, alg)



            else:
                #如果i!=0，那么就需要自己获取config，其中 core_arm_orders, llc_arm_orders,mb_arm_orders记录着core、llc、mb
                #三种资源的各种排列组合方式的列举
                core_list, llc_config, mb_config = gen_config(chosen_arms, core_arm_orders, llc_arm_orders,
                                                              mb_arm_orders)
                time.sleep(1)
                #ipc: instructions per core
                context, another_context, th_reward, fair_reward, p95_list = get_now_ipc(lc_app, bg_app, load_list, core_list, evne)
                store_file = save_file(app_id, load_list, th_reward, fair_reward, context, chosen_arms[0][app_id[0]],
                                       list(chosen_arms[1].values()),
                                       list(chosen_arms[2].values()))

                f_d_w.writerow(store_file)

                chose_arm_storage.append([core_list, llc_config, mb_config])
                
                #调整bandit
                if alg == "throughput" and th_reward>th_reward_baseline:  #th转fair
                    alg = "fair"
                    last_th_reward = th_reward
                    fair_reward = last_fair_reward  #使用上一次转换前遗留的reward
                elif alg == "fair" and fair_reward>fair_reward_baseline:  #fair转th
                    alg = "throughput"
                    last_fair_reward = fair_reward
                    th_reward = last_th_reward

            if "mab_2" in locals().keys():
                if "mab_3" in locals().keys():
                    reward_arms_1, chosen_arms_1, cumulative_reward, G = onlineEvaluate(mab_1, th_reward, fair_reward,
                                                                                                 reward_arms,chosen_arms,
                                                                                                 cumulative_reward,
                                                                                                 context,
                                                                                                 another_context,
                                                                                                 G,
                                                                                                 alg)
                    reward_arms_2, chosen_arms_2, cumulative_reward, G = onlineEvaluate(mab_2, th_reward, fair_reward,
                                                                                                 reward_arms,chosen_arms,
                                                                                                 cumulative_reward,
                                                                                                 context,
                                                                                                 another_context,
                                                                                                 G,
                                                                                                 alg)
                    reward_arms_3, chosen_arms_3, cumulative_reward, G = onlineEvaluate(mab_3, th_reward, fair_reward,
                                                                                                 reward_arms,chosen_arms,
                                                                                                 cumulative_reward,
                                                                                                 context,
                                                                                                 another_context,
                                                                                                 G,
                                                                                                 alg)
                    #mab123投票决策
                    if chosen_arms_1 == chosen_arms_2:
                        chosen_arms = chosen_arms_1
                        reward_arms = reward_arms_1

                    elif chosen_arms_1 == chosen_arms_3:
                        chosen_arms = chosen_arms_3
                        reward_arms = reward_arms_3
                    elif chosen_arms_2 == chosen_arms_3:
                        chosen_arms = chosen_arms_2
                        reward_arms = reward_arms_2
                    else:       #否则随机选择一个动作
                        random_index = random.randrange(3)
                        tmp = [chosen_arms_1,chosen_arms_2,chosen_arms_3]
                        chosen_arms = tmp[random_index]
                        tmp = [reward_arms_1,reward_arms_2,reward_arms_3]
                        reward_arms = tmp[random_index]
                else:
                    reward_arms_1, chosen_arms_1, cumulative_reward, G = onlineEvaluate(mab_1, th_reward, fair_reward, reward_arms,chosen_arms,
                                                                                             cumulative_reward,
                                                                                             context, another_context,
                                                                                             G, alg)
                    reward_arms_2, chosen_arms_2, cumulative_reward, G = onlineEvaluate(mab_2, th_reward, fair_reward, reward_arms,chosen_arms,
                                                                                             cumulative_reward,
                                                                                             context, another_context,
                                                                                             G, alg)
                    if random.randint(0,1) == 0:
                        chosen_arms = chosen_arms_1
                        reward_arms = reward_arms_1
                    else:
                        chosen_arms = chosen_arms_2
                        reward_arms = reward_arms_2
            else:
                reward_arms, chosen_arms, cumulative_reward, G = onlineEvaluate(mab_1, th_reward, fair_reward, reward_arms,chosen_arms,
                                                                                         cumulative_reward,
                                                                                         context, another_context, G, alg)

            use_time = (datetime.datetime.now() - now).seconds
            #p95_list：输出实验的latency和reward情况
            if th_reward > 0:
                p95_list.extend([alg, th_reward, fair_reward, use_time, i])
                f_w.writerow(p95_list)
                logging.error("p95_list,{}".format(p95_list))
            else:
                f_w.writerow(p95_list)
                logging.error("p95_list,{}".format(p95_list))

            print(f"{i}th,{th_reward}")

        #运行完成之后，从onlineEvaluate返回的每一轮交互得到的reward列表中，选择出最大的reward，以及它对应的那个决策方案
        best_reward_id = np.argmax(reward_arms,axis=0)
        th_best_config = chose_arm_storage[best_reward_id[0]-1]
        th_best_reward = reward_arms[best_reward_id[0]][0]
        fair_best_config = chose_arm_storage[best_reward_id[1]-1]
        fair_best_reward = reward_arms[best_reward_id[1]][1]

        # subprocess.call("sudo kill -9 $(ps -ef|grep /tmp/tailbench.inputs/|grep -v grep|awk '{print $2}')",shell=True)
        subprocess.call("sudo kill -9 $(ps -ef|grep /tmp/parsec-3.0/pkgs/|grep -v grep|awk '{print $2}')",
                        shell=True)

        print(f"best th_config {th_best_config}, best th_reward {th_best_reward}, best fair_config {fair_best_config}, best fair_reward {fair_best_reward}")
        print(f"last config {core_list},{llc_config},{mb_config},{load_list}, last th_reward {th_reward}, last fair_reward {fair_reward}")

        use_time = (datetime.datetime.now() - now).seconds
        print(f'Mean reward of LinUCB with th_alpha = {th_alpha}, fair_reward = {fair_alpha} is: ', np.mean(reward_arms))
        print("use_time", use_time)


def onlineEvaluate(mab, th_reward, fair_reward, reward_arms, chosen_arms,cumulative_reward, context, another_context, G, alg):
    """
    根据环境交互得到的reward, context,other_context，对A_c，b_c，A_l，b_l，A_b，b_b这些参数进行更新
    根据上一轮运行获得的性能指标，以及不断更新的A,b两类参数，计算p_*_t，分析得到每个app的下一步的core_action, llc_action, band_action，
    同时对reward_arms cumulative_reward, G进行更新

    :param mab: LinUCB_th类的对象
    :param rewards: 上一轮运行得到的reward，ipc/delay
    :param contexts: 记录每个app的22个性能数据，counter
    :param reward_arms:列表，记录每一次交互的reward
    :param G:储存当前所有交互的reward之和
    :param cumulative_reward:列表，记录每一次交互后的G
    :return:更新后的reward_arms cumulative_reward, G，以及选择core action后的chosen_arms
    """
    #chosen_arms =[core_action, llc_action, band_action] = [{app1:int,app2:int ,...},   {app1:[int,int],app2:[int,int]...},   {app1:int,app2:int,...}]
    #根据环境交互得到的reward, context,other_context，对A_c，b_c，A_l，b_l，A_b，b_b这些参数进行更新
    mab.update(chosen_arms[0], chosen_arms[1], chosen_arms[2], th_reward, fair_reward, context, another_context, alg)
    #根据上一轮运行获得的性能指标，以及不断更新的A,b两类参数，计算p_*_t，
    #分析得到每个app的下一步的core_action, llc_action, band_action，
    core_action, llc_action, band_action = mab.play(context,another_context, alg)

    # choose a core action
    core_compare = []   
    for key in core_action.keys():
        core_compare.append(core_action[key])
    tmp = dict(Counter(core_compare))
    tmp = [key for key, value in tmp.items() if value > 1]  

    if random.randint(1,10) > 8: #概率0.2
        core_final_arm_id = random.choice(core_compare)
    else:
        if tmp != []:
            core_final_arm_id = random.choice(tmp)
        else:
            core_final_arm_id = random.choice(core_compare)
    for key in core_action.keys():
        core_action[key] = core_final_arm_id


    reward_arms.append([th_reward,fair_reward])

    # mab.update(core_action, llc_action, band_action, th_reward, context, another_context)
    G += [th_reward,fair_reward]
    cumulative_reward.append(G)

    chosen_arms = [core_action, llc_action, band_action]

    return reward_arms,chosen_arms, cumulative_reward, G

def get_name_list(conum):
    f_name = open(f"/home/crb/all_random/{conum}_test/classify_{conum}_bcname_new.csv", "r")

    f_name_r = csv.reader(f_name)
    f_name_list = []
    for row in f_name_r:
        f_name_list.append(row[:-1])
    return f_name_list


'''
evne：环境
f_d_w：写入文件到vote_sample_times_data_1.csv
f_w：写入文件到vote_sample_times_1.csv
'''

if __name__ == "__main__":
    core_arm_orders = arm_cor_numapp()  #返回一个dict，里面记录每一种colocation size下所有的可能性
    llc_arm_orders, mb_arm_orders = get_llc_bandwith_config()
    evne = []
    alg = "throughput"

    with open("/home/crb/CPU2017/micro_event_we_choose.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            evne.append(line)


    # sample times
    logging.basicConfig(
        filename=f"/home/crb/bandit_clite/qos_result/bandit/vote_sample_times_1.txt",
        level=logging.ERROR)
    f_d = open("/home/crb/bandit_clite/qos_result/bandit/vote_sample_times_data_1.csv", "w", newline="")
    f_d_w = csv.writer(f_d)

    f = open("/home/crb/bandit_clite/qos_result/bandit/vote_sample_times_1.csv", "w", newline="")
    f_w = csv.writer(f)
    colocation_list = [['img-dnn', 'masstree', 'moses', 'fluidanimate']]  ##colocation_list[list[]],每一个list[]中都记录着一组需要共定位的app


    now = datetime.datetime.now()
    train_success(load_list_i=0, load_list_m=0)
    
    time.sleep(5)
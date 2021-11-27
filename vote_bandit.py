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
from bandit.get_arm_qos import update_core_arm_context, arm_cor_numapp, get_llc_bandwith_config
from bandit.run_and_getconfig_qos import gen_config, get_now_ipc, run_bg_benchmark, gen_init_config,LC_APP_NAMES
from bandit.store_data import save_file
# Info: arm分几层 各种各存一个矩阵 一个个决定后给出最后决策 reward是一样的所有是同时决定
# cache:55 arms/per app
# app band :10 arms/per app
# cpu arm 扩容方便

# 1_st: 只用上部分历史数据, 下一个colocation没出现的就删掉



class LinUCB():

    def __init__(self, ndims, alpha, app_id, core_arm_orders, llc_narms=55, band_namrms=10):
        self.num_app = len(app_id)
        assert 2 <= self.num_app <= 7, 'the num of colocation size is wrong'
        self.all_core_narms_p = {2: 8, 3: 34, 4: 92, 5: 130, 6: 86, 7: 28}
        self.core_narms = self.all_core_narms_p[self.num_app]
        self.app_id = app_id
        self.llc_narms = llc_narms
        self.band_namrms = band_namrms
        # number of context features
        self.ndims = ndims
        # explore-exploit parameter
        self.alpha = alpha
        self.core_arm_orders = core_arm_orders

        self.A_c = {}
        self.b_c = {}
        self.p_c_t = {}

        self.A_l = {}
        self.b_l = {}
        self.p_l_t = {}

        self.A_b = {}
        self.b_b = {}
        self.p_b_t = {}

        for i in app_id:
            self.A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims* 2))
            self.b_c[i] = np.zeros((self.core_narms, self.ndims* 2, 1))
            self.p_c_t[i] = np.zeros(self.core_narms)

            self.A_l[i] = np.zeros((self.llc_narms, self.ndims* 2, self.ndims* 2))
            self.b_l[i] = np.zeros((self.llc_narms, self.ndims* 2, 1))
            self.p_l_t[i] = np.zeros(self.llc_narms)

            self.A_b[i] = np.zeros((self.band_namrms, self.ndims* 2, self.ndims* 2))
            self.b_b[i] = np.zeros((self.band_namrms, self.ndims* 2, 1))
            self.p_b_t[i] = np.zeros(self.band_namrms)
            for arm in range(self.core_narms):
                self.A_c[i][arm] = np.eye(self.ndims* 2)

            for arm in range(self.llc_narms):
                self.A_l[i][arm] = np.eye(self.ndims* 2)

            for arm in range(self.band_namrms):
                self.A_b[i][arm] = np.eye(self.ndims* 2)

        super().__init__()
        return

    def add_del_app(self, app_id):
        assert 2 <= self.num_app <= 7, 'the num of colocation size is wrong'
        A_c, A_l, A_b = 0, 0, 0
        for i in self.A_c.keys():
            A_c += self.A_c[i]
            A_l += self.A_l[i]
            A_b += self.A_b[i]



        self.core_narms = self.all_core_narms_p[len(app_id)]

        for i in app_id:
            if i not in self.A_c.keys():
                self.A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
                self.b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
                self.p_c_t[i] = np.zeros((self.core_narms))
                for arm in range(self.core_narms):
                    self.A_c[i][arm] = A_c[arm] / self.num_app

                self.A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
                self.b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
                self.p_l_t[i] = np.zeros((self.llc_narms))
                for arm in range(self.llc_narms):
                    self.A_l[i][arm] = A_l[arm] / self.num_app

                self.A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
                self.b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
                self.p_b_t[i] = np.zeros((self.band_namrms))
                for arm in range(self.band_namrms):
                    self.A_b[i][arm] = A_b[arm] / self.num_app
            # if i not in self.A_c.keys():
            #     self.A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
            #     self.b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
            #     self.p_c_t[i] = np.zeros((self.core_narms))
            #     for arm in range(self.core_narms):
            #         self.A_c[i][arm] = np.eye(self.ndims * 2)
            #
            #     self.A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
            #     self.b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
            #     self.p_l_t[i] = np.zeros((self.llc_narms))
            #     for arm in range(self.llc_narms):
            #         self.A_l[i][arm] = np.eye(self.ndims * 2)
            #
            #     self.A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
            #     self.b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
            #     self.p_b_t[i] = np.zeros((self.band_namrms))
            #     for arm in range(self.band_namrms):
            #         self.A_b[i][arm] = np.eye(self.ndims * 2)

        self.A_c, self.b_c, self.p_c_t = update_core_arm_context(self.app_id, app_id, self.core_arm_orders, self.A_c,
                                                                 self.b_c, self.p_c_t,
                                                                 self.ndims * 2)
        self.num_app = len(app_id)
        self.app_id = app_id

    def play(self, context,other_context):
        assert len(context[self.app_id[0]]) == self.ndims, 'the shape of context size is wrong'
        core_action = {}
        llc_action = {}
        band_action = {}
        contexts = {}
        # gains per each arm
        # only calculate the app in this colocation
        for key in self.app_id:
            A = self.A_c[key]
            b = self.b_c[key]
            contexts[key] = np.hstack((context[key],other_context[key]))

            for i in range(self.core_narms):
                # initialize theta hat

                theta = inv(A[i]).dot(b[i])
                # get context of each arm from flattened vector of length 100
                cntx = np.array(contexts[key])
                # get gain reward of each arm
                self.p_c_t[key][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(A[i]).dot(cntx)))

            core_action[key] = np.random.choice(np.where(self.p_c_t[key] == max(self.p_c_t[key]))[0])

            A = self.A_l[key]
            b = self.b_l[key]
            for i in range(self.llc_narms):
                theta = inv(A[i]).dot(b[i])
                cntx = np.array(contexts[key])
                self.p_l_t[key][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(A[i]).dot(cntx)))

            llc_action[key] = np.random.choice(np.where(self.p_l_t[key] == max(self.p_l_t[key]))[0])

            A = self.A_b[key]
            b = self.b_b[key]
            for i in range(self.band_namrms):
                theta = inv(A[i]).dot(b[i])
                cntx = np.array(contexts[key])
                self.p_b_t[key][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(A[i]).dot(cntx)))
            band_action[key] = np.random.choice(np.where(self.p_b_t[key] == max(self.p_b_t[key]))[0])

        return core_action, llc_action, band_action

    def update(self, core_arms, llc_arms, band_arms, reward, context,other_context):
        contexts ={}
        print(core_arms)
        for key in self.app_id:
            arm = core_arms[key]


            contexts[key] = np.hstack((context[key],other_context[key]))

            self.A_c[key][arm] += np.outer(np.array(contexts[key]),
                                           np.array(contexts[key]))

            self.b_c[key][arm] = np.add(self.b_c[key][arm].T,
                                        np.array(contexts[key]) * reward).reshape(
                self.ndims*2, 1)

            arm = llc_arms[key]
            self.A_l[key][arm] += np.outer(np.array(contexts[key]),
                                           np.array(contexts[key]))
            self.b_l[key][arm] = np.add(self.b_l[key][arm].T,
                                        np.array(contexts[key]) * reward).reshape(
                self.ndims*2, 1)

            arm = band_arms[key]
            self.A_b[key][arm] += np.outer(np.array(contexts[key]),
                                           np.array(contexts[key]))
            self.b_b[key][arm] = np.add(self.b_b[key][arm].T,
                                        np.array(contexts[key]) * reward).reshape(
                self.ndims*2, 1)



def train_success(load_list_i,load_list_m,pre_rounds=100,rounds=100):
    nof_counters = 22
    nof_colocation = len(colocation_list)

    init_alg = "fair"

    alpha = 0.01
    mab_1 = LinUCB(nof_counters, alpha, colocation_list[0], core_arm_orders)

    store_dict = {}.fromkeys(["app_id", "load_list", "reward", "core_config", "llc_config", "mb_config", "counters","other_counters"])

    for col_items in range(nof_colocation):
        now = datetime.datetime.now()

        app_id = colocation_list[col_items]

        lc_app, bg_app = [], []
        for i in app_id:
            if i in LC_APP_NAMES:
                lc_app.append(i)
            else:
                bg_app.append(i)

        print(f"Start run {col_items}th")


        chose_arm_storage = []
        reward_arms = []
        cumulative_reward = []
        G = 0
        history = []

        core_list, llc_config, mb_config, chosen_arms = gen_init_config(app_id, core_arm_orders, llc_arm_orders,
                                                                        alg=init_alg)

        if bg_app != []:
            run_bg_benchmark(bg_app, core_list[len(lc_app):])


        if col_items!= nof_colocation-1:
            nrounds = pre_rounds
        else:
            nrounds = rounds

        load_list = [load_list_i] *(len(colocation_list[col_items])-1)
        load_list.append(load_list_m)

        f_w.writerow([colocation_list[col_items],load_list])
        logging.error("colocation,load_list, {} {}".format(colocation_list[col_items],load_list))
        for i in range(nrounds):
            if nrounds % 50 == 0:
                if "mab_2" in locals().keys():
                    if "mab_3" in locals().keys():
                        mab_1 = LinUCB(nof_counters, alpha, colocation_list[col_items], core_arm_orders)
                    else:
                        mab_3 = LinUCB(nof_counters, alpha, colocation_list[col_items], core_arm_orders)
                else:
                    mab_2 = LinUCB(nof_counters, alpha, colocation_list[col_items], core_arm_orders)

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
                context, another_context, reward, p95_list = get_now_ipc(lc_app, bg_app, load_list, core_list, evne)

                store_file = save_file(app_id, load_list, reward, context, chosen_arms[0][app_id[0]], list(chosen_arms[1].values()),
                                              list(chosen_arms[2].values()))

                f_d_w.writerow(store_file)

                chose_arm_storage.append([core_list, llc_config, mb_config])
                reward_arms, chosen_arms, cumulative_reward, G = onlineEvaluate(mab_1, reward, reward_arms,chosen_arms,
                                                                                         cumulative_reward,
                                                                                         context, another_context, G)



            else:

                core_list, llc_config, mb_config = gen_config(chosen_arms, core_arm_orders, llc_arm_orders,
                                                              mb_arm_orders)
                time.sleep(1)
                context, another_context, reward,p95_list = get_now_ipc(lc_app, bg_app, load_list, core_list, evne)
                store_file = save_file(app_id, load_list, reward, context, chosen_arms[0][app_id[0]],
                                       list(chosen_arms[1].values()),
                                       list(chosen_arms[2].values()))

                f_d_w.writerow(store_file)

                chose_arm_storage.append([core_list, llc_config, mb_config])


            if "mab_2" in locals().keys():
                if "mab_3" in locals().keys():
                    reward_arms_1, chosen_arms_1, cumulative_reward, G = onlineEvaluate(mab_1, reward,
                                                                                                 reward_arms,chosen_arms,
                                                                                                 cumulative_reward,
                                                                                                 context,
                                                                                                 another_context,
                                                                                                 G)
                    reward_arms_2, chosen_arms_2, cumulative_reward, G = onlineEvaluate(mab_2, reward,
                                                                                                 reward_arms,chosen_arms,
                                                                                                 cumulative_reward,
                                                                                                 context,
                                                                                                 another_context,
                                                                                                 G)
                    reward_arms_3, chosen_arms_3, cumulative_reward, G = onlineEvaluate(mab_3, reward,
                                                                                                 reward_arms,chosen_arms,
                                                                                                 cumulative_reward,
                                                                                                 context,
                                                                                                 another_context,
                                                                                                 G)

                    if chosen_arms_1 == chosen_arms_2:
                        chosen_arms = chosen_arms_1
                        reward_arms = reward_arms_1

                    elif chosen_arms_1 == chosen_arms_3:
                        chosen_arms = chosen_arms_3
                        reward_arms = reward_arms_3
                    elif chosen_arms_2 == chosen_arms_3:
                        chosen_arms = chosen_arms_2
                        reward_arms = reward_arms_2
                    else:
                        random_index = random.randrange(3)
                        tmp = [chosen_arms_1,chosen_arms_2,chosen_arms_3]
                        chosen_arms = tmp[random_index]
                        tmp = [reward_arms_1,reward_arms_2,reward_arms_3]
                        reward_arms = tmp[random_index]
                else:
                    reward_arms_1, chosen_arms_1, cumulative_reward, G = onlineEvaluate(mab_1, reward, reward_arms,chosen_arms,
                                                                                             cumulative_reward,
                                                                                             context, another_context,
                                                                                             G)
                    reward_arms_2, chosen_arms_2, cumulative_reward, G = onlineEvaluate(mab_2, reward, reward_arms,chosen_arms,
                                                                                             cumulative_reward,
                                                                                             context, another_context,
                                                                                             G)
                    if random.randint(0,1) == 0:
                        chosen_arms = chosen_arms_1
                        reward_arms = reward_arms_1
                    else:
                        chosen_arms = chosen_arms_2
                        reward_arms = reward_arms_2
            else:
                reward_arms, chosen_arms, cumulative_reward, G = onlineEvaluate(mab_1, reward, reward_arms,chosen_arms,
                                                                                         cumulative_reward,
                                                                                         context, another_context, G)

            use_time = (datetime.datetime.now() - now).seconds
            if reward > 0:
                p95_list.extend([reward, use_time, i])
                f_w.writerow(p95_list)
                logging.error("p95_list,{}".format(p95_list))

            else:
                f_w.writerow(p95_list)
                logging.error("p95_list,{}".format(p95_list))

            print(f"{i}th,{reward}")

        best_reward_id = np.argmax(reward_arms)
        best_config = chose_arm_storage[best_reward_id-1]
        best_reward = reward_arms[best_reward_id]

        # subprocess.call("sudo kill -9 $(ps -ef|grep /tmp/tailbench.inputs/|grep -v grep|awk '{print $2}')",shell=True)
        subprocess.call("sudo kill -9 $(ps -ef|grep /tmp/parsec-3.0/pkgs/|grep -v grep|awk '{print $2}')",
                        shell=True)

        print(f"best config {best_config}, best reward {best_reward}")
        print(f"last config {core_list},{llc_config},{mb_config},{load_list}, last reward {reward}")

        use_time = (datetime.datetime.now() - now).seconds
        print(f'Mean reward of LinUCB with alpha = {alpha} is: ', np.mean(reward_arms))
        print("use_time", use_time)


def onlineEvaluate(mab, reward, reward_arms,chosen_arms,cumulative_reward, context, another_context, G):
    """
    :param mab:
    :param rewards: ipc/delay
    :param contexts: counter
    :param nrounds:
    :return:
    """

    mab.update(chosen_arms[0], chosen_arms[1], chosen_arms[2], reward, context, another_context)

    core_action, llc_action, band_action = mab.play(context,another_context)

    # choose a core action
    core_compare = []
    for key in core_action.keys():
        core_compare.append(core_action[key])
    tmp = dict(Counter(core_compare))
    tmp = [key for key, value in tmp.items() if value > 1]

    if random.randint(1,10) > 8:
        core_final_arm_id = random.choice(core_compare)
    else:
        if tmp != []:
            core_final_arm_id = random.choice(tmp)
        else:
            core_final_arm_id = random.choice(core_compare)
    for key in core_action.keys():
        core_action[key] = core_final_arm_id


    reward_arms.append(reward)

    # mab.update(core_action, llc_action, band_action, reward, context, another_context)
    G += reward
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




if __name__ == "__main__":
    core_arm_orders = arm_cor_numapp()
    llc_arm_orders, mb_arm_orders = get_llc_bandwith_config()
    evne = []

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
    colocation_list = [['img-dnn', 'masstree', 'moses', 'fluidanimate']]


    now = datetime.datetime.now()
    train_success(load_list_i=0, load_list_m=0)
    time.sleep(5)
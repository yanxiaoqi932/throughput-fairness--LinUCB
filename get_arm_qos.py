# coding: utf-8
# Author: crb
# Date: 2021/7/19 14:13

import sys
import os
import numpy as np
import itertools

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


# Info:
class permuSolution():
    """
    给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
    """
    def permuteUnique(self, nums):
        nums.sort()
        self.res = []
        check = [0 for i in range(len(nums))]

        self.backtrack([], nums, check)
        return self.res

    def backtrack(self, sol, nums, check):
        if len(sol) == len(nums):
            self.res.append(sol)
            return

        for i in range(len(nums)):
            if check[i] == 1:
                continue
            if i > 0 and nums[i] == nums[i - 1] and check[i - 1] == 0:
                continue
            check[i] = 1
            self.backtrack(sol + [nums[i]], nums, check)
            check[i] = 0


def get_core_config(num_app,nof_cores=9):
    # 获得此种colocation size下的config，不重复
    # the core nums can be chose, we assume that the max colocation size is 7, the min is 2
    value_list = [1] * 6

    value_list = value_list + [2,3] * 4+ [4]*2+[ 5, 6, 7, 8]

    def n(num):
        count = 0
        while num:
            num &= (num - 1)
            count += 1

        return count

    bit = 1 << len(value_list)
    res = []

    for i in range(1, bit):
        if n(i) == num_app:
            s = 0
            temp = []

            for j in range(len(value_list)):
                if ((i & 1 << j) != 0):
                    s += value_list[j]
                    temp.append(value_list[j])
            if s == nof_cores:
                res.append(temp)

    res = np.array(res)
    x = 0
    a = [3, 5, 7, 11, 13, 17, 19]
    for ii in range(num_app):
        x += res[:, ii] * a[ii] ** ii

    idx = np.unique(x, return_index=True)[1]
    res = res[idx]

    return res


def arm_cor_numapp():
    '''
    每一种colocation size下所有的可能性，即2app,[4,5],[5,4],...,这样
    这样给出固定顺序后方便给实际配置时候的映射
    {2: [[4, 5], [5, 4], [3, 6], [6, 3], [2, 7], [7, 2], [1, 8], [8, 1]],
    3: [[2, 3, 4], [2, 4, 3], [3, 2, 4], [3, 4, 2], [4, 2, 3], [4, 3, 2], ...[]],
    ...}
    '''
    
    a = permuSolution()  #给定一个可包含重复数字的序列 nums ，按任意顺序返回所有不重复的全排列
    arm_init_orders = {}
    for j in range(2, 8):
        core_config = get_core_config(j) #core_config=list[list[]]
        core_arm = []
        for i in core_config:
            core_arm.extend(a.permuteUnique(i)) #extend()函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
        arm_init_orders[j] = core_arm

    return arm_init_orders


def get_key(dict, value):
    '''
    返回dict中与value值相同的v对应的key
    '''
    for k, v in dict.items():
        if v == value:
            return k
def update_core_arm_context(old, new, arm_init_orders, th_A, th_b, th_p, fair_A, fair_b, fair_p, counter_size):
    """
    :param old: old app id
    :param new: new app id
    :param arm_init_oreders: 所有cpu核共定位的组合排列 {2:[[4,5],[5,4],...], 3:[[]], ...}
    :return:根据旧的A, b, p参数，对新的app对应的 A, b, p参数进行赋值， A, b, p中包含了所有app的参数值，尽可能地进行了更新
    """
    if old.sort() == new.sort():
        return th_A,th_b,th_p,fair_A, fair_b, fair_p

    # all_core_narms_p = {2: 8, 3: 21, 4: 28, 5: 25, 6: 36, 7: 7}
    all_core_narms_p = {2: 8, 3: 34, 4: 88, 5: 125, 6: 86, 7: 28}

    old_core_narms = all_core_narms_p[len(old)]
    new_core_narms = all_core_narms_p[len(new)]
    # loc to verify actual config, because core config is given by fixed matrix
    # like [5,1,1,1],if app 0 still alives, the result with 5 should be left
    location = []  #记录旧的app中，与新app重叠的app的index
    for i in new:
        for j in range(len(old)):
            if i == old[j]:
                location.append(j)
    th_new_A = {}
    th_new_b = {}
    th_new_p = {}
    fair_new_A = {}
    fair_new_b = {}
    fair_new_p = {}
    for i in new:
        th_new_A[i] = np.zeros((new_core_narms, counter_size, counter_size))  #counter_size=self.ndims * 2
        th_new_b[i] = np.zeros((new_core_narms, counter_size, 1))
        th_new_p[i] = np.zeros(new_core_narms)
        fair_new_A[i] = np.zeros((new_core_narms, counter_size, counter_size))  #counter_size=self.ndims * 2
        fair_new_b[i] = np.zeros((new_core_narms, counter_size, 1))
        fair_new_p[i] = np.zeros(new_core_narms)

        for arm in range(new_core_narms):
            th_new_A[i][arm] = np.eye(counter_size)
            fair_new_A[i][arm] = np.eye(counter_size)

    count = 0
    for key in th_new_A.keys():
        #如果新的app中，有部分决策与旧的app相同，那么可以直接采用旧app的A, b, p参数
        if key in old:
            loc = location[count]
            tmp = []
            for i in range(new_core_narms):
                for j in range(old_core_narms):
                    for m in range(len(new)):
                        if m > len(old):
                            break
                        #arm_init_orders[len(new)][i][m]中，len(new)代表多少个app共定位（cpu分成几份），
                        # i代表选择哪种分配方式（决策），m代表查看第m个app分配到的cpu核数
                        if arm_init_orders[len(new)][i][m] == arm_init_orders[len(old)][j][loc] and (j not in tmp):
                            tmp.append(j)
                            th_new_A[key][i] = th_A[key][j]
                            th_new_b[key][i] = th_b[key][j]
                            th_new_p[key][i] = th_p[key][j]
                            fair_new_A[key][i] = fair_A[key][j]
                            fair_new_b[key][i] = fair_b[key][j]
                            fair_new_p[key][i] = fair_p[key][j]
                            break

            if count < len(location) - 1:
                count += 1
        else:
            history_core_num = th_A[key].shape[0]
            num_of_app_of_this_history = get_key(all_core_narms_p,history_core_num)

            tmp = []
            for i in range(new_core_narms):
                for j in range(history_core_num):
                    for m in range(len(new)):
                        if m > num_of_app_of_this_history:
                            break
                        if arm_init_orders[len(new)][i][m] == arm_init_orders[num_of_app_of_this_history][j][m] and (j not in tmp):
                            tmp.append(j)
                            th_new_A[key][i] = th_A[key][j]
                            th_new_b[key][i] = th_b[key][j]
                            th_new_p[key][i] = th_p[key][j]
                            fair_new_A[key][i] = fair_A[key][j]
                            fair_new_b[key][i] = fair_b[key][j]
                            fair_new_p[key][i] = fair_p[key][j]
                            break


    for key in th_A.keys():
        if key not in new_A.keys():
            th_new_A[key] = th_A[key]
            th_new_b[key] = th_b[key]
            th_new_p[key] = th_p[key]
            fair_new_A[key] = fair_A[key]
            fair_new_b[key] = fair_b[key]
            fair_new_p[key] = fair_p[key]


    return th_new_A, th_new_b, th_new_p, fair_new_A, fair_new_b, fair_new_p


def get_llc_bandwith_config():
    nof_llc = 10
    nof_band = 10
    llc_config = []
    mb_config = []
    for i in range(1, nof_llc + 1):
        for j in range(i, nof_llc + 1):
            llc_config.append([i, j])
    for i in range(1, nof_band + 1):
        mb_config.append(i)
    return llc_config, mb_config



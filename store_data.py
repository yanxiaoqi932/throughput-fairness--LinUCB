# coding: utf-8
# Author: crb
# Date: 2021/8/29 0:41
import sys
import os
import csv
from numpy import array
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from run_and_getconfig_qos import app_refer_id as ben_id
# Info: store history data

def ben_cor_id():
    """
    :return: ben_name -> id
    """
    bench_name = ["specjbb", "masstree", "xapian", "sphinx", "moses", "img-dnn"]
    id = [i for i in range(len(bench_name))]
    d = dict(zip(bench_name,id))
    return d


app_refer_id = {"specjbb":0, "masstree":1, "xapian":2, "sphinx":3, "moses":4, "img-dnn":5,
                'blackscholes':6,
               'canneal':7,
               'fluidanimate':8,
               'freqmine':9,
               'streamcluster':10,
               'swaptions':11
}

def save_file(app_id,load_list,th_reward,fair_reward,context,core_list,llc_config,mb_config):
    out_list = []
    app_num_id =[]
    add_zero = [0 for i in range(7-len(app_id))]
    for i in app_id:
        app_num_id.append(app_refer_id[i])
    out_list.extend(app_num_id)
    out_list.extend(add_zero)

    out_list.extend(load_list)
    out_list.extend(add_zero)

    tmp = []
    for i in context.values():
        i = i.tolist()
        tmp.extend(i)
    out_list.extend(tmp)
    num = (7-len(app_id))*22
    out_list.extend([0 for i in range(num)])
    # add 1 to differt to zero-padding
    out_list.append(core_list+1)


    out_list.extend([i+1 for i in llc_config])
    out_list.extend(add_zero)
    out_list.extend([i+1 for i in mb_config])
    out_list.append(th_reward)
    out_list.append(fair_reward)


    return out_list
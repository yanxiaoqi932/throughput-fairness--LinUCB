# coding: utf-8
# Author: crb
# Date: 2021/7/18 23:34
import datetime
import sys
import numpy as np
import time
from scipy import stats
import os
import subprocess
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# Info:

LC_APP_NAMES     = [
                'masstree',
                'xapian'  ,
                'img-dnn' ,
                'sphinx'  ,
                'moses'   ,
                'specjbb'
                ]

# QoS requirements of LC apps (time in ms)
LC_APP_QOSES     = {
                'masstree' : 100    ,#250.0 ,
                'xapian'   : 20    ,#100.0 ,
                'img-dnn'  : 50    ,#100.0 ,
                'sphinx'   : 1500  ,#2000.0,
                'moses'    : 500    ,#1000.0,
                'specjbb'  : 10      #10
    ,#10.0

                }
# QPS levels
LC_APP_QPSES     = {
                'masstree' : list(range(50, 550, 50))               ,
                'xapian'   : list(range(80, 880, 80))               ,
                'img-dnn'  : list(range(140, 1540, 140))            ,
                'sphinx'   : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                'moses'    : list(range(10, 110, 10))             ,
                'specjbb'  : list(range(300, 3300, 300))
                }

# BG apps
BCKGRND_APPS  = [
                'blackscholes' ,
                'canneal'      ,
                'fluidanimate' ,
                'freqmine'     ,
                'streamcluster',
                'swaptions'
                ]

app_refer_id = {"specjbb":12, "masstree":1, "xapian":2, "sphinx":3, "moses":4, "img-dnn":5,
                'blackscholes':6,
               'canneal':7,
               'fluidanimate':8,
               'freqmine':9,
               'streamcluster':10,
               'swaptions':11
}

app_docker_dict = {
        'masstree' : "5069",
        'xapian'   : "3adb",
        'img-dnn'  : "306f",
        'sphinx'   : "f7eb",
        'moses'    : "ad58",
        'specjbb'  : "8900",
        'blackscholes' : "881f",
        'canneal'      : "95d1",
        'fluidanimate' : "b5ce",
        'freqmine'     : "cb3b",
        'streamcluster': "7bc0",
        'swaptions'    : "8b12"
      }

APP_docker_ppid = {
        'masstree' : "3135",
        'xapian'   : "3000",
        'img-dnn'  : "3248",
        'sphinx'   : "2881",
        'moses'    : "2604",
        'specjbb'  : "8900",
        'blackscholes' : "2476",
        'canneal'      : "2364",
        'fluidanimate' : "2211",
        'freqmine'     : "2056",
        'streamcluster': "1926",
        'swaptions'    : "1618"
      }

WR_MSR_COMM       = "wrmsr -a "
RD_MSR_COMM       = "rdmsr -a -u "

# MSR register requirements
IA32_PERF_GBL_CTR = "0x38F"  # Need bits 34-32 to be 1
IA32_PERF_FX_CTRL = "0x38D"  # Need bits to be 0xFFF
MSR_PERF_FIX_CTR0 = "0x309"


def get_LC_app_latency_and_judge(lc_app_name):
    """
    simple version: if one app met qos,reward = -1
    complex version:
    :param lc_app_name:
    :return:
    """

    def get_lat(dir):
        with open(dir, "r") as f:
            ff = f.readlines()
            assert "latency" in ff[0], "Lat file read failed!"
            a = ff[0].split("|")[0]
            lat = a[24:-3]
            return float(lat)
    tmp =[]

    flag = 0
    for i in lc_app_name:
        dir = f'/home/crb/bandit_clite/share_data/{i}.txt'
        while True:
            if os.path.exists(dir):
                if os.path.getsize(dir) != 0:
                    p95 = get_lat(dir)
                    tmp.append(p95)
                    break

        if p95 > LC_APP_QOSES[i]:
            # qos not guaranteed
            flag = 1
    subprocess.call("sudo rm /home/crb/bandit_clite/share_data/*.txt", shell=True)

    if flag == 1:
        return -1,tmp
    else:
        return 1,tmp






def run_bg_benchmark(bg_list,core_list):
    # 这个持久化运行
    total_command = []
    for i in range(len(bg_list)):
        command = f"docker exec {app_docker_dict[bg_list[i]]} taskset -c {core_list[i]} python /tmp/parsec-3.0/./run_crb.py {bg_list[i]} 8 &"
        total_command.append(command)

    subprocess.call(" ".join(total_command), shell=True, stdout= open(os.devnull, 'w'))
    # warm up
    time.sleep(20)


def run_lc_benchmark(lc_list, load_list,core_list):
    # inputs:['masstree','moses','img-dnn'],[1,2,5],["0-3","4-6","7-8"]
    # notes that: the values in load_list should begin from 0
    # 这个跑一次给一次arm
    total_command = []
    for i in range(len(lc_list)):
        print(i,lc_list[i],load_list[i])
        qps = LC_APP_QPSES[lc_list[i]][load_list[i]]
        cores = int(core_list[i][-1]) - int(core_list[i][0]) +1
        command = f"docker exec {app_docker_dict[lc_list[i]]} taskset -c {core_list[i]} python /tmp/tailbench-v0.9/{lc_list[i]}/testtt.py {qps} {cores} &"
        total_command.append(command)
    subprocess.call(" ".join(total_command), shell=True, stdout= open(os.devnull, 'w'))

def deal_features(features):
    l_9 = [0, 1, 3, 4, 5, 6, 12, 13, 16, 18,20]
    l_8 = [2, ]
    l_7 = [11, 14, 15, 19, ]
    l_6 = [7, 8, 10]
    l_5 = [9, ]
    l_4 = [17]
    l_3 = [11]
    out_dealed_features = []
    for i in range(len(features) // 22):
        tmp_features = features[i * 22:(i + 1) * 22]
        for i in range(len(tmp_features)):
            if i in l_9:
                tmp_features[i] /= 1e9
            elif i in l_8:
                tmp_features[i] /= 1e8
            elif i in l_7:
                tmp_features[i] /= 1e7
            elif i in l_6:
                tmp_features[i] /= 1e6
            elif i in l_5:
                tmp_features[i] /= 1e5
            elif i in l_4:
                tmp_features[i] /= 1e4
            elif i in l_3:
                tmp_features[i] /= 1e3
        out_dealed_features.extend(tmp_features)
    return out_dealed_features






def read_IPS_directlty(core_config):

    ipsP = os.popen("sudo " + RD_MSR_COMM + MSR_PERF_FIX_CTR0)

    # Calculate the IPS
    IPS = 0.0
    cor = [int(c) for c in core_config.split(',')]
    print("cor", cor)
    ind = 0
    for line in ipsP.readlines():
        if ind in cor:
            IPS += float(line)
        ind += 1

    return IPS


def get_now_ipc(lc_app,bg_app, load_list,core_list, evne):
    now = datetime.datetime.now()
    benchmark_list = lc_app + bg_app
    total_command = []
    event = ",".join(evne)
    run_lc_benchmark(lc_app,load_list,core_list[:len(lc_app)])

    evne_tmp = []
    insn_tmp = []

    for i in range(len(benchmark_list)):
        if i in LC_APP_NAMES:
            target = f"/tmp/tailbench.inputs/{benchmark_list[i]}/"
        else:
            if benchmark_list[i] == 'canneal' or benchmark_list[i] == 'streamcluster':
                target = f"/tmp/parsec-3.0/pkgs/kernels/{benchmark_list[i]}/"
            else:
                target = f"/tmp/parsec-3.0/pkgs/apps/{benchmark_list[i]}/"
        cmd_run = "sudo ps aux | grep {}".format(target)
        out = os.popen(cmd_run).read()

        if len(out.splitlines()) < 2:
            print("==============================rerun")
            run_bg_benchmark(bg_app, core_list[len(lc_app):])

        for name in evne:
            evne_tmp.append(name + "_" + str(benchmark_list[i]))
        insn_tmp.append(f"insn per cycle_{str(benchmark_list[i])}")

        subprocess.call(f'sudo taskset -apc {core_list[i]} {APP_docker_ppid[benchmark_list[i]]} > /dev/null', shell=True)
        perf_command = f"sudo perf stat -e {event} -C {core_list[i]} sleep 0.5"

        total_command.append(perf_command)
    while True:
        try:
            r = subprocess.run("&".join(total_command), shell=True, check=True,
                               capture_output=True)
            r_ = str(r.stderr.decode())

            rs = r_.split('\n')

            label = dict.fromkeys(insn_tmp, 0)

            d = dict.fromkeys(evne_tmp, 0)

            # 用于储存究竟属于哪个app
            flag = -1
            for index, line in enumerate(rs):
                rr = line.split(' ')
                rr = [i for i in rr if i != ""]

                if len(rr) < 2 or "elapsed" in rr:
                    continue

                if "Performance" in line:
                    cpu = line[39:-2]
                    for i in range(len(benchmark_list)):
                        if core_list[i] == cpu:
                            label[f"insn per cycle_{benchmark_list[i]}"] = float(rs[index + 3][55:59])
                            flag = benchmark_list[i]
                            break
                    continue

                key_name = rr[1] + "_" + str(flag)
                d[key_name] = float(rr[0].replace(",", ""))


            tmp = list(d.values())
            tmp_ = deal_features(tmp)
            feature_out = {}.fromkeys(benchmark_list)
            counters_without_self = {}.fromkeys(benchmark_list)
            for i in range(len(benchmark_list)):
                feature_out[benchmark_list[i]] = np.array(tmp_[22 * i:22 * (i + 1)])
                for j in range(len(benchmark_list)):
                    if i != j:
                        try:
                            if counters_without_self[benchmark_list[i]] == None:
                                counters_without_self[benchmark_list[i]] = np.array(tmp_[22 * j:22 * (j + 1)])
                        except:
                            counters_without_self[benchmark_list[i]] += np.array(tmp_[22 * j:22 * (j + 1)])

            now_ipc = list(label.values())

            Qos_cal,p95_list = get_LC_app_latency_and_judge(lc_app)

            if Qos_cal == -1:
                reward = -1
            else:
                if len(bg_app) == 0:
                    for i in range(len(benchmark_list)):
                        core = int(core_list[i][-1]) - int(core_list[i][0]) + 1
                        now_ipc[i] *= core

                    reward = sum(now_ipc)
                else:
                    for i in range(len(bg_app)):
                        core = int(core_list[len(lc_app)+i][-1]) - int(core_list[len(lc_app)+i][0]) + 1
                        now_ipc[i] *= core

                    reward = sum(now_ipc)

                    print("#################################", reward)


            if len(tmp) != 22 * len(benchmark_list) or len(now_ipc) != len(benchmark_list):
                print("ei")
                continue
            else:
                break
        except:
            print("emmm...")
            continue
    times =   datetime.datetime.now() -now
    print("==================================getnowipc",times)
    return feature_out, counters_without_self, reward,p95_list




def l_r_convert_config(left, right):
    if type(left) != int:
        try:
            left = int(left)
        except:
            left = int(float(left.strip('\'& \"')))
    if type(right) != int:
        try:
            right = int(right)
        except:
            right = int(float(right.strip('\'& \"')))

    bin_string = []
    for m in range(1, 12):
        if left <= m <= right:
            bin_string.append(1)
        else:
            bin_string.append(0)
    sum1 = bin_string[0] * 2 + bin_string[1]
    sum2 = bin_string[2] * 8 + bin_string[3] * 4 + bin_string[4] * 2 + bin_string[5]
    sum3 = bin_string[6] * 8 + bin_string[7] * 4 + bin_string[8] * 2 + bin_string[9]
    ans = "0x" + str(sum1) + str(hex(sum2)[-1]) + str(hex(sum3)[-1])
    return ans


def refer_core(core_config):
    # [2,4,3] => ["0,1","2,3,4,5","6,7,8"]
    app_cores = [""] * len(core_config)
    endpoint_left = 0
    for i in range(len(core_config)):
        endpoint_right = endpoint_left + core_config[i] - 1
        app_cores[i] = ",".join([str(c) for c in list(range(endpoint_left, endpoint_right+1))])
        endpoint_left = endpoint_right + 1
    assert endpoint_right == 8,f"print {app_cores},give wrong cpu config"
    return app_cores





def gen_init_config(app_id,core_arm_orders,llc_arm_orders,alg="fair"):
    app_num = len(app_id)
    nof_core = 9
    nof_llc = 10
    nof_mb = 10
    if alg == "fair":
        each_core_config = nof_core // app_num
        res_core_config = nof_core % app_num
        core_config = [each_core_config] * (app_num-1)
        if res_core_config >= each_core_config:
            for i in range(res_core_config):
                core_config[i]+=1
            core_config.append(1)
        else:
            core_config.append(each_core_config+res_core_config)

        core_list = refer_core(core_config)


        for i in range(len(core_arm_orders[app_num])):
            if core_arm_orders[app_num][i] == core_config:
                core_arms = dict.fromkeys(app_id,i)



        endpoint_left = 1
        each_llc_config = nof_llc // app_num
        llc_config = []
        llc_arms={}.fromkeys(app_id)

        for i in range(app_num):
            if i == app_num - 1:
                tmp_l = [endpoint_left, 10]
                llc_config.append(tmp_l)
                for j in range(len(llc_arm_orders)):
                    if llc_arm_orders[j] == tmp_l:
                        llc_arms[app_id[i]] = j
                break
            endpoint_right = endpoint_left + each_llc_config - 1
            tmp_l = [endpoint_left, endpoint_right]
            llc_config.append(tmp_l)
            for j in range(len(llc_arm_orders)):
                if llc_arm_orders[j] == tmp_l:
                    llc_arms[app_id[i]] = j
            endpoint_left = endpoint_right + 1


        each_mb_config = nof_mb // app_num
        res_mb_config = nof_mb % app_num
        mb_config = [each_mb_config] * (app_num-1)

        if res_mb_config > each_mb_config:
            for i in range(res_mb_config):
                mb_config[i] += 1
            mb_config.append(1)
        else:
            mb_config.append(each_mb_config + res_mb_config)

        arms = [i - 1 for i in mb_config]
        mb_arms = dict(zip(app_id, arms))
        print(core_arms)
        chosen_arms = [core_arms,llc_arms,mb_arms]


        for i in range(len(core_config)):
            subprocess.run('sudo pqos -a "llc:{}={}"'.format(i, core_list[i]), shell=True, capture_output=True)
            subprocess.run('sudo pqos -e "llc:{}={}"'.format(i, l_r_convert_config(llc_config[i][0], llc_config[i][1])),
                           shell=True, capture_output=True)
            subprocess.run('sudo pqos -e "mba:{}={}"'.format(i, int(float(mb_config[i])) * 10), shell=True,
                           capture_output=True)

        return core_list,llc_config,mb_config,chosen_arms



def gen_config(chosen_arms, core_arm_orders, llc_arm_orders, mb_arm_orders):

    core_arm, llc_arm, mb_arm = chosen_arms[0], chosen_arms[1], chosen_arms[2]
    core_config, llc_config, mb_config = [], [], []
    # core的臂选择给出一样结果，如果没有就选择第一个应用给的

    for key in core_arm.keys():
        llc_config.append(llc_arm_orders[llc_arm[key]])
        mb_config.append(mb_arm_orders[mb_arm[key]])

    core_config = core_arm_orders[len(core_arm.keys())][list(core_arm.values())[0]]

    core_list = refer_core(core_config)


    for i in range(len(core_config)):
        subprocess.run('sudo pqos -a "llc:{}={}"'.format(i, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "llc:{}={}"'.format(i, l_r_convert_config(llc_config[i][0], llc_config[i][1])),
                       shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "mba:{}={}"'.format(i, int(float(mb_config[i])) * 10), shell=True,
                       capture_output=True)
    return core_list, llc_config, mb_config
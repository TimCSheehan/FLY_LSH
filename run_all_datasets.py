import os
import numpy as np
import run_dist_func_v2


sav_root = '/home/tsheehan/py_code/dist_metrics'

def get_file_list():
    fls = np.array(os.listdir(sav_root))
    str_string = 'm_k_loop_'
    my_fls = np.where([t_str[:len(str_string)]==str_string for t_str in fls])[0]
    f_have = [fls[ff][len(str_string):-4] for ff in my_fls]
    return np.array(f_have)

# u_str = 'Erand'
# d1_want = np.array([20,40, 60, 100,200,300,400,500])
# n_want = np.array([50,100,150,200,300,400,500])
# d_sets_want = [u_str + str(x1) + '_' + str(x2) for x1 in d1_want for x2 in n_want]
u_str = 'MNIST'
n_want = np.array([50,100,150,200,300,400,500])
d_sets_want = [u_str  + '_' + str(x2) for x2 in n_want]

print('Planned Items:')
for item in d_sets_want:
    print(item)

for i in d_sets_want:
    f_list = get_file_list()
    if any(f_list==i):
            print(i + ' already complete!')
            continue
    else:
        dat_use,n = i.split('_')
        n_use = int(n)
        run_dist_func_v2.run_distance_analysis(dat_use,n_use)
        print(i + ' Complete!')
        
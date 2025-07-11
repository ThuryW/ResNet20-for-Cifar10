import torch
import sys
import numpy as np
import pandas as pd

torch.set_printoptions(threshold=sys.maxsize) # 显示所有的元素，不会有省略号出现
np.set_printoptions(threshold=np.inf)
checkpoint_path = '/home/wangtianyu/pytorch_resnet_cifar100/save_resnet32/model_70.20.th'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']

for name, param in state_dict.items():
    param_np = param.numpy()
    
    if len(param_np.shape) > 2:
        # 将高维数组展平成二维数组
        param_np = param_np.reshape(param_np.shape[0], -1)
    elif param_np.shape == ():
        # 将标量转换为二维数组
        param_np = param_np.reshape(1, 1)
    
    # 保存为 CSV 文件
    pd.DataFrame(param_np).to_csv(f'./weights_32_cifar100/{name}.csv', header=False, index=False)

## ResNet20 for Cifar10
### How to run
Just run
```shell
python train.py # --lr 0.001 ...
python test.py  # --path ./checkpoint/yourfilename
```
You can add args if you wanna change hyperparameters, see `train.py`.

Use `-r` to resume from checkpoint.

### Export weights
Save the model weights in csv files
```shell
python export_weights.py
```

### Train result

|Network | Top-1 accuracy |
|:------:|:------:|
|ResNet20| 92.23% |
|ResNet32| TBD |
|ResNet44| TBD |

### 训练神经网络的经验
- 其实不太需要用学习率调节器，手动中断调节就行了，一般用0.001、0.0001、0.00001三种（当然，也可以用StepLR实现此效果）
- 要有耐心，每种学习率的结果都尽量训练到足够高值再中断更新学习率
- 如果实在不放心，可以把认为理想的checkpoint先保留下来，下次resume
- 训练中后期准确率爬升会非常慢，要等训练集准确率上去了，测试准确率才能也爬升
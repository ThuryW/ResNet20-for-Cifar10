## ResNet20 for Cifar10
### How to run
Just run
```shell
python train.py # --lr 0.001 ...
python test.py
```
You can add args if you wanna change hyperparameters, see `train.py` 

Use `-r` to resume from checkpoint

### Export weights
Save the model weights in csv files
```shell
python export_weights.py
```
## Defence Project
Important: Before you start to run the code, download the cifar10 dataset for training, validating and testing by the file download_cifar10.sh.
Basically, you can use the commands: 
```
chmod +x download_cifar10.sh
./download_cifar10.sh
```
Please revise the `tasks/defense_project/predict.py` and `tasks/defense_project/train.py` files to implement different defense methods.

You can use the following commands to evaluate the result: `python defense_project-Evaluator.py`.

The defense model wil be saved under `models/defense_project-model.pth`.

Please upload the defense model `models/defense_homework-model.pth` and the PY files `tasks/defense_project/predict.py` and `tasks/defense_project/train.py` to the gradescope by the name `model.pth`, `predict.py` and `train.py` without being zipped.

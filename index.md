# Gru4Rec with/without Context in Tensorflow
It is the tensorflow version of the algorithm in ["Session-based Recommendations With Recurrent Neural Networks"](https://arxiv.org/abs/1511.06939 "Session-based Recommendations With Recurrent Neural Networks").
The code is based on the original source code at (https://github.com/hidasib/GRU4Rec). I also added a new module called Gru4recWithContext which is based on both user sequential item clicks and user context.

## Environment
You can install all the requirements in a Docker. Or just setup a virtual environment and run:
```bash
pip install -r requirements.txt
```
---
## Services
The module Gru4rec includes two modules, one is the original Gru4rec. The other is Gru4recWithContext that takes both
item click sequences and a variety of other information. 

## Data
Gru4rec takes only item click sequences, each row of data has three fields "SessionId", "ItemId", "Time". 
Gru4recWithContext takes additional information as well that are tagged with a label, for instance "Context".
 
## Run

### Original Gru4rec
Run training process by  passing data directory, checkpoint directory and data file name.
```bash
python ./run/training_gru4rec.py ./data/{} ./checkpoints/ sample_data.csv
```
Run evaluation by passing data directory, checkpoint directory, data file name, and batch size.
```bash
python ./run/eval_gru4rec.py ./data/{} ./checkpoints/ sample_data.csv 400
```
### Gru4recWithContext
Run training process by  passing data directory, checkpoint directory and data file name.
```bash
python ./run/training_gru4rec_with_context.py "Context" ./data/{} ./checkpoints/ sample_data.csv
```
Run evaluation by passing data directory, checkpoint directory, data file name, and batch size.
```bash
python ./run/eval_gru4rec_with_context.py "Context" ./data/{} ./checkpoints/ sample_data.csv 400
```
### Prediction
You can set up an AWS Lambda for predicting user item scores using Gru4recNp module. Given setting up a Lambda function with Tensorflow is not straight forward.

### Customized Optimizer
I also wrote RMSPropWithMomentum module which is the optimizer used in the original paper. 

## Literature
This code is based on the following papers
- ["Session-based Recommendations With Recurrent Neural Networks"](https://arxiv.org/abs/1511.06939).
- ["Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"](https://arxiv.org/abs/1706.03847).

# Version 2 code

## model training

No overlap BLSTM 1e-1 0.005
No overlap LSTM 1e-3 0.002


### My issues
1. Shouldn't use LSTM with projection
2. epochs with a big number is weak for 1Cycle learning rate.
3. The model size is very important. 
4. accumate loss is not working well
<!-- 4. The generator should be list to make sure the results  -->

### optuna tuning
loss 0.03860168904066086 and parameters: {'num_layers': 2, 'hidden_size': 64, 'learning_rate': 0.01} is the best
deeper and narrower, better effect. 
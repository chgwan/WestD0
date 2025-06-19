# D0 signals estimation of West

## ML models
- [ ] Transformer model test
- [ ] ShotWinDS
- [ ] Beta_flat

## Some filters
### Beta_flat development.
shot: 57829, Ip_ref_end > Beta_end
shot: 58182, disruption shot, beta_end large before Ip_ref end
shot: 57384, big fluctuation. Ramp-up only. 

## searc_space results
LSTM: 0.011 - 0.019
LSTM_R1: 0.052
LSTM > MLP
Former > GPT > ERT 

Former 100, 4, 32, 1e-1, loss: 0.024
ERT 100, 4, 32, 1e-3, loss: 0.14 
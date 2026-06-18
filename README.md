# MNIST-Fashion
Full implementation of an MLP trained on MNIST-Fashion. This includes an implementation of various optimisers (SGD, RMSProp, NAG, Adam and some variations). A hyperparameter sweep of 50 configurations was performed using W&B's sweep functionaility. Based on the sweep results, the best model configuration produced a 87% test accuracy.
# To train a particular config:
run the script: python train.py --wandb_project your_project_name --wandb_entity your_wandb_username\
If you want to train a specific configuration, run (example script): python train.py -wp your_project -we your_username -d mnist -e 15 -o sgd -a relu\
supported configurations:
        'epochs':        {'values': [5, 10]},\
        'num_layers':    {'values': [3, 4, 5]},\
        'hidden_size':   {'values': [32, 64, 128]},\
        'weight_decay':  {'values': [0, 0.0005, 0.5]},\
        'learning_rate': {'values': [1e-3, 1e-4]},\
        'optimizer':     {'values': ['sgd','momentum','nesterov','rmsprop','adam','nadam']},\
        'batch_size':    {'values': [16, 32, 64]},\
        'weight_init':   {'values': ['random', 'Xavier']},\
        'activation':    {'values': ['sigmoid', 'tanh', 'relu']},\
        'loss':          {'values': ['cross_entropy', 'mse']}
# Weights and Biases Project Links:
[**Final Project Report**](https://wandb.ai/parthd1901-bits-pilani/fashion-mnist-mlp/reports/CS6910-Assignment-1--VmlldzoxNzIzMzQ1Nw) | 
[**W&B Sweep Runs Dashboard**](https://wandb.ai/parthd1901-bits-pilani/fashion-mnist-mlp/sweeps/oqtudxe7/runs/0j2rx6ht?nw=nwuserparthd1901)

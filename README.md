# MNIST-Fashion
Full implementation of an MLP trained on MNIST-Fashion. This includes an implementation of various optimisers (SGD, RMSProp, NAG, Adam and some variations). A hyperparameter sweep of 50 configurations was performed using W&B's sweep functionaility. Based on the sweep results, the best model configuration produced a 87% test accuracy.
# To train a particular config:
run the script: python train.py --wandb_project your_project_name --wandb_entity your_wandb_username
\if you want to train a specific configuration, run (example script): python train.py -wp your_project -we your_username -d mnist -e 15 -o sgd -a relu
# Weights and Biases Project Links:
[**Final Project Report**](https://wandb.ai/parthd1901-bits-pilani/fashion-mnist-mlp/reports/CS6910-Assignment-1--VmlldzoxNzIzMzQ1Nw) | 
[**W&B Sweep Runs Dashboard**](https://wandb.ai/parthd1901-bits-pilani/fashion-mnist-mlp/sweeps/oqtudxe7/runs/0j2rx6ht?nw=nwuserparthd1901)

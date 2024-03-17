# CS6910_Assignment1
Feedforward Neural Network with Weights & Biases Integration

This Python script trains a feedforward neural network using different hyperparameters and integrates with the Weights & Biases platform for experiment tracking. It provides options to specify various parameters such as dataset, number of epochs, batch size, optimizer, learning rate, activation function, weight initialization method, and more.



Make sure you have the required libraries installed. You can install them using pip:

pip install wandb tensorflow numpy matplotlib seaborn
Open a terminal or command prompt and navigate to the directory containing the Python script.

Run the script with the desired command-line arguments to train the neural network:


python train.py --wandb_project <project_name> --wandb_entity <entity_name> --dataset <dataset_name> --epochs <num_epochs> --batch_size <batch_size> --loss <loss_function> --optimizer <optimizer_name> --learning_rate <learning_rate> --momentum <momentum_value> --beta <beta_value> --beta1 <beta1_value> --beta2 <beta2_value> --epsilon <epsilon_value> --weight_decay <weight_decay_value> --weight_init <weight_init_method> --num_layers <num_hidden_layers> --hidden_size <hidden_layer_size> --activation <activation_function>


Replace <...> with the appropriate values for your experiment. You can refer to the script's documentation for more details on each parameter.

Once the script finishes running, you can view the experiment results on the Weights & Biases dashboard by logging in to your account.

Command-line Arguments

wandb_project: Name of the project to track experiments in the Weights & Biases dashboard.

wandb_entity: Wandb Entity used to track experiments in the Weights & Biases dashboard.

dataset: Dataset to use for training (options: mnist, fashion_mnist).

epochs: Number of epochs to train the neural network.

batch_size: Batch size used for training.

loss: Loss function to use (options: mean_squared_error, cross_entropy).

optimizer: Optimizer to use for training (options: sgd, momentum, nag, rmsprop, adam, nadam).

learning_rate: Learning rate used by the optimizer.

momentum: Momentum value for momentum-based optimizers.

beta: Beta value for optimizers like RMSprop.

beta1: Beta1 value for Adam, Nadam optimizer.

beta2: Beta2 value for Adam , Nadamoptimizer.

epsilon: Epsilon value for optimizers like RMSprop.

weight_decay: Weight decay parameter for regularization.

weight_init: Weight initialization method (options: random, Xavier).

num_layers: Number of hidden layers in the neural network.

hidden_size: Number of neurons in each hidden layer.

activation: Activation function used in the hidden layers (options:sigmoid, tanh, ReLU).

Experiment Tracking
This script integrates with the Weights & Biases platform for experiment tracking. It logs various metrics such as training accuracy, validation accuracy, training loss, validation loss, and more. You can visualize and analyze these metrics on the Weights & Biases dashboard to gain insights into your model's performance and hyperparameter tuning.

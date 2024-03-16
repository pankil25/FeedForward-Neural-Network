import argparse
import numpy as np
import wandb
from keras.datasets import fashion_mnist, mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Nadam

# Define default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    'wandb_project': 'myprojectname',
    'wandb_entity': 'myname',
    'dataset': 'fashion_mnist',
    'epochs': 1,
    'batch_size': 4,
    'loss': 'cross_entropy',
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'momentum': 0.5,
    'beta': 0.5,
    'beta1': 0.5,
    'beta2': 0.5,
    'epsilon': 0.000001,
    'weight_decay': 0.0,
    'weight_init': 'random',
    'num_layers': 1,
    'hidden_size': 4,
    'activation': 'sigmoid'
}

# Function to build a simple feedforward neural network
def build_model(input_shape, num_classes, hidden_size, activation):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for _ in range(DEFAULT_HYPERPARAMETERS['num_layers']):
        model.add(Dense(hidden_size, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Function to train the model
def train_model(x_train, y_train, x_test, y_test, args):
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # Build and compile the model
    model = build_model(x_train.shape[1:], len(np.unique(y_train)), args.hidden_size, args.activation)
    model.compile(loss=args.loss, optimizer=args.optimizer, metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_test, y_test))

    # Log validation accuracy to wandb
    _, accuracy = model.evaluate(x_test, y_test)
    wandb.log({'accuracy': accuracy})

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train neural network with Weights & Biases integration")
    parser.add_argument("-wp", "--wandb_project", default=DEFAULT_HYPERPARAMETERS['wandb_project'], help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default=DEFAULT_HYPERPARAMETERS['wandb_entity'], help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", default=DEFAULT_HYPERPARAMETERS['dataset'], choices=["mnist", "fashion_mnist"], help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_HYPERPARAMETERS['epochs'], help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=DEFAULT_HYPERPARAMETERS['batch_size'], help="Batch size used to train neural network")
    parser.add_argument("-l", "--loss", default=DEFAULT_HYPERPARAMETERS['loss'], choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("-o", "--optimizer", default=DEFAULT_HYPERPARAMETERS['optimizer'], choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=DEFAULT_HYPERPARAMETERS['learning_rate'], help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=DEFAULT_HYPERPARAMETERS['momentum'], help="Momentum")
    parser.add_argument("-beta", "--beta", type=float, default=DEFAULT_HYPERPARAMETERS['beta'], help="Beta")
    parser.add_argument("-beta1", "--beta1", type=float, default=DEFAULT_HYPERPARAMETERS['beta1'], help="Beta1")
    parser.add_argument("-beta2", "--beta2", type=float, default=DEFAULT_HYPERPARAMETERS['beta2'], help="Beta2")
    parser.add_argument("-eps", "--epsilon", type=float, default=DEFAULT_HYPERPARAMETERS['epsilon'], help="Epsilon")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=DEFAULT_HYPERPARAMETERS['weight_decay'], help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", default=DEFAULT_HYPERPARAMETERS['weight_init'], choices=["random", "Xavier"], help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=DEFAULT_HYPERPARAMETERS['num_layers'], help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=DEFAULT_HYPERPARAMETERS['hidden_size'], help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", default=DEFAULT_HYPERPARAMETERS['activation'], choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function")
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Preprocess dataset
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    # Train the model
    train_model(x_train, y_train, x_test, y_test, args)

if __name__ == "__main__":
    main()


import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description="Welcome to the MLP course's Pytorch training and inference helper script"
    )

    parser.add_argument(
        "--continue_from_epoch",
        nargs="?",
        type=int,
        default=-1,
        help="Epoch you want to continue training from while restarting an experiment",
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        default=7112018,
        help="Seed to use for random number generator for experiment",
    )
    parser.add_argument(
        "--num_blocks_per_stage",
        nargs="?",
        type=int,
        default=5,
        help="Number of convolutional blocks in each stage, not including the reduction stage."
        " A convolutional block is made up of two convolutional layers activated using the "
        " leaky-relu non-linearity",
    )
    parser.add_argument(
        "--num_epochs",
        nargs="?",
        type=int,
        default=100,
        help="Total number of epochs for model training",
    )
    parser.add_argument(
        "--experiment_name",
        nargs="?",
        type=str,
        default="exp_1",
        help="Experiment name - to be used for building the experiment folder",
    )
    parser.add_argument(
        "--use_gpu",
        nargs="?",
        type=str2bool,
        default=True,
        help="A flag indicating whether we will use GPU acceleration or not",
    )
    parser.add_argument(
        "--cuda_num",
        nargs="?",
        type=int,
        default=-1,
        help="Cuda device number to use. If -1, use the first one available.",
    )
    parser.add_argument(
        "--weight_decay_coefficient",
        nargs="?",
        type=float,
        default=0,
        help="Weight decay to use for Adam",
    )
    parser.add_argument(
        "--nn_type",
        type=str,
        default="conv_block",
        help="Type of NN to use in the experiment",
    )
    parser.add_argument(
        "--layers", nargs="?", type=int, default=1, help="Number of layers to use."
    )
    parser.add_argument(
        "--hidden_dim",
        nargs="?",
        type=int,
        default=5,
        help="Number of hidden dimensions.",
    )
    parser.add_argument(
        "--learning_rate",
        nargs="?",
        type=float,
        default=0.001,
        help="Learning rate to use for Adam optimizer.",
    )
    parser.add_argument(
        "--dropout",
        nargs="?",
        type=float,
        default=0.0,
        help="dropout probability",
    )
    parser.add_argument(
        "--bias",
        nargs="?",
        type=str2bool,
        default=True,
        help="Whether to include bias",
    )
    args = parser.parse_args()
    print(args)
    return args

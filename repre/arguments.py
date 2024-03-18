import argparse


def parse_args():
    parser = argparse.ArgumentParser("learning psd representation")

    # training
    parser.add_argument("--data_size", type=int, default=100000, help="batch size of data")
    parser.add_argument("--traj_length", type=int, default=200, help="length of trajectory")
    
    parser.add_argument("--num_epoch", type=int, default=200000, help="number of epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="size of minibatch")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for adam optimizer")

    # latent space
    parser.add_argument("--period", type=int, default=5, help="(half) period of skill")
    parser.add_argument("--latent_dim", type=int, default=2, help="latent dimension for skill")

    # evaluation
    parser.add_argument("--num_epoch_for_eval", type=int, default=500, help="period for evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=1000, help="name of the scenario script")
    
    return parser.parse_args()

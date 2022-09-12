import numpy as np

from src.CKE import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=20, help='embedding size')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=50, help='embedding size')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=50, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=40, help='embedding size')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=30, help='embedding size')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.998 	 train_acc: 0.984 	 eval_auc: 0.788 	 eval_acc: 0.724 	 test_auc: 0.796 	 test_acc: 0.727 		[0.17, 0.28, 0.37, 0.42, 0.42, 0.45, 0.45, 0.48]
book	train_auc: 1.000 	 train_acc: 0.997 	 eval_auc: 0.692 	 eval_acc: 0.650 	 test_auc: 0.695 	 test_acc: 0.651 		[0.09, 0.2, 0.25, 0.27, 0.27, 0.32, 0.36, 0.37]
ml	train_auc: 0.938 	 train_acc: 0.857 	 eval_auc: 0.890 	 eval_acc: 0.808 	 test_auc: 0.892 	 test_acc: 0.809 		[0.21, 0.27, 0.46, 0.54, 0.54, 0.6, 0.65, 0.68]
yelp	train_auc: 0.907 	 train_acc: 0.815 	 eval_auc: 0.825 	 eval_acc: 0.758 	 test_auc: 0.825 	 test_acc: 0.760 		[0.09, 0.18, 0.33, 0.33, 0.33, 0.45, 0.46, 0.48]
'''
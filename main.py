import argparse
from train import train
from test import test


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', "--dataset", default='mirflickr', type=str, help='dataset name')
parser.add_argument('-cp', "--checkpoint_path", default='./model/model_mirflickr/lr-0.00075_lr-decay_0.50_lr-times_4.0_nll-0.50_l2-1.00_c-10.00/vae-14014', type=str, help='The path to a checkpoint from which to fine-tune.')

parser.add_argument('-dd', "--data_dir", default='./data/mirflickr/mirflickr_data.npy', type=str, help='The path of input observation data')
parser.add_argument('-train_idx', "--train_idx", default='./data/mirflickr/mirflickr_train_idx.npy', type=str, help='The path of training data index')
parser.add_argument('-valid_idx', "--valid_idx", default='./data/mirflickr/mirflickr_val_idx.npy', type=str, help='The path of validation data index')
parser.add_argument('-test_idx', "--test_idx", default='./data/mirflickr/mirflickr_test_idx.npy', type=str, help='The path of testing data index')

parser.add_argument('-bs', "--batch_size", default=128, type=int, help='the number of data points in one minibatch')
parser.add_argument('-tbs', "--test_batch_size", default=128, type=int, help='the number of data points in one testing or validation batch')
parser.add_argument('-lr', "--learning_rate", default=1e-3, type=float, help='initial learning rate')
parser.add_argument('-epoch', "--max_epoch", default=200, type=int, help='max epoch to train')
parser.add_argument('-wd', "--weight_decay", default=1e-5, type=float, help='weight decay rate')
parser.add_argument('-lrdr', "--lr_decay_ratio", default=0.5, type=float, help='The decay ratio of learning rate')
parser.add_argument('-lrdt', "--lr_decay_times", default=3.0, type=float, help='The number of times learning rate decays')
parser.add_argument('-ntest', "--n_test_sample", default=10000, type=int, help='The sampling times for the testing')
parser.add_argument('-ntrain', "--n_train_sample", default=100, type=int, help='The sampling times for the training')
parser.add_argument('-z', "--z_dim", default=100, type=int, help='z dimention: the number of the independent normal random variables in DMS the rank of the residual covariance matrix')

parser.add_argument('-label_dim', "--label_dim", default=100, type=int, help='the number of labels in current training')
parser.add_argument('-latent_dim', "--latent_dim", default=50, type=int, help='the number of labels in current training')
parser.add_argument('-meta_offset', "--meta_offset", default=0, type=int, help='the offset caused by meta data')
parser.add_argument('-feat_dim', "--feature_dim", default=15, type=int, help='the dimensionality of the features')

parser.add_argument('-se', "--save_epoch", default=1, type=int, help='epochs to save the checkpoint of the model')
parser.add_argument('-max_keep', "--max_keep", default=3, type=int, help='maximum number of saved model')
parser.add_argument('-check_freq', "--check_freq", default=120, type=int, help='checking frequency')

parser.add_argument('-nll_coeff', "--nll_coeff", default=0.1, type=float, help='nll_loss coefficient')
parser.add_argument('-l2_coeff', "--l2_coeff", default=1.0, type=float, help='l2_loss coefficient')
parser.add_argument('-c_coeff', "--c_coeff", default=200., type=float, help='c_loss coefficient')
parser.add_argument('-scale_coeff', "--scale_coeff", default=1.0, type=float, help='mu/logvar scale coefficient')
parser.add_argument('-keep_prob', "--keep_prob", default=0.5, type=float, help='drop out rate')
parser.add_argument('-resume', "--resume", action='store_true', help='whether to resume a ckpt')
parser.add_argument('-write_to_test_sh', "--write_to_test_sh", action='store_true', help='whether to modify test.sh')
parser.add_argument('-mode', "--mode", type=str, help='training/test mode')
parser.add_argument('-r_sigma', "--residue_sigma", default='', type=str, help='what sigma r to use')

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError("mode %s is not supported." % args.mode)

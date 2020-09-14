import numpy as np
import torch
import sys
import datetime
from copy import deepcopy
import evals
from utils import build_path, get_label, get_feat
from model import VAE, compute_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')


def test(args):
    THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]

    METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']
    print('reading npy...')
    data = np.load(args.data_dir)
    test_idx = np.load(args.test_idx)
    print('reading completed')

    print('building network...')
    vae = VAE(args).to(device)
    vae.load_state_dict(torch.load(args.checkpoint_path))
    vae.eval()

    print("loaded model: %s" % (args.checkpoint_path))

    def test_step(test_idx):
        all_nll_loss = 0
        all_l2_loss = 0
        all_c_loss = 0
        all_total_loss = 0

        all_indiv_prob = []
        all_label = []
        all_indiv_max = []

        sigma=[]
        real_batch_size=min(args.batch_size, len(test_idx))
        
        N_test_batch = int( (len(test_idx)-1)/real_batch_size ) + 1

        for i in range(N_test_batch):
            if i % 20 == 0:
                print("%.1f%% completed" % (i*100.0/N_test_batch))

            start = real_batch_size*i
            end = min(real_batch_size*(i+1), len(test_idx))

            input_feat = get_feat(data,test_idx[start:end], args.meta_offset, args.label_dim, args.feature_dim)
            input_label = get_label(data,test_idx[start:end], args.meta_offset, args.label_dim)

            if (all_label == []):
                all_label = input_label
            else:
                all_label = np.concatenate((all_label, input_label))

            input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
            input_label = deepcopy(input_label).float().to(device)

            with torch.no_grad():
                label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = vae(input_label, input_feat) 
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar, vae.r_sqrt_sigma, args)

            all_nll_loss += nll_loss*(end-start)
            #all_l2_loss += l2_loss*(end-start)
            all_c_loss += c_loss*(end-start)
            all_total_loss += total_loss*(end-start)

            indiv_prob = indiv_prob.cpu().data.numpy()

            if (all_indiv_prob == []):
                all_indiv_prob = indiv_prob
            else:
                all_indiv_prob = np.concatenate((all_indiv_prob, indiv_prob))


        nll_loss = all_nll_loss / len(test_idx)
        #l2_loss = all_l2_loss / len(test_idx)
        c_loss = all_c_loss / len(test_idx)
        total_loss = all_total_loss / len(test_idx)
        return all_indiv_prob, all_label

    indiv_prob, input_label = test_step(test_idx)
    n_label = indiv_prob.shape[1]

    best_test_metrics = None
    for threshold in THRESHOLDS:
        test_metrics = evals.compute_metrics(indiv_prob, input_label, threshold, all_metrics=True)
        if best_test_metrics == None:
            best_test_metrics = {}
            for metric in METRICS:
                best_test_metrics[metric] = test_metrics[metric]
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_test_metrics[metric] = min(best_test_metrics[metric], test_metrics[metric])
                else:
                    best_test_metrics[metric] = max(best_test_metrics[metric], test_metrics[metric])
    if 'ebird' in args.dataset:
        ecology = ulti.compute_all(indiv_prob, input_label)
        best_test_metrics = {**best_test_metrics, **ecology}
        METRICS += ["Species_Acc", "Species_Dis", "Species_Cali", "Species_Prec", \
                    "Richness_Acc", "Richness_Dis", "Richness_Cali", "Richness_Prec", \
                    "Community_Acc", "Community_Dis", "Community_Cali", "Community_Prec"]

    print("****************")
    for metric in METRICS:
        print(metric, ":", best_test_metrics[metric])
    print("****************")

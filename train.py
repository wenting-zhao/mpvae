import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os
import datetime
from copy import copy, deepcopy
import evals
from utils import build_path, get_label, get_feat
from model import VAE, compute_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')
THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]

METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']

def train(args):
    print('reading npy...')
    np.random.seed(4) # set the random seed of numpy
    data = np.load(args.data_dir) #load data from the data_dir
    train_idx = np.load(args.train_idx) #load the indices of the training set
    valid_idx = np.load(args.valid_idx) #load the indices of the validation set
    test_idx = np.load(args.test_idx)
    labels = get_label(data, train_idx, args.meta_offset, args.label_dim) #load the labels of the training set

    print("min:", np.amin(labels))
    print("max:", np.amax(labels))

    print("positive label rate:", np.mean(labels)) #print the rate of the positive labels in the training set
    param_setting = "lr-{}_lr-decay_{:.2f}_lr-times_{:.1f}_nll-{:.2f}_l2-{:.2f}_c-{:.2f}".format(args.learning_rate, args.lr_decay_ratio, args.lr_decay_times, args.nll_coeff, args.l2_coeff, args.c_coeff)
    build_path('summary/{}/{}'.format(args.dataset, param_setting))
    build_path('model/model_{}/{}'.format(args.dataset, param_setting))
    summary_dir = 'summary/{}/{}'.format(args.dataset, param_setting)
    model_dir = 'model/model_{}/{}'.format(args.dataset, param_setting)

    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size) # compute the number of iterations in each epoch
    n_iter = one_epoch_iter * args.max_epoch
    print("one_epoch_iter:", one_epoch_iter)
    print("total_iter:", n_iter)

    print("showing the parameters...")
    print(args)

    writer = SummaryWriter(log_dir=summary_dir)

    print('building network...')

    #building the model 
    vae = VAE(args).to(device)
    vae.train()

    #log the learning rate 
    writer.add_scalar('learning_rate', args.learning_rate)

    #use the Adam optimizer 
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

    if args.resume:
        vae.load_state_dict(torch.load(args.checkpoint_path))
        current_step = int(args.checkpoint_path.split('/')[-1].split('-')[-1]) 
        print("loaded model: %s" % args.label_checkpoint_path)
    else:
        current_step = 0

    # smooth means average. Every batch has a mean loss value w.r.t. different losses
    smooth_nll_loss=0.0 # label encoder decoder cross entropy loss
    smooth_nll_loss_x=0.0 # feature encoder decoder cross entropy loss
    smooth_c_loss = 0.0 # label encoder decoder ranking loss
    smooth_c_loss_x=0.0 # feature encoder decoder ranking loss
    smooth_kl_loss = 0.0 # kl divergence
    smooth_total_loss=0.0 # total loss
    smooth_macro_f1 = 0.0 # macro_f1 score
    smooth_micro_f1 = 0.0 # micro_f1 score
    #smooth_l2_loss = 0.0

    best_loss = 1e10
    best_iter = 0
    best_macro_f1 = 0.0 # best macro f1 for ckpt selection in validation
    best_micro_f1 = 0.0 # best micro f1 for ckpt selection in validation
    best_acc = 0.0 # best subset acc for ckpt selction in validation

    temp_label=[]
    temp_indiv_prob=[]

    best_test_metrics = None


    # training the model
    for one_epoch in range(args.max_epoch):
        print('epoch '+str(one_epoch+1)+' starts!')
        np.random.shuffle(train_idx) # random shuffle the training indices

        for i in range(int(len(train_idx)/float(args.batch_size))+1):
            optimizer.zero_grad()
            start = i*args.batch_size
            end = min(args.batch_size*(i+1), len(train_idx))
            input_feat = get_feat(data,train_idx[start:end], args.meta_offset, args.label_dim, args.feature_dim) # get the NLCD features 
            input_label = get_label(data,train_idx[start:end], args.meta_offset, args.label_dim) # get the prediction labels 
            input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
            input_label = deepcopy(input_label).float().to(device)
            label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = vae(input_label, input_feat) 

            #train the model for one step and log the training loss
            if args.residue_sigma == "random":
                r_sqrt_sigma = torch.from_numpy(np.random.uniform(-np.sqrt(6.0/(args.label_dim+args.z_dim)), np.sqrt(6.0/(args.label_dim+args.z_dim)), (args.label_dim, args.z_dim))).to(device)
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar, r_sqrt_sigma, args)
            else:
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar, vae.r_sqrt_sigma, args)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_metrics = evals.compute_metrics(indiv_prob.cpu().data.numpy(), input_label.cpu().data.numpy(), 0.5, all_metrics=False)
            macro_f1, micro_f1 = train_metrics['maF1'], train_metrics['miF1']

            smooth_nll_loss += nll_loss
            smooth_nll_loss_x += nll_loss_x
            #smooth_l2_loss += l2_loss
            smooth_c_loss += c_loss
            smooth_c_loss_x += c_loss_x
            smooth_kl_loss += kl_loss
            smooth_total_loss += total_loss
            smooth_macro_f1 += macro_f1
            smooth_micro_f1 += micro_f1
            
            temp_label.append(input_label) #log the labels
            temp_indiv_prob.append(indiv_prob) #log the individual prediction of the probability on each label

            current_step += 1
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', lr, current_step)

            if current_step % args.check_freq==0: #summarize the current training status and print them out
                nll_loss = smooth_nll_loss / float(args.check_freq)
                nll_loss_x = smooth_nll_loss_x / float(args.check_freq)
                #l2_loss = smooth_l2_loss / float(args.check_freq)
                c_loss = smooth_c_loss / float(args.check_freq)
                c_loss_x = smooth_c_loss_x / float(args.check_freq)
                kl_loss = smooth_kl_loss / float(args.check_freq)
                total_loss = smooth_total_loss / float(args.check_freq)
                macro_f1 = smooth_macro_f1 / float(args.check_freq)
                micro_f1 = smooth_micro_f1 / float(args.check_freq)
                
                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob), (-1))
                temp_label = np.reshape(np.array(temp_label), (-1))
                
                #temp_indiv_prob = np.reshape(temp_indiv_prob,(-1, args.label_dim))
                #temp_label = np.reshape(temp_label,(-1, args.label_dim))

                time_str = datetime.datetime.now().isoformat()
                print("step=%d  %s\nlr=%.6f\nmacro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\ntotal_loss=%.6f\n" % (current_step, time_str, lr, macro_f1, micro_f1, nll_loss*args.nll_coeff, nll_loss_x*args.nll_coeff, c_loss*args.c_coeff, c_loss_x*args.c_coeff, kl_loss, total_loss))
                #print("step=%d  %s\nlr=%.6f\nmacro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\tl2_loss=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\ntotal_loss=%.6f\n" % (current_step, time_str, lr, macro_f1, micro_f1, nll_loss*args.nll_coeff, nll_loss_x*args.nll_coeff, l2_loss*args.l2_coeff, c_loss*args.c_coeff, c_loss_x*args.c_coeff, kl_loss, total_loss))
                temp_indiv_prob=[]
                temp_label=[]

                smooth_nll_loss = 0
                smooth_nll_loss_x = 0
                #smooth_l2_loss = 0
                smooth_c_loss = 0
                smooth_c_loss_x = 0
                smooth_kl_loss = 0
                smooth_total_loss = 0
                smooth_macro_f1 = 0
                smooth_micro_f1 = 0

            if current_step % int(one_epoch_iter*args.save_epoch)==0: #exam the model on validation set
                print("--------------------------------")
                # exam the model on validation set
                current_loss, val_metrics = valid(data, vae, writer, valid_idx, current_step, args)
                macro_f1, micro_f1 = val_metrics['maF1'], val_metrics['miF1']

                # select the best checkpoint based on some metric on the validation set
                # here we use macro F1 as the selection metric but one can use others
                if val_metrics['maF1'] > best_macro_f1:
                    print('macro_f1:%.6f, micro_f1:%.6f, nll_loss:%.6f, which is better than the previous best one!!!'%(macro_f1, micro_f1, current_loss))

                    best_loss = current_loss
                    best_iter = current_step

                    print('saving model')
                    torch.save(vae.state_dict(), model_dir+'/vae-'+str(current_step))
                    print('have saved model to ', model_dir)
                    print()

                    if args.write_to_test_sh:
                        test_sh_path = "script/run_test_%s.sh" % args.dataset
                        if os.path.exists(test_sh_path):
                            ckptFile = open(test_sh_path, "r")
                            command = []
                            for line in ckptFile:
                                arg_lst = line.strip().split(' ')
                                for arg in arg_lst:
                                    if 'model/model_{}/lr-'.format(args.dataset) in arg:
                                        command.append('model/model_{}/{}/vae-{}'.format(args.dataset, param_setting, best_iter))
                                    else:
                                        command.append(arg)
                            ckptFile.close()
                        else:
                            command = ("python main.py --data_dir %s --test_idx %s --label_dim %d --z_dim %d --feature_dim %d --nll_coeff %s --c_coeff %s --batch_size 64 --mode test -cp %s" % (args.data_dir, args.test_idx, args.label_dim, args.z_dim, args.feature_dim, args.nll_coeff, args.c_coeff, 'model/model_{}/{}/vae-{}'.format(args.dataset, param_setting, best_iter))).strip().split(' ')
                        
                        ckptFile = open(test_sh_path, "w")
                        ckptFile.write(" ".join(command)+"\n")
                        ckptFile.close()
                best_macro_f1 = max(best_macro_f1, val_metrics['maF1'])
                best_micro_f1 = max(best_micro_f1, val_metrics['miF1'])
                best_acc = max(best_acc, val_metrics['ACC'])
                
                print("--------------------------------")

def valid(data, vae, summary_writer, valid_idx, current_step, args):
    vae.eval()
    print("performing validation...")

    all_nll_loss = 0
    all_l2_loss = 0
    all_c_loss = 0
    all_total_loss = 0

    all_indiv_prob = []
    all_label = []

    real_batch_size=min(args.batch_size, len(valid_idx))
    for i in range(int((len(valid_idx)-1)/real_batch_size)+1):
        start = real_batch_size*i
        end = min(real_batch_size*(i+1), len(valid_idx))

        input_feat = get_feat(data,valid_idx[start:end], args.meta_offset, args.label_dim, args.feature_dim)
        input_label = get_label(data,valid_idx[start:end], args.meta_offset, args.label_dim)
        input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(input_label)
        input_label = deepcopy(input_label).float().to(device)

        with torch.no_grad():
            label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = vae(input_label, input_feat) 
            total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar, vae.r_sqrt_sigma, args)
    
        all_nll_loss += nll_loss*(end-start)
        #all_l2_loss += l2_loss*(end-start)
        all_c_loss += c_loss*(end-start)
        all_total_loss += total_loss*(end-start)

        for j in deepcopy(indiv_prob).cpu().data.numpy():
            all_indiv_prob.append(j)
        for j in deepcopy(input_label).cpu().data.numpy():
            all_label.append(j)

    # collect all predictions and ground-truths
    all_indiv_prob = np.array(all_indiv_prob)
    all_label = np.array(all_label)

    nll_loss = all_nll_loss/len(valid_idx)
    l2_loss = all_l2_loss/len(valid_idx)
    c_loss = all_c_loss/len(valid_idx)
    total_loss = all_total_loss/len(valid_idx)

    best_val_metrics = None
    for threshold in THRESHOLDS:
        val_metrics = evals.compute_metrics(all_indiv_prob, all_label, threshold, all_metrics=True)

        if best_val_metrics == None:
            best_val_metrics = {}
            for metric in METRICS:
                best_val_metrics[metric] = val_metrics[metric]
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_val_metrics[metric] = min(best_val_metrics[metric], val_metrics[metric])
                else:
                    best_val_metrics[metric] = max(best_val_metrics[metric], val_metrics[metric])

    time_str = datetime.datetime.now().isoformat()
    acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], best_val_metrics['ebF1'], best_val_metrics['maF1'], best_val_metrics['miF1']

    # nll_coeff: BCE coeff, lambda_1
    # c_coeff: Ranking loss coeff, lambda_2
    print("**********************************************")
    print("valid results: %s\nacc=%.6f\tha=%.6f\texam_f1=%.6f, macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tc_loss=%.6f\ttotal_loss=%.6f" % (time_str, acc, ha, ebf1, maf1, mif1, nll_loss*args.nll_coeff, c_loss*args.c_coeff, total_loss))
    print("**********************************************")


    summary_writer.add_scalar('valid/nll_loss', nll_loss, current_step)
    summary_writer.add_scalar('valid/l2_loss', l2_loss, current_step)
    summary_writer.add_scalar('valid/c_loss', c_loss, current_step)
    summary_writer.add_scalar('valid/total_loss',total_loss, current_step)
    summary_writer.add_scalar('valid/macro_f1', maf1, current_step)
    summary_writer.add_scalar('valid/micro_f1', mif1, current_step)
    summary_writer.add_scalar('valid/exam_f1', ebf1, current_step)
    summary_writer.add_scalar('valid/acc', acc, current_step)
    summary_writer.add_scalar('valid/ha', ha, current_step)

    vae.train()

    return nll_loss, best_val_metrics

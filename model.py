import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        # feature layers
        self.fx1 = nn.Linear(args.feature_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)

        self.fd_x1 = nn.Linear(args.feature_dim+args.latent_dim, 256)
        self.fd_x2 = nn.Linear(256, 512)
        self.feat_mp_mu = nn.Linear(512, args.label_dim)

        # label layers
        self.fe1 = nn.Linear(args.feature_dim+args.label_dim, 512)
        self.fe2 = nn.Linear(512, 256) 
        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        self.label_mp_mu = nn.Linear(512, args.label_dim)

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        # things they share
        self.dropout = nn.Dropout(p=args.keep_prob)
        self.scale_coeff = args.scale_coeff
        if args.residue_sigma == 'random':
            r_sqrt_sigma = nn.Parameter(torch.from_numpy(np.random.uniform(-np.sqrt(6.0/(args.label_dim+args.z_dim)), np.sqrt(6.0/(args.label_dim+args.z_dim)), (args.label_dim, args.z_dim))), requires_grad=False)
        elif args.residue_sigma == 'zero':
            r_sqrt_sigma = nn.Parameter(torch.zeros((args.label_dim, args.z_dim)), requires_grad=False)
        else:
            r_sqrt_sigma = nn.Parameter(torch.from_numpy(np.random.uniform(-np.sqrt(6.0/(args.label_dim+args.z_dim)), np.sqrt(6.0/(args.label_dim+args.z_dim)), (args.label_dim, args.z_dim))))
        self.register_parameter("r_sqrt_sigma", r_sqrt_sigma)

    def label_encode(self, x):
        h1 = self.dropout(F.relu(self.fe1(x)))
        h2 = self.dropout(F.relu(self.fe2(h1)))
        mu = self.fe_mu(h2) * self.scale_coeff
        logvar = self.fe_logvar(h2) * self.scale_coeff
        return mu, logvar

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff
        logvar = self.fx_logvar(h3) * self.scale_coeff
        return mu, logvar

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def feat_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def label_decode(self, z):
        h3 = F.relu(self.fd1(z))
        h4 = F.relu(self.fd2(h3))
        return self.label_mp_mu(h4)

    def feat_decode(self, z):
        h4 = F.relu(self.fd_x1(z))
        h5 = F.relu(self.fd_x2(h4))
        return self.feat_mp_mu(h5)

    def label_forward(self, x, feat):
        x = torch.cat((feat, x), 1)
        mu, logvar = self.label_encode(x)
        z = self.label_reparameterize(mu, logvar)
        return self.label_decode(torch.cat((feat, z), 1)), mu, logvar

    def feat_forward(self, x):
        mu, logvar = self.feat_encode(x)
        z = self.feat_reparameterize(mu, logvar)
        return self.feat_decode(torch.cat((x, z), 1)), mu, logvar

    def forward(self, label, feature):
        label_out, label_mu, label_logvar = self.label_forward(label, feature)
        feat_out, feat_mu, feat_logvar = self.feat_forward(feature)
        return label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar

def build_multi_classification_loss(predictions, labels):
    shape = tuple(labels.shape)
    labels = labels.float()
    y_i = torch.eq(labels, torch.ones(shape).to(device))
    y_not_i = torch.eq(labels, torch.zeros(shape).to(device))

    truth_matrix = pairwise_and(y_i, y_not_i).float()
    sub_matrix = pairwise_sub(predictions, predictions)
    exp_matrix = torch.exp(-5*sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, dim=[2,3])
    y_i_sizes = torch.sum(y_i.float(), dim=1)
    y_i_bar_sizes = torch.sum(y_not_i.float(), dim=1)
    normalizers = y_i_sizes * y_i_bar_sizes
    loss = torch.div(sums, 5*normalizers) # 100*128  divide  128
    zero = torch.zeros_like(loss) # 100*128 zeros
    loss = torch.where(torch.logical_or(torch.isinf(loss), torch.isnan(loss)), zero, loss)
    loss = torch.mean(loss)
    return loss

def pairwise_and(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return torch.logical_and(column, row)

def pairwise_sub(a, b):
    column = torch.unsqueeze(a, 3)
    row = torch.unsqueeze(b, 2)
    return column - row

def cross_entropy_loss(logits, labels, n_sample):
    labels = torch.tile(torch.unsqueeze(labels, 0), [n_sample, 1, 1])
    ce_loss = nn.BCEWithLogitsLoss(labels=labels, logits=logits)
    ce_loss = torch.mean(torch.sum(ce_loss, dim=1))
    return ce_loss

def compute_loss(input_label, fe_out, fe_mu, fe_logvar, fx_out, fx_mu, fx_logvar, r_sqrt_sigma, args):
    kl_loss = torch.mean(0.5*torch.sum((fx_logvar-fe_logvar)-1+torch.exp(fe_logvar-fx_logvar)+torch.square(fx_mu-fe_mu)/(torch.exp(fx_logvar)+1e-6), dim=1)) 
    # construct a semi-positive definite matrix
    sigma = torch.mm(r_sqrt_sigma, r_sqrt_sigma.T)

    # covariance = residual_covariance + identity
    covariance = sigma + torch.eye(args.label_dim).to(device)
        
    # epsilon
    eps1=torch.tensor([1e-6]).float().to(device)

    n_sample = args.n_train_sample if args.mode == "train" else args.n_test_sample
    n_batch = fe_out.shape[0]

    # standard Gaussian samples
    noise = torch.normal(0, 1, size=(n_sample, n_batch, args.z_dim)).to(device)
    
    # see equation (3) in the paper for this block
    B = r_sqrt_sigma.T.float().to(device)
    sample_r = torch.tensordot(noise, B, dims=1)+fe_out #tensor: n_sample*n_batch*label_dim
    sample_r_x = torch.tensordot(noise, B, dims=1)+fx_out #tensor: n_sample*n_batch*label_dim
    norm=torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    
    # the probabilities w.r.t. every label in each sample from the batch
    # size: n_sample * n_batch * label_dim
    # eps1: to ensure the probability is non-zero
    E = norm.cdf(sample_r)*(1-eps1)+eps1*0.5
    # similar for the feature branch
    E_x = norm.cdf(sample_r_x)*(1-eps1)+eps1*0.5

    def compute_BCE_and_RL_loss(E):
        #compute negative log likelihood (BCE loss) for each sample point
        sample_nll = -(torch.log(E)*input_label+torch.log(1-E)*(1-input_label))
        logprob=-torch.sum(sample_nll, dim=2)

        #the following computation is designed to avoid the float overflow (log_sum_exp trick)
        maxlogprob=torch.max(logprob, dim=0)[0]
        Eprob=torch.mean(torch.exp(logprob-maxlogprob), axis=0)
        nll_loss=torch.mean(-torch.log(Eprob)-maxlogprob)

        # compute the ranking loss (RL loss) 
        c_loss = build_multi_classification_loss(E, input_label)
        return nll_loss, c_loss

    # BCE and RL losses for label branch
    nll_loss, c_loss = compute_BCE_and_RL_loss(E)
    # BCE and RL losses for feature branch
    nll_loss_x, c_loss_x = compute_BCE_and_RL_loss(E_x)
       
    # if in the training phase, the prediction 
    indiv_prob = torch.mean(E_x, axis=0)

    # total loss: refer to equation (5)
    total_loss = (nll_loss + nll_loss_x) * args.nll_coeff + (c_loss + c_loss_x) * args.c_coeff + kl_loss * 1.1

    return total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob

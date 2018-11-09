#!/usr/bin/env python
# encoding: utf-8

from sinkhorn import _squared_distances

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit

import util
from sinkhorn import sinkhorn_loss_primal
from sinkhorn import sinkhorn_loss_dual

import numpy as np

import base_module
from mmd import mix_rbf_mmd2

torch.utils.backcompat.keepdim_warning.enabled = True


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean(1,keepdim=True)
        return output.view(1)


# Get argument
#print('coucou')
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
#print(parser)
args = parser.parse_args()

#print(args)



if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

def run_gan(args,loss, batch_size, epsilon = 1, niter_sink = 1):

    os.system('mkdir {0}_{1}_eps{2}_niter{3}_batch{4}'.format(args.experiment,loss,epsilon,niter_sink,batch_size))

    args.manual_seed = 1126
    np.random.seed(seed=args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    cudnn.benchmark = True

    # Get data
    trn_dataset = util.get_data(args, train_flag=True)
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    # construct encoder/decoder modules
    hidden_dim = args.nz
    G_decoder = base_module.Decoder(args.image_size, args.nc, k=args.nz, ngf=64)
    D_encoder = base_module.Encoder(args.image_size, args.nc, k=hidden_dim, ndf=64)
    D_decoder = base_module.Decoder(args.image_size, args.nc, k=hidden_dim, ngf=64)

    netG = NetG(G_decoder)
    netD = NetD(D_encoder, D_decoder)
    one_sided = ONE_SIDED()
    #print("netG:", netG)
    #print("netD:", netD)
    #print("oneSide:", one_sided)

    netG.apply(base_module.weights_init)
    netD.apply(base_module.weights_init)
    one_sided.apply(base_module.weights_init)

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]

    # put variable into cuda device
    fixed_noise = torch.cuda.FloatTensor(10**4, args.nz, 1, 1).normal_(0, 1)
    one = torch.cuda.FloatTensor([1])
    mone = one * -1
    if args.cuda:
        netG.cuda()
        netD.cuda()
        one_sided.cuda()
    fixed_noise = Variable(fixed_noise, requires_grad=False)

    # setup optimizer
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)


    time = timeit.default_timer()
    gen_iterations = 0
    for t in range(args.max_iter):
        data_iter = iter(trn_loader)
        i = 0
        while (i < len(trn_loader)):
            # ---------------------------
            #        Optimize over NetD
            # ---------------------------
            for p in netD.parameters():
                p.requires_grad = True

            
            if i == len(trn_loader):
                break

            # clamp parameters of NetD encoder to a cube
            # do not clamp paramters of NetD decoder!!!
            for p in netD.encoder.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = data_iter.next()
            i += 1
            netD.zero_grad()

            x_cpu, _ = data
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            f_enc_X_D, f_dec_X_D = netD(x)

            noise = torch.cuda.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
            noise = Variable(noise, volatile=True)  # total freeze netG
            y = Variable(netG(noise).data)

            f_enc_Y_D, f_dec_Y_D = netD(y)

            if loss == 'sinkhorn_primal':

                sink_D = 2*sinkhorn_loss_primal(f_enc_X_D, f_enc_Y_D, epsilon,batch_size,niter_sink) \
                        - sinkhorn_loss_primal(f_enc_Y_D, f_enc_Y_D, epsilon, batch_size,niter_sink) \
                        - sinkhorn_loss_primal(f_enc_X_D, f_enc_X_D, epsilon, batch_size,niter_sink)
                errD = sink_D 

            elif loss == 'sinkhorn_dual' :

                sink_D = 2*sinkhorn_loss_dual(f_enc_X_D, f_enc_Y_D, epsilon,batch_size,niter_sink) \
                        - sinkhorn_loss_dual(f_enc_Y_D, f_enc_Y_D, epsilon, batch_size,niter_sink) \
                        - sinkhorn_loss_dual(f_enc_X_D, f_enc_X_D, epsilon, batch_size,niter_sink)
                errD = sink_D 

            else:
                mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
                mmd2_D = F.relu(mmd2_D)
                errD = mmd2_D 


            errD.backward(mone)
            optimizerD.step()

            # ---------------------------
            #        Optimize over NetG
            # ---------------------------
            for p in netD.parameters():
                p.requires_grad = False


            netG.zero_grad()

            x_cpu, _ = data
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            f_enc_X, f_dec_X = netD(x)

            noise = torch.cuda.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
            noise = Variable(noise)
            y = netG(noise)

            f_enc_Y, f_dec_Y = netD(y)

          
            ###### Sinkhorn loss #########

            if loss == 'sinkhorn_primal':
            
                sink_G = 2*sinkhorn_loss_primal(f_enc_X, f_enc_Y, epsilon,batch_size,niter_sink) \
                        - sinkhorn_loss_primal(f_enc_Y, f_enc_Y, epsilon, batch_size,niter_sink) \
                        - sinkhorn_loss_primal(f_enc_X, f_enc_X, epsilon, batch_size,niter_sink)
                errG = sink_G 

            elif loss == 'sinkhorn_dual':
            
                sink_G = 2*sinkhorn_loss_dual(f_enc_X, f_enc_Y, epsilon,batch_size,niter_sink) \
                        - sinkhorn_loss_dual(f_enc_Y, f_enc_Y, epsilon, batch_size,niter_sink) \
                        - sinkhorn_loss_dual(f_enc_X, f_enc_X, epsilon, batch_size,niter_sink)
                errG = sink_G 

            else :
                mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
                mmd2_G = F.relu(mmd2_G)
                errG = mmd2_G 

            errG.backward(one)
            optimizerG.step()

            gen_iterations += 1

            if gen_iterations%20 == 1:
            	print('generator iterations ='+str(gen_iterations))

            if gen_iterations%500 == 1:
            	y_fixed = netG(fixed_noise)
            	y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
            	imgfilename = '{0}_{1}_eps{2}_niter{3}_batch{5}/imglist_{4}'.format(args.experiment,loss,epsilon,niter_sink,gen_iterations , batch_size)
            	torch.save(y_fixed.data,imgfilename)
            	print('images saved! generator iterations ='+str(gen_iterations))

            if gen_iterations>10**5:
            	print('done!')
            	break

        if gen_iterations>10**5:
        	print('done!')
        	break
            

######################################################################################################
######################################################################################################
############################################################################################################################################################################################################
######################################################################################################
######################################################################################################


batch_size_list = [100,250,500,1000]
niter_list = [1,10,100]
epsilon_list = [10**4,10**3,10**2,10]

exp_number = 0
print('starting...')
for batch_size in batch_size_list :
    exp_number +=1
    print('exp '+str(exp_number))
    run_gan(args, 'mmd', batch_size)
    for niter_sink in niter_list :
        for epsilon in epsilon_list:
            exp_number +=1
            print('exp '+str(exp_number))
            run_gan(args,'sinkhorn_primal', batch_size, epsilon, niter_sink)
            exp_number +=1
            print('exp '+str(exp_number))
            run_gan(args,'sinkhorn_dual', batch_size, epsilon, niter_sink)
            

import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np

class ANT(nn.Module):
    def __init__(self,base_network):
            super(ANT, self).__init__()
            '''

            base_network (default: None):
                network used to perform anchor training (takes 6 input channels)

            '''
            if base_network is not None:
                self.net = base_network
                if self.net.conv1.weight.shape[1]!=6:
                    raise ValueError('Base Network has incorrect number of input channels (must be 6 for RGB datasets)')
            else:
                raise Exception('base network needs to be defined')


            # define corruption functions for consistency training

            self.txs = transforms.Compose([
                transforms.RandomResizedCrop(size=32,scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.5),
                                        transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5))], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                ])

    def process_batch(self,x,anchors,corrupt,n_anchors):
        '''
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will use a shuffled input minibatch to forward( ) as anchors for that batch (random)

            During *inference* it is recommended to keep the anchors fixed for all samples.

            n_anchors is chosen as min(n_batch,n_anchors)
        '''
        n_img = x.shape[0]
        # if anchors is None:
            # anchors = x[torch.randperm(n_img),:,:,:]

        ## make anchors (n_anchors) --> n_img*n_anchors
        ids = np.random.choice(anchors.shape[0],n_anchors, replace=False)

        A = torch.repeat_interleave(anchors[ids,:,:,:],n_img,dim=0)

        if corrupt:
            refs = self.txs(A)
        else:
            refs = A
        ## before computing residual, make minibatch (n_img) --> n_img* n_anchors
        diff = x.tile((n_anchors,1,1,1)) - A
        batch = torch.cat([refs,diff],axis=1)
        return batch

    def calibrate(self,mu,sig):
        '''
        For ImageNet we use mu_hat = mu/c
        For CIFAR10/100 we use mu_hat = mu/(1+exp(c))
        '''
        c = torch.mean(sig,1)
        c = c.unsqueeze(1).expand(mu.shape)
        return torch.div(mu,1+torch.exp(c))
        # return torch.div(mu,c)

    def forward(self,x,anchors=None,corrupt=False,n_anchors=1,return_std=False):
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        # n_anchors = min(x.shape[0],n_anchors)
        a_batch = self.process_batch(x,anchors=anchors,corrupt=corrupt,n_anchors=n_anchors)

        p = self.net(a_batch)
        p = p.reshape(n_anchors,x.shape[0],p.shape[1])
        mu = p.mean(0)

        if return_std:
            std = p.sigmoid().std(0)
            return self.calibrate(mu,std), std
        else:
            return mu

if __name__=='__main__':
    '''
    EXAMPLE USAGE
    '''
    from models.resnetv2 import resnet20
    base_net = resnet20(nc=6,n_classes=10)
    model = ANT(base_net)

    inputs = torch.randn(64,3,32,32)
    anchors = torch.randn(32,3,32,32)

    ## acts like a vanilla model
    pred = model(inputs)

    ## returns std dev of predictions
    pred1,unc1 = model(inputs,n_anchors=5,return_std=True)

    ## returns std dev of predictions for specified anchors
    pred2,unc2 = model(inputs,anchors=anchors,n_anchors=5,return_std=True)

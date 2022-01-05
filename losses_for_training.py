#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Michael (S. Cai)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

test_sparse_dense = True


class WeightedSoftMarginLoss(nn.Module):
    """
    CVM
    """
    ### the value of margin is given according to the facenet
    def __init__(self, loss_weight=10.0):
        super(WeightedSoftMarginLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, sat_global, grd_global, mini_batch, batch_hard_count=0):
        dist_array = 2 - 2 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diagonal(dist_array)
        
        if(batch_hard_count==0):
            pair_n = mini_batch*(mini_batch - 1.0)
            
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))) / pair_n
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))) / pair_n
            
            loss = (loss_g2s + loss_s2g) / 2.0     
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
            triplet_dist_g2s = triplet_dist_g2s - torch.diag(torch.diagonal(triplet_dist_g2s))
            top_k_g2s, _ = torch.topk((triplet_dist_g2s.t()), batch_hard_count)
            loss_g2s = torch.mean(top_k_g2s)
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
            triplet_dist_s2g = triplet_dist_s2g - torch.diag(torch.diagonal(triplet_dist_s2g))
            top_k_s2g, _ = torch.topk(triplet_dist_s2g, batch_hard_count)
            loss_s2g = torch.mean(top_k_s2g)
            
            loss = (loss_g2s + loss_s2g) / 2.0
            #loss = loss_g2s
            
        pos_dist_avg = pos_dist.mean()
        nega_dist_avg = dist_array.mean()

        return loss, pos_dist_avg, nega_dist_avg.sum()   
    

### OR version
class WeightedSoftMarginLossOR(nn.Module):
    """
    CVM complemented with orientation regression 
    """
    ### the value of margin is given according to the facenet
    def __init__(self, loss_weight=10.0):
        super(WeightedSoftMarginLossOR, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, sat_global, grd_global, mini_batch, batch_hard_count, angle_label, angle_pred):
        dist_array = 2 - 2 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diagonal(dist_array)
        
        if(batch_hard_count==0):
            pair_n = mini_batch*(mini_batch - 1.0)
            
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight)) / pair_n
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            loss_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight)) / pair_n
            
            #loss = (loss_g2s + loss_s2g) / 2.0     
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.loss_weight))
            top_k_g2s, _ = torch.topk(torch.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = torch.mean(top_k_g2s)
            
            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.loss_weight))
            top_k_s2g, _ = torch.topk(triplet_dist_s2g, batch_hard_count)
            loss_s2g = torch.mean(top_k_s2g)
            
            #loss = (loss_g2s + loss_s2g) / 2.0
        
        ### angle regression
        dist_OR = (angle_pred - angle_label).pow(2).sum(1)
        # ground view as anchor
        loss_OR_g = dist_OR.repeat(dist_array.size()[0],1) # the way of repeat and t() according to pos_dist 
        loss_OR_g = loss_OR_g / pair_n
        
        # satellite view as anchor
        loss_OR_s = dist_OR.repeat(dist_array.size()[0],1).t() # the way of repeat and t() according to pos_dist 
        loss_OR_s = loss_OR_s / pair_n
        
        
        #loss_OR = (loss_OR_g + loss_OR_s) / 2.0
        
        # loss combine
        theta1 = 10.0
        theta2 = 5.0
        loss_g2s = theta1*loss_g2s + theta2*loss_OR_g # ground as anchor
        loss_s2g = theta1*loss_s2g + theta2*loss_OR_s # satellite as anchor
        
        loss_merge = torch.sum(loss_g2s + loss_s2g) / 2.0
        
        pos_dist_avg = pos_dist.mean()
        nega_dist_avg = (dist_array - torch.diag(pos_dist)) / pair_n

        return loss_merge, pos_dist_avg, nega_dist_avg.sum()  
    

    
############ HER version 01
class HER_TriLoss_OR_UnNorm(nn.Module):
    """
    HER_TriLoss_OR_UnNorm (Hard Exemplar Reweighting Triplet Loss)
    """
    ### init
    def __init__(self, margin=10.0):
        super(HER_TriLoss_OR_UnNorm, self).__init__()
        #self.l2 = nn.MSELoss()
        #self.cos_dis = nn.CosineSimilarity(dim=, eps=1e-7)
    '''
    def forward(self, sat_global, grd_global, marginCal):
        # scaling fector, sat_global, grd_global, marginCal are 3 vars can be scaled
        # recommended settings: 1.0 for un-normalized features and 5.0 for normalized features
        alpha = 1.0
        sat_global = alpha * sat_global
        grd_global = alpha * grd_global
        
        self.margin = (alpha**2) * marginCal
        
        length_v = grd_global.size()[0]
        
        
        dist_sat = (sat_global.pow(2)).sum(1).reshape(length_v,1).t()
        dist_grd = (grd_global.pow(2)).sum(1).reshape(length_v,1)
        
        distance_negative = dist_grd + dist_sat - 2 * torch.matmul(grd_global, sat_global.t())        
        
        distance_positive = torch.diagonal(distance_negative)
        
        ### distance rectification factor - beta
        beta = self.margin/2.0
        
        ### rectified distance for computing weight mask
        dist_rec = distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative + beta
        
        # dist_clamp = torch.clamp(dist_clamp, max=30.0) # max=30.0, for preventing overflow which leads to inf or nan
    
        p = 1.0/(1.0 + torch.exp( dist_rec ))
        
        ### weight mask generating 
        w_mask = F.relu(-torch.log2(p + 0.00000001))
        
        ### weight mask pruning
        w_low = -np.log2(1.0/(1.0 + np.exp(  -1.0*self.margin + beta ) + 0.00000001) )
        w_high = -np.log2(1.0/(1.0 + np.exp(  -0.0*self.margin + beta ) + 0.00000001) )
        
        w_mask[w_mask<w_low] = 0.1/grd_global.shape[0] # pruning over simple data
        
        w_mask[w_mask>w_high] = w_high # pruning over extreme hard data
        
        
        # diagonal elements need to be neglected (set to zero)
        w_mask = w_mask - torch.diag(torch.diagonal(w_mask))
        
        # main loss computing
        losses = w_mask * torch.log(1.0 + torch.exp( (distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative)))

        return losses.mean(), distance_positive.mean(), distance_negative.mean()
    '''
    def forward(self, sat_global, grd_global, marginCal, angle_label=None, angle_pred=None, sat_global_inv=None, theta1=1, theta2=0.025):
           
        # scaling fector, sat_global, grd_global, marginCal are 3 vars can be scaled
        # recommended settings: 1.0 for un-normalized features and 5.0 for normalized features
        alpha = 5.0
        sat_global = alpha * sat_global
        grd_global = alpha * grd_global
        if sat_global_inv != None:
            sat_global_inv = alpha * sat_global_inv
        
        self.margin = (alpha**2) * marginCal
        
        length_v = grd_global.size()[0]
        
        #distance_negative = torch.autograd.Variable(torch.zeros(length_v,length_v)).cuda()
        #for l in range(length_v):
            #distance_negative[l] = ( grd_global[l].repeat(length_v,1) - sat_global ).pow(2).sum(1)
            
        dist_grd = (grd_global.pow(2)).sum(1).reshape(length_v,1) #[n,1]

        #dist_sat = (sat_global**2).sum(2) # shape = [n,m] 
        dist_sat = (sat_global.pow(2)).sum(1).reshape(1,length_v) #[1,m]
        
        if sat_global_inv != None:
            #dist_sat_inv_mask = torch.einsum('mc,nc->nmc',sat_global_inv,grd_global.ne(0)) # shape = [n,m,c] 
            #dist_sat_inv = (dist_sat_inv_mask**2).sum(2) # shape = [n,m] 
            dist_sat_inv = (sat_global_inv.pow(2)).sum(1).reshape(1,length_v) #[1,m]
        
        distance_negative = dist_grd + dist_sat - 2 * grd_global@sat_global.T #torch.einsum('nc,mnc->nm',grd_global,sat_global)   
        
        
        if sat_global_inv != None:
            distance_negative_inv = dist_grd + dist_sat_inv - 2 * grd_global@sat_global_inv.T #torch.einsum('nc,mnc->nm',grd_global,sat_global_inv)
            inv_select = torch.where(distance_negative<=distance_negative_inv, 1, 0)
            #distance_negative = self.relu(distance_negative-distance_negative_inv)*distance_negative +self.relu(distance_negative_inv-distance_negative)*distance_negative_inv
            #distance_negative = torch.where(distance_negative<=distance_negative_inv, distance_negative, distance_negative_inv)
            distance_negative = inv_select*distance_negative+(1-inv_select)*distance_negative_inv
            
        distance_positive = torch.diagonal(distance_negative)
        
        
        
        ### distance rectification factor - beta
        beta = self.margin/2.0
        
        ### rectified distance for computing weight mask
        dist_rec = distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative + beta
        
        # dist_clamp = torch.clamp(dist_clamp, max=30.0) # max=30.0, for preventing overflow which leads to inf or nan
    
        p = 1.0/(1.0 + torch.exp( dist_rec ))
        
        ### weight mask generating 
        w_mask = F.relu(-torch.log2(p + 0.00000001))
        
        ### weight mask pruning
        w_low = -np.log2(1.0/(1.0 + np.exp(  -1.0*self.margin + beta ) + 0.00000001) )
        w_high = -np.log2(1.0/(1.0 + np.exp(  -0.0*self.margin + beta ) + 0.00000001) )
        
        w_mask[w_mask<w_low] = 0.1/grd_global.shape[0] # pruning over simple data
        
        w_mask[w_mask>w_high] = w_high # pruning over extreme hard data
        
        
        # diagonal elements need to be neglected (set to zero)
        w_mask = w_mask - torch.diag(torch.diagonal(w_mask))
        
        # main loss computing
        losses = w_mask * torch.log(1.0 + torch.exp( (distance_positive.repeat(grd_global.size()[0],1).t() - distance_negative)))
                
        ### orientation regression loss 
        if angle_pred != None:
            if 1:
                losses_OR = (angle_pred-torch.cos(angle_label)).pow(2)
                print("losses_OR: ", losses_OR.detach().cpu().numpy())
                losses_OR = losses_OR.repeat(grd_global.size()[0],1).t()
    
                # OR loss computing
                losses_OR = w_mask * losses_OR 
                
                #loss combining, as a recommendation - theta1 : theta2 = 2 : 1 (here theta2 can be a number in {1,2,3,...,10})
                losses = theta1*losses + theta2*losses_OR 
                loss = losses.mean()
            else:
                losses_OR = (angle_pred-torch.cos(angle_label)).pow(2)
                print("losses_OR: ", losses_OR.detach().cpu().numpy())
                loss = theta1*losses.mean()+theta2*losses_OR.mean()
        else:
            loss = losses.mean()
            
        
        ###### exp based loss

        return loss, distance_positive.mean(), distance_negative.mean()
    

    




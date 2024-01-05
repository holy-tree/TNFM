import os
import math


def adjust_learning_rate(optimizer, lr, epoch, total_epochs):
    
    init_lr = lr
    
    cur_lr = init_lr * 0.5 * (
        1. + math.cos(math.pi * (epoch) / (total_epochs)))

    # cur_lr = init_lr * (1 - epoch / total_epochs)**0.9
    

    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group:
            param_group['lr'] = cur_lr * param_group['lr_scale']
        else:
            param_group['lr'] = cur_lr
    return cur_lr
        
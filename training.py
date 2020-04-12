from core import (PiecewiseLinear, Crop, FlipLR, Cutout, Transform, normalise, pad, transpose, Timer, localtime, TableLogger, StatsLogger,union)
from torch_backend import (batch_norm, Batches, SGD, cifar10, cifar100, Correct,
                           Flatten, Mul, Identity, Add, trainable_params)
from torch import nn
import torch
import apex.amp as amp

transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

def get_cifar10_date_batches(batch_size_train = 512, batch_size_test = 512, transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]):
    CIAFR10_DATA_DIR = './data'
    dataset_10 = cifar10(root = CIAFR10_DATA_DIR)
    train_set_10 = list(zip(transpose(normalise(pad(dataset_10['train']['data'], 4))), dataset_10['train']['labels']))
    test_set_10 = list(zip(transpose(normalise(dataset_10['test']['data'])), dataset_10['test']['labels']))
    train_batches_cifar10 = Batches(Transform(train_set_10, transforms), batch_size_train, shuffle=True, 
                            set_random_choices=True, drop_last=True)
    test_batches_cifar10 = Batches(test_set_10, batch_size_test, shuffle=False, drop_last=False)
    return train_batches_cifar10, test_batches_cifar10


def get_cifar100_date_batches(batch_size = 512, transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]):
    CIAFR100_DATA_DIR = './data'
    dataset_100 = cifar100(root = CIAFR100_DATA_DIR)
    train_set_100 = list(zip(transpose(normalise(pad(dataset_100['train']['data'], 4), mean=[0.4914, 0.4822, 0.4465], std=[0.2675, 0.2565, 0.2761])), dataset_100['train']['labels']))
    test_set_100 = list(zip(transpose(normalise(dataset_100['test']['data'], mean=[0.4914, 0.4822, 0.4465], std=[0.2675, 0.2565, 0.2761])), dataset_100['test']['labels']))
    train_batches_cifar100 = Batches(Transform(train_set_100, transforms), batch_size, shuffle=True, 
                            set_random_choices=True, drop_last=True)
    test_batches_cifar100 = Batches(test_set_100, batch_size, shuffle=False, drop_last=False)
    return train_batches_cifar100, test_batches_cifar100



class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=6, drop_last=False):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.cuda(), 'target': y.cuda().long()} for (x,y) in self.dataloader)   #####half()
    
    def __len__(self): 
        return len(self.dataloader)


def train_epoch(model, train_batches, test_batches, optimizer, timer, test_time_in_total=True):
    train_stats, train_time = run_batches(model, train_batches, True, optimizer), timer()

    test_stats, test_time = run_batches(model, test_batches, False), timer(test_time_in_total)\
    
    return { 
        'train time': train_time, 'train loss': train_stats.mean('loss'), 'train acc': train_stats.mean('correct'), 
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time, 
    }

def test_epoch(model,test_batches,timer,test_time_in_total=True):
    test_stats, test_time = run_batches(model, test_batches, False), timer(test_time_in_total)
    return { 
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time, 
    }
    
def train(model, optimizer, train_batches, test_batches, epochs, 
          loggers=(), test_time_in_total=True, timer=None):  
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(model, train_batches, test_batches, optimizer, timer, test_time_in_total=test_time_in_total) 
        summary = union({'epoch': epoch+1, 'lr': optimizer.param_values()['lr']*train_batches.batch_size}, epoch_stats)
        for logger in loggers:
            logger.append(summary)    
    return summary

def test(model,test_batches,loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    epoch_stats = test_epoch(model, test_batches, timer, test_time_in_total=test_time_in_total) 
    summary = union( epoch_stats)
    for logger in loggers:
        logger.append(summary)    
    return summary

def run_batches(model, batches, training, optimizer=None, stats=None):
    stats = stats or StatsLogger(('loss', 'correct'))

    for batch in batches:
        inp = batch["input"]
        target = batch["target"]
        
        if training:
            model.train()
            output = model(inp)
            output = {"loss":loss(output, target), "correct":acc(output, target)}
            loss_out = output['loss'].sum()
            with amp.scale_loss(loss_out, optimizer._opt) as scaled_loss:
                scaled_loss.backward()
          
            optimizer.step()
            model.zero_grad() 
        
        else:
            model.eval()
            with torch.no_grad():
                output = model(inp)
            output = {"loss":loss(output, target), "correct":acc(output, target)}
        
        stats.append(output) 
    return stats

celoss = nn.CrossEntropyLoss(reduce=False)

def acc(out, target):
    return out.max(dim = 1)[1] == target

def loss(out, target):
    return celoss(out, target)

def get_lr(epochs,train_batches,batch_size):
#     lr_schedule = PiecewiseLinear([0, int(0.208333*epochs), epochs], [0, 0.4, 0])、
#     lr_schedule = PiecewiseLinear([0, epochs/4+1, epochs], [0, 0.4, 0])
#     lr_schedule = PiecewiseLinear([0, epochs/4+1, epochs-4, epochs], [0, 0.4, 0.001,0])
    lr_schedule = PiecewiseLinear([0, epochs/4+1, epochs-3, epochs], [0, 0.4, 0.001,0.0001])
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size
    return lr

def training(model, batches, epochs=24, batch_size=512, opt_level="O1"):
    train_batches,test_batches = batches
        
    lr = get_lr(epochs, train_batches, batch_size)
    opt = SGD(trainable_params(model), lr=lr, momentum=0.9, weight_decay=5e-4*batch_size, nesterov=True)
    model, opt._opt = amp.initialize(model, opt._opt, opt_level=opt_level)
    return train(model, opt, train_batches, test_batches, epochs, loggers=(TableLogger(),))






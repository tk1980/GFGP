#%%
import torch
import torch.nn as nn

from .maxentpool_cuda import maxentpool2d as maxentpool

class MaxEntPoolingCuda2d(nn.AvgPool2d):
    def __init__(self, num_features, kernel_size, stride=None, padding=0, ceil_mode=False,
                 activation='sigmoid', hidden_node=None):
        if hidden_node is None:
            hidden_node = num_features // 2

        toPair = lambda x: x if type(x) is tuple else (x,x)
        super(MaxEntPoolingCuda2d, self).__init__(toPair(kernel_size), stride=toPair(stride), padding=toPair(padding), 
                                                    ceil_mode=ceil_mode, count_include_pad=False)

        self.ToParam = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_features, hidden_node, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(hidden_node),
            nn.ReLU(False),
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
        )
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            raise ValueError("maximum-entropy pooling activation has to be 'sigmoid'/'softplus', "
                         "but got {}".format(activation))
        
    def forward(self, input):
        W = self.activation(self.ToParam(input))

        return maxentpool.maxentpool2d_func.apply(input, W, self.kernel_size, self.stride, self.padding, self.ceil_mode)
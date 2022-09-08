import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class NegativeDictionary(nn.Module):
    """ Negative dictionary for contrastive learning module. The anomalous probability map
    is calculated by the cosine distance between input feature and the dictionary elements.
    If anomalous probability is higher than the threshold, the corresponding feature will
    retain to decoder forward. Else, the feature will be replaced by the element of dictionary.

    Args:
        in_channels (int): Number of input feature map channels.
        num_elements (int): Number of elements for each shape of negative dictionary.
        threshold (float): Threshold for high pass regroup module. If threshold is None,
            the mean value will be selected. Default: 0.625.
        share_dict (bool): If true, all shape of negative dictionary will be the same.
            Default: False.
        groups (int): Number of input groups.
        drop_dict_rate (float): Dictionary dropout rate.
        drop_feat_rate (float): Feature forward to decoder dropout rate.
    """

    def __init__(self,
                 in_channels,
                 num_elements,
                 threshold=0.625,
                 share_dict=True,
                 groups=6,
                 drop_dict_rate=0.,
                 drop_feat_rate=0.):
        super().__init__()
        self.threshold = threshold
        self.share_dict = share_dict
        self.groups = groups
        if share_dict:
            self.negative_dict = nn.Parameter(torch.zeros(num_elements, in_channels))
        else:
            self.negative_dict = nn.ModuleList()
            for i in range(groups):
                self.negative_dict = nn.Parameter(torch.zeros(groups, num_elements, in_channels))
        trunc_normal_(self.negative_dict, std=.02)

        self.drop_dict = nn.Dropout(p=drop_dict_rate)
        self.drop_feat = nn.Dropout(p=drop_feat_rate)

    def generate_anomalous_probability_map(self, feature, dictionary, act_weights=None):
        if act_weights is not None:
            feature = torch.einsum('b c h w, c->b c h w ', feature, act_weights)
            dictionary = torch.einsum('n c, c->n c ', dictionary, act_weights)
        dictionary = self.drop_dict(dictionary)
        feature_norm = feature / feature.norm(dim=1, keepdim=True)
        dictionary_norm = dictionary / dictionary.norm(dim=1, keepdim=True)
        similarity = torch.einsum('b c h w, n c->b n h w ', feature_norm, dictionary_norm)
        apm, index_max = torch.min((1 - similarity) / 2, dim=1, keepdim=True)

        return apm, index_max

    def high_pass(self, feature, dictionary, apm, index):
        B, C, H, W = feature.shape
        if self.threshold is None:
            threshold = torch.mean(apm)
        else:
            threshold = self.threshold
        replace_map = torch.where(apm < threshold, 1, 0)
        index_expand = index.flatten().unsqueeze(-1).expand(-1, C)
        restructured_feature = torch.gather(dictionary, 0, index_expand).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        feature = feature * (1 - replace_map) + restructured_feature * replace_map

        return feature

    def forward(self, x, is_positive=False, act_weights=None):
        if isinstance(x, tuple):
            x = list(x)
            hidden = x[-1]
        else:
            hidden = x

        if self.share_dict and self.groups > 1:
            assert isinstance(hidden, list)
            assert len(hidden) == self.groups

        if isinstance(hidden, list):
            self.apm = []
            for i in range(len(hidden)):
                if act_weights is not None:
                    act_weights_ = act_weights.reshape(self.groups, -1)[i]
                else:
                    act_weights_ = None
                feature_ = hidden[i]
                if self.share_dict:
                    dictionary_ = self.negative_dict
                else:
                    dictionary_ = self.negative_dict[i]
                apm_, index_ = self.generate_anomalous_probability_map(feature_, dictionary_, act_weights_)
                self.apm.append(apm_)
                if is_positive:
                    hidden[i] = self.high_pass(feature_, dictionary_, apm_, index_)
                hidden[i] = self.drop_feat(hidden[i])
            x[-1] = hidden
            x = tuple(x)
        else:
            assert self.share_dict
            if act_weights is not None:
                act_weights_ = act_weights
            else:
                act_weights_ = None
            self.apm, index = self.generate_anomalous_probability_map(hidden, self.negative_dict, act_weights_)
            if is_positive:
                x = self.high_pass(hidden, self.negative_dict, self.apm, index)
        return x


class DictionaryLoss:
    """This criterion computes the similarity loss of anomalous probability maps.

    Args:
        use_entropy (bool): If True, use the entropy loss, else, use L1 loss
    """
    def __init__(self, use_entropy=True):
        self.use_entropy = use_entropy

    def __call__(self, apm, is_positive=False):
        if is_positive:
            apm = 1 - apm
        # apm = apm.max()
        if self.use_entropy:
            return (- (apm * torch.log(1. - apm))).mean()
        else:
            return apm.mean()
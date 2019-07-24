import torch.nn as nn
import torch
import torch.nn.functional as F
class CosineClassifier(nn.Module):
    def __init__(self, with_att=False, novel_only=True):
        super(CosineClassifier, self).__init__()
        self.with_att = with_att
        self.novel_only = novel_only

    @staticmethod
    def compute_similarity(scalar, s, q):
        dot_product = q.matmul(s.t())
        cosine_similarity = dot_product * scalar
        softmax_similarities = F.softmax(cosine_similarity, dim=1)
        return softmax_similarities

    def concatweight(self,  basefeat, basefeat_att, supportfeat, supportfeat_att, queryfeat, queryfeat_att):
        eps = 1e-10
        if self.novel_only and self.with_att:
            output_stack = 0.7*supportfeat + 0.3*supportfeat_att
            sum_support = torch.sum(torch.pow(output_stack, 2), 1)
            s_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            query_features = 0.7*queryfeat + 0.3*queryfeat_att
            return self.compute_similarity(s_magnitude, output_stack, query_features)

        if self.novel_only and self.with_att is False:
            output_stack = supportfeat
            sum_support = torch.sum(torch.pow(output_stack, 2), 1)
            s_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            query_features = queryfeat
            return self.compute_similarity(s_magnitude, output_stack, query_features)

        if self.novel_only is False and self.with_att:
            basetrain = 0.7*basefeat + 0.3*basefeat_att
            output_stack = 0.7*supportfeat + 0.3*supportfeat_att
            query_features = 0.7*queryfeat + 0.3*queryfeat_att
            all_support = torch.cat((basetrain, output_stack),0)
            sum_support = torch.sum(torch.pow(all_support, 2), 1)  #
            s_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            return self.compute_similarity(s_magnitude, all_support, query_features)

        if self.novel_only is False and self.with_att is False:
            all_support = torch.cat((basefeat, supportfeat),0)
            sum_support = torch.sum(torch.pow(all_support, 2), 1)
            s_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            query_features = queryfeat
            return self.compute_similarity(s_magnitude, all_support, query_features)

    def forward(self, basefeat=None, basefeat_att=None,
                supportfeat=None, supportfeat_att=None, queryfeat=None, queryfeat_att=None):

        return self.concatweight(basefeat, basefeat_att,
                                 supportfeat, supportfeat_att, queryfeat, queryfeat_att)


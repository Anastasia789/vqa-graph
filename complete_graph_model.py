#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from layers import NeighbourhoodGraphConvolution as GraphConvolution
from torch.nn.utils.weight_norm import weight_norm



class Model(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 pretrained_wemb,
                 dropout,
                 return_attention = False,
                 n_kernels=8,
                 neighbourhood_size=10):

        '''
        ## Variables:
        - vocab_size: dimensionality of input vocabulary
        - emb_dim: question embedding size
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        - n_kernels : number of Gaussian kernels for convolutions
        - bias: whether to add a bias to Gaussian kernels
        '''

        super(Model, self).__init__()

        # Set parameters
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.neighbourhood_size = neighbourhood_size
        self.return_attention = return_attention

        # initialize word embedding layer weight
        self.wembed = nn.Embedding(vocab_size, emb_dim)
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))

        # question encoding
        self.q_lstm = nn.GRU(input_size=emb_dim, hidden_size=hid_dim)

        # dropout layers
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_q = nn.Dropout(p=dropout/2)

        # visual attention
        self.img_attention = VisualAttention(feat_dim, hid_dim, combined_feat_dim=hid_dim)
        self.graph_attention = VisualAttention(feat_dim + hid_dim, hid_dim, combined_feat_dim=hid_dim)





        # graph convolution layers
        self.graph_convolution_1 = \
            GraphConvolution(feat_dim, hid_dim * 2, n_kernels, 2)
        self.graph_convolution_2 = \
            GraphConvolution(hid_dim * 2, hid_dim, n_kernels, 2)

        # output classifier for graph
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))

        # output classifier for img
        self.img_out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))
        self.img_out_2 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))

    def forward(self, question, image, K, qlen):
        '''
        ## Inputs:
        - question (batch_size, max_qlen): input tokenised question
        - image (batch_size, K, feat_dim): input image features
        - K (int): number of image features/objects in the image
        - qlen (batch_size): vector describing the length (in words) of each input question
        ## Returns:
        - logits (batch_size, out_dim)
        '''

        attentions = []

        K = int(K[0].cpu().data.numpy())

        # Compute question encoding
        emb = self.wembed(question)
        packed = pack_padded_sequence(emb, qlen, batch_first=True)  # questions have variable lengths
        _, hid = self.q_lstm(packed)
        qenc = hid[0].unsqueeze(1)
        qenc_repeat = qenc.repeat(1, K, 1)
        question_embed = hid[0]

        batch_size = image.size(0)

        #select top k objects with highest attention


        attention, img_ques_comb = self.img_attention(image, question_embed)
        attentions.append(attention)

        topk_idx = torch.topk(attention, k=self.neighbourhood_size, dim=1)[1]
        topk_img = torch.gather(image, 1, topk_idx.unsqueeze(-1).expand(-1, -1, image.size(2)))

        # extract bounding boxes and compute centres for top k objects
        bb = topk_img[:, :, -4:].contiguous()
        bb_size = (bb[:, :, 2:] - bb[:, :, :2])
        bb_centre = bb[:, :, :2] + 0.5 * bb_size

        neighbourhood_pseudo = self._compute_pseudo(bb_centre)

        neighbourhood_image = topk_img.unsqueeze(dim=1).expand(batch_size,
                                                               self.neighbourhood_size, self.neighbourhood_size, -1)


        # bb = image[:, :, -4:].contiguous()
        # bb_size = (bb[:, :, 2:]-bb[:, :, :2])
        # bb_centre = bb[:, :, :2] + 0.5*bb_size
        #
        # neighbourhood_pseudo = self._compute_pseudo(bb_centre)
        # neighbourhood_image = image.unsqueeze(dim=1).expand(batch_size, K, K, -1)

        hidden_graph_1 = self.graph_convolution_1(
            neighbourhood_image, neighbourhood_pseudo)
        hidden_graph_1 = F.relu(hidden_graph_1)
        hidden_graph_1 = self.dropout(hidden_graph_1)

        # graph convolution 2
        hidden_graph_1 = hidden_graph_1.unsqueeze(dim=1).expand(hidden_graph_1.size(0),
                                                               self.neighbourhood_size, self.neighbourhood_size, -1)

        # hidden_graph_1 = hidden_graph_1.unsqueeze(dim=1).expand(hidden_graph_1.size(0),
        #                                                        K, K, -1)

        hidden_graph_2 = self.graph_convolution_2(
            hidden_graph_1, neighbourhood_pseudo)
        hidden_graph_2 = F.relu(hidden_graph_2)





        # attention_combined = torch.einsum("bi, bi->bi",
        #                                   torch.gather(attention, 1, topk_idx),
        #                                   graph_att)
        #
        #
        # graph_feat = torch.einsum("bli, bl->bi", hidden_graph_2, attention_combined)

        # attention on graph  nodes
        graph_nodes_feat= torch.cat((topk_img, hidden_graph_2), dim=2)
        graph_att, graph_ques_combine= self.graph_attention(graph_nodes_feat, question_embed)

        attentions.append(
            torch.scatter(input=torch.zeros(batch_size, K), dim=1, index=topk_idx.cpu(), src=graph_att.cpu()))

        graph_feat = torch.einsum("bli, bl->bi", graph_nodes_feat, graph_att)


        # Output classifier graph
        hidden_1 = self.out_1(graph_ques_combine)
        hidden_1 = F.relu(hidden_1)
        hidden_1 = self.dropout(hidden_1)
        logits_1 = self.out_2(hidden_1)

        # Output classifier img
        hidden_2 = self.img_out_1(img_ques_comb)
        hidden_2 = F.relu(hidden_2)
        hidden_2 = self.dropout(hidden_2)
        logits_2 = self.img_out_2(hidden_2)

        logits = logits_1 + logits_2


        if self.return_attention:
            return logits, attentions
        else:
            return logits



    def _compute_pseudo(self, bb_centre):
        '''

        Computes pseudo-coordinates from bounding box centre coordinates

        ## Inputs:
        - bb_centre (batch_size, K, coord_dim)
        - polar (bool: polar or euclidean coordinates)
        ## Returns:
        - pseudo_coord (batch_size, K, K, coord_dim)
        '''

        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)

        return pseudo_coord








class VisualAttention(nn.Module):
    def __init__(self, image_feat_dim, question_embeding_dim,
        combined_feat_dim):
        super(VisualAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.question_embeding_dim = question_embeding_dim
        self.combined_feat_dim = combined_feat_dim

        self.image_proj = weight_norm(nn.Linear(image_feat_dim, combined_feat_dim), dim=None)
        self.txt_proj = weight_norm(nn.Linear(question_embeding_dim, combined_feat_dim), dim=None)

        self.attention_layer = weight_norm(nn.Linear(in_features=combined_feat_dim, out_features=1), dim=None)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, img_features, question_feat):
        """returns attention weights for image features"""
        img_proj = F.relu(self.image_proj(img_features))
        q_proj = F.relu(self.txt_proj(question_feat))


        joint_features = torch.einsum('blk,bk->blk', img_proj, q_proj)
        joint_features = self.dropout(joint_features)

        raw_attention = self.attention_layer(joint_features)
        attention = F.softmax(raw_attention, dim=1)
        attention = attention.squeeze(-1)

        img_attended = torch.einsum("bli, bl->bi", img_features, attention)
        combined_feat = F.relu(self.image_proj(img_attended))*q_proj

        return attention.squeeze(-1), combined_feat






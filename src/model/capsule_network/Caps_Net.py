from src.model.capsule_network.layers.utils import load_word_embedding_matrix
import math
from src.model.capsule_network.layers.elu_layer import Elu_layer
from src.model.capsule_network.layers.conv_layer import ConvLayer
from src.model.capsule_network.layers.caps_conv_layer import ConvCapsLayer
from src.model.capsule_network.layers.routing import Routing, CapsuleNorm
from src.model.capsule_network.Caps_config import Config

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

class ExtractionCapNet(nn.Module):
    def __init__(self, word_embed_dim, capsule_num, filter_ensemble_size, dropout_ratio, intermediate_size, sentence_length):
        super(ExtractionCapNet, self).__init__()

        # self.embedding_layer = nn.Embedding.from_pretrained(embeddings=torch.from_numpy(load_word_embedding_matrix("./embeddings/fasttext_lankadeepa_gossiplanka_300_5")), freeze=True)

        self.elu_layer = Elu_layer(in_channels=1, out_channels=capsule_num,
                                   num_features=word_embed_dim, filter_ensemble_size=filter_ensemble_size)
        self.conv_layer = ConvLayer(in_channels=1, out_channels=capsule_num, num_features=word_embed_dim,
                                    filter_ensemble_size=filter_ensemble_size, dropout_ratio=dropout_ratio)
        self.caps_conv_layer = ConvCapsLayer(
            in_channels=capsule_num, out_channels=intermediate_size[0]*intermediate_size[1], intermediate_size=intermediate_size, dropout_ratio=dropout_ratio, filter_ensemble_size=(int(sentence_length - filter_ensemble_size + 1), 1))  #int(sentence_length - (filter_ensemble_size//2)*2)
        
        self.routing_1 = Routing(num_capsule=16,dim_capsule=16,input_shape=intermediate_size, routing=True,num_routing=3)
        self.routing_2 = Routing(num_capsule=4,dim_capsule=16,input_shape=(16,16), routing=True,num_routing=3)

        self.capsule_norm = CapsuleNorm()
        # self.init_weight()

    # def init_weight(self):
        # torch.nn.init.xavier_normal_(self.routing_1.W)
        # torch.nn.init.xavier_normal_(self.routing_2.W)
        # torch.nn.init.xavier_normal_(self.routing_3.W)

    def forward(self, x):
        # embeddings = self.embedding_layer(x)
        x = torch.unsqueeze(x, 1).float()
        elu_layer = self.elu_layer(x)
        print("Elu layer shape - ",elu_layer.shape)
        conv_layer = self.conv_layer(elu_layer, x)
        print("Conv layer shape - ", conv_layer.shape)
        caps_conv_layer = self.caps_conv_layer(conv_layer)
        print("Caps conv layer shape - ", caps_conv_layer.shape)
        routing_1 = self.routing_1(caps_conv_layer)
        print("routing_1 layer shape - ", routing_1.shape)
        routing_2 = self.routing_2(routing_1)
        print("routing_2 layer shape - ", routing_2.shape)
        capsule_norm = self.capsule_norm(routing_2)
        print("capsule_norm layer shape - ", capsule_norm.shape)
        # print(capsule_norm.size())
        return(capsule_norm)

# config = Config(
#     pretrain_vec=torch.from_numpy(load_word_embedding_matrix("./embeddings/fasttext_lankadeepa_gossiplanka_300_5")))

# model = ExtractionCapNet(word_embed_dim=300, output_size=4, hidden_size=128,
#                       capsule_num=16, filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)


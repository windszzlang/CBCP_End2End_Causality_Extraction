import torch
import torch.nn as nn
from torchcrf import CRF


# ner model with crf
class ArgumentExtractorModel(nn.Module):
    def __init__(self,
                 bert_model,
                 num_labels=5,
                 use_center_index=False,
                 use_center_distance=False,
                 embedding_dim=256,
                 **kwargs):
        # structure: bert
        super(ArgumentExtractorModel, self).__init__()

        self.bert = bert_model
        self.bert_config = self.bert.config

        # get hidden_size
        self.hidden_size = self.bert_config.hidden_size
        # get option
        self.use_center_index = use_center_index
        self.use_center_distance = use_center_distance

        if self.use_center_index:
            self.conditional_layer_norm = ConditionalLayerNorm(self.hidden_size, eps=self.bert_config.layer_norm_eps)

        if self.use_center_distance:
            # structure: center distance embedding
            self.center_distance_embedding = nn.Embedding(num_embeddings=512, embedding_dim=embedding_dim)
            # alter the out dimension of bert
            self.hidden_size += embedding_dim
            # structure: layer normalization
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.bert_config.layer_norm_eps)

        # structure: full connected layer with activation and dropout
        self.linear_layer = nn.Linear(self.hidden_size, num_labels)

        self.crf = CRF(num_labels, batch_first=True)

        # special initialize
        if use_center_distance:
            init_blocks = [self.center_distance_embedding, self.layer_norm]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                center_index=None,
                center_distance=None,
                gold=None):
        # emb.shape=(batch_size, sequence_length, 768)
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        seq_out = bert_out

        if self.use_center_index:
            center_label_feature = self._batch_gather(seq_out, center_index)
            center_label_feature = center_label_feature.view([center_label_feature.size()[0], -1])
            seq_out = self.conditional_layer_norm(seq_out, center_label_feature)

        if self.use_center_distance:
            center_distance_feature = self.center_distance_embedding(center_distance)
            seq_out = torch.cat([seq_out, center_distance_feature], dim=-1)
            seq_out = self.layer_norm(seq_out)
            # seq_out = self.dropout_layer(seq_out)

        seq_out = self.linear_layer(seq_out)
        crf_out = self.crf.decode(seq_out)

        # keep the device of model out consistently!
        out = torch.tensor(crf_out, device=token_type_ids.device).float()[:, 1:-1]

        if gold is not None:
            crf_loss = -1. * self.crf(seq_out, gold.long(), attention_mask.byte())
            loss = crf_loss
            out = out, loss
        return out

    @staticmethod
    def _init_weights(blocks, initializer_range=0.02):
        '''
        make Linear/Embeddig/LayerNorm initialized as Bert
        '''
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=initializer_range)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)

    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        '''
        locate some value in a batch
        :param data: size=(b, sequence_length, h)
        :param index: size=(b, n), n: length of index
        :return: size=(b, n, h)
        '''
        # repeat_interleave(input(self), repeats, dim=None) -> Tensor
        # copy data according to dimension
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # size=(b, n, h)
        # gather(input, dim, index) -> Tensor
        # like aggregate function, use index to locate values at a specific position in the input
        return torch.gather(data, 1, index)

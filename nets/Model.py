import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from nets.graph_encoder import Encoder
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)

class ModelParameters(object):
    def __init__(self, args):
        super(ModelParameters, self).__init__()
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.n_encode_layers = args['n_encode_layers']
        self.dropout = args['dropout']
        self.tanh_clipping = args['tanh_clipping']
        self.mask_inner = args['mask_inner']
        self.mask_logits = args['mask_logits']
        self.n_heads = args['n_heads']
        self.checkpoint_encoder = args['checkpoint_encoder']
        self.alpha = args['alpha']



class AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor  # for compute compatibility []
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor  # for compute final logits  []

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],
                glimpse_val=self.glimpse_val[:, key],
                logit_key=self.logit_key[key]
            )
        return tuple.__getitem__(self, key)


class Model(nn.Module):

    def __init__(self, opts, problem):
        super(Model, self).__init__()
        self.embedding_dim = opts.embedding_dim
        self.hidden_dim = opts.hidden_dim
        self.n_encode_layers = opts.n_encode_layers
        self.temp = 1.0
        self.dropout = opts.dropout

        self.tanh_clipping = opts.tanh_clipping

        self.mask_inner = opts.mask_inner
        self.mask_logits = opts.mask_logits

        self.problem = problem
        self.n_heads = opts.n_heads
        self.checkpoint_encoder = opts.checkpoint_encoder

        step_context_dim = 2 * self.embedding_dim
        node_dim = 2

        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

        self.init_embed = nn.Linear(node_dim, self.embedding_dim)

        self.embeder = Encoder(
            trans_layers=self.n_encode_layers,
            n_heads=self.n_heads,
            d_model=self.embedding_dim,
            d_ffd=self.hidden_dim,
            alpha=opts.alpha,
            dropout=self.dropout
        )

        # compute fixed params (glimpse key, glimpse value, logit key)
        self.project_node_embeddings = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, self.embedding_dim, bias=False)

        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def forward(self, input, return_pi=False):
        '''

        :param input: [bs, gs, node_dim]
        :param return_pi:
        :return:
        '''
        # [bs, gs, d_model]
        if self.checkpoint_encoder and self.training:
            embeddings = checkpoint(self.embeder, self.init_embed(input))
        else:
            embeddings = self.embeder(self.init_embed(input))

        # [bs, num_steps, graph_size], [bs, num_step]
        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)

        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _inner(self, input, embeddings):
        '''

        :param input: [bs, gs, nd]
        :param embeddings: [bs, gs, d_model]
        :return:
        '''
        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        fixed = self._precompute(embeddings)

        i = 0
        while not state.all_finished():
            # log_p.size = [bs, num_steps, graph_size]
            log_p, mask = self._get_log_p(fixed, state)
            # [bs]
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            state = state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # [bs, num_steps, graph_size], [bs, num_step]
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _calc_log_likelihood(self, _log_p, pi, mask):
        '''

        :param _log_p: [bs, num_step, gs]
        :param pi: [bs, num_stpes]
        :param mask:
        :return:
        '''

        # [bs, num_step]
        log_p = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        # [bs]
        return log_p.sum(1)

    def _precompute(self, embeddings, num_steps=1):
        '''

        :param embeddings: [bs, gs, d_model]
        :return:
        '''
        graph_embeded = embeddings.mean(1)
        # [bs, 1, d_model]
        fixed_context = self.project_fixed_context(graph_embeded)[:, None, :]

        # [bs, 1, gs, d_model]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, state, normalize=True):
        '''
        num_steps = graph_size
        :param fixed:
        :param state:
        :return:  [bs, num_steps, gs]
        '''
        # [bs, 1, d_model]
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # [n_heads, bs, 1, gs, head_dim]
        glimpse_K, glimpse_V, logit_K = self._get_attention_ndoe_data(fixed, state)

        mask = state.get_mask()

        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        return log_p, mask

    def _select_node(self, probs, mask):

        if self.decode_type == 'greedy':
            _, selected = probs.max(1)

        elif self.decode_type == 'sampling':
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, 'Unknown decode type'
            
        return selected

    def _make_heads(self, v, num_steps=None):
        '''

        :param v: [bs, 1, gs, d_model]
        :param num_steps:
        :return: [n_heads, bs, n_steps, gs, head_dim] = [n_heads, bs, 1, gs, head_dim]
        '''
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # [n_heads, bs, n_steps, gs, head_dim]
        )

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        '''

        :param embeddings: [bs, gs, d_model]
        :param state:
        :param from_depot:
        :return:
        '''
        # [bs, 1]
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if num_steps == 1:
            if state.i.item() == 0:
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)

        # 这里应该用不到，对tsp num_steps==1
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _get_attention_ndoe_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        '''

        :param query: [bs, 1, d_model]
        :param glimpse_K: [n_heads, bs, 1, gs, head_dim]
        :param glimpse_V:
        :param logit_K: [bs, 1, gs, d_model]
        :param mask:
        :return:[bs, n_steps, gs], [bs, n_steps, d_model]
        '''

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # [n_heads, bs, n_steps, 1, key_size] = [n_heads, bs, 1, 1, k_s]
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # [n_heads, bs, n_steps, 1, gs]
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # [n_heads, bs, n_steps, 1, head_dim]
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # [bs, n_steps, 1, d_model]
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        final_Q = glimpse

        # [bs, n_steps, gs]
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embeder(self.init_embed(input))),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

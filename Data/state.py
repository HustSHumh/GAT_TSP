import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class State(NamedTuple):
    loc: torch.Tensor
    dist: torch.Tensor
    ids:torch.Tensor

    first_a: torch.Tensor       # [bs, 1]
    prev_a: torch.Tensor        # [bs, 1]
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a,
                prev_a=self.prev_a,
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return tuple.__getitem__(self, key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):
        '''

        :param loc: [bs, gs, node_dim]
        :param visited_dtype:
        :return:
        '''
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)

        return State(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            first_a=prev_a,
            prev_a=prev_a,
            visited_=(
                torch.zeros(
                    batch_size, 1, n_loc, dtype=torch.uint8,device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)
        )

    def get_final_cost(self):
        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        # 直接用前两次的选点
        first_a = self.prev_a

        prev_a = selected[:, None]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        # first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)
        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i+1)

    def all_finished(self):
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_ > 0

    def get_mm(self, k=None):
        if k is None:
            k = self.loc.size(-2) - self.i.item()

        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solution(self, actions):
        return actions



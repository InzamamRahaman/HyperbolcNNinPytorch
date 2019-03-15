from torch.optim.optimizer import Optimizer, required


class RSGD(Optimizer):

    def __init__(self, params, lr=required,
                 rgrad=required, exp_map=required):
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'exp_map': exp_map
        }
        super().__init__(params, defaults)

    def step(self, lr=None, counts=None, **kwargs):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                exp_map = group['exp_map']

                if p.grad is not None:
                    d_p = p.grad.data
                    if d_p.is_sparse():
                        d_p = d_p.coalesce()
                    d_p = rgrad(d_p)
                    d_p.mul_(-lr)
                    new_data = exp_map(p.data, d_p)
                    p.data.copy_(new_data)
        return loss




import torch
import torch.nn.functional as F
import math
import torch.nn as nn

# This file contains three network architectures:
# KAN, DenseKAN, KANWithAttention, and KANWithFiLM.

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )


    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)


        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class DenseBlock(nn.Module):
    def __init__(self,in_features,out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],):
        super(DenseBlock,self).__init__()
        self.KAN_fc=KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
        self.Base_Act=base_activation()

    def forward(self,x):
        # print(x.shape)
        x=torch.cat(x,dim=2)
        return self.Base_Act(self.KAN_fc(x))


class DenseKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(DenseKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.KAN_fc1=KANLinear(
                    layers_hidden[0],
                    layers_hidden[1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
        self.db1=DenseBlock(layers_hidden[0]+layers_hidden[1],
                    layers_hidden[2],
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.db2=DenseBlock(layers_hidden[0]+layers_hidden[1]+layers_hidden[2],
                    layers_hidden[3],
                    grid_size=grid_size,
                    spline_order=spline_order,
        )

        self.db3=DenseBlock(layers_hidden[0]+layers_hidden[1]+layers_hidden[2]+layers_hidden[3],
                    layers_hidden[4],
                    grid_size=grid_size,
                    spline_order=spline_order,
        )

        self.KAN_fc2=KANLinear(layers_hidden[0]+layers_hidden[1]+layers_hidden[2]+layers_hidden[3]+layers_hidden[4],
                    layers_hidden[4]*2,
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.drpout=nn.Dropout(0.5)

        self.KAN_fc3=KANLinear(layers_hidden[4]*2,
                    layers_hidden[4],
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.Base_Act=base_activation()

    def forward(self, x: torch.Tensor, update_grid=False):
        x1=self.Base_Act(self.KAN_fc1(x))
        x2=self.db1([x,x1])
        x3=self.db2([x,x1,x2])
        x4=self.db3([x,x1,x2,x3])

        x_cat=torch.cat([x,x1,x2,x3,x4],dim=2)

        x=self.Base_Act(self.KAN_fc2(x_cat))
        x=self.drpout(x)
        x=self.KAN_fc3(x)

        return x


class AttentionLayer(nn.Module):
    def __init__(self,input_dim):
        super(AttentionLayer,self).__init__()
        self.attention=nn.Linear(input_dim,1)

    def forward(self,x):
        attention_weight=F.softmax(self.attention(x),dim=1)
        context_vector=torch.sum(attention_weight*x,dim=1)

        return context_vector

class KANWithAttention(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANWithAttention, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation=base_activation()

        self.KAN_fc1=KANLinear(
                    layers_hidden[0],
                    layers_hidden[1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
        self.attention1=AttentionLayer(layers_hidden[1])

        self.KAN_fc2=KANLinear(layers_hidden[1],
                    layers_hidden[2],
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.attention2 = AttentionLayer(layers_hidden[2])

        self.KAN_fc3 = KANLinear(layers_hidden[2],
                                 layers_hidden[3],
                                 grid_size=grid_size,
                                 spline_order=spline_order, )

        self.attention3 = AttentionLayer(layers_hidden[3])

        self.KAN_fc4 = KANLinear(layers_hidden[3],
                                 layers_hidden[4],
                                 grid_size=grid_size,
                                 spline_order=spline_order, )


    def forward(self, x: torch.Tensor, update_grid=False):
        # print(x.shape)
        x=self.base_activation(self.KAN_fc1(x))
        x=self.attention1(x)
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x=self.base_activation(self.KAN_fc2(x))
        x = self.attention2(x)
        x = x.unsqueeze(dim=1)
        x=self.base_activation(self.KAN_fc3(x))
        x = self.attention3(x)
        x=self.KAN_fc4(x)
        x=x.unsqueeze(dim=1)
        # print(x.shape)

        return x

class FilM(nn.Module):
    def __init__(self,input_dim,modulation_dim,

                 ):
        super(FilM,self).__init__()

        self.gamma=nn.Linear(modulation_dim,input_dim)
        self.beta=nn.Linear(modulation_dim,input_dim)

    def forward(self,x,z):
        gamma=torch.tanh(self.gamma(z))
        beta=torch.tanh(self.beta(z))


        return gamma*x+beta

class KANWithFiLM(nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANWithFiLM, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        self.KAN_fc1=KANLinear(layers_hidden[0],
                                 layers_hidden[1],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film1=FilM(layers_hidden[1],layers_hidden[1])



        self.KAN_fc2=KANLinear(layers_hidden[1],
                                 layers_hidden[2],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film2=FilM(layers_hidden[2],layers_hidden[2])


        self.KAN_fc3=KANLinear(layers_hidden[2],
                                 layers_hidden[3],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film3 = FilM(layers_hidden[3], layers_hidden[3])

        self.KAN_fc4=KANLinear(layers_hidden[3],
                                 layers_hidden[4],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

    def forward(self,x):
        x=self.base_activation(self.KAN_fc1(x))
        x=self.film1(x,x)

        x=self.base_activation(self.KAN_fc2(x))
        x=self.film2(x,x)

        x=self.base_activation(self.KAN_fc3(x))
        x=self.film3(x,x)

        x=self.KAN_fc4(x)

        return x
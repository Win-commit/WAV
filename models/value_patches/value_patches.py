import math
from typing import Any, Dict, Optional, Tuple


import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


class ValueRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 57,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Always compute rope in fp32
        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)

        grid = grid / self.base_seq_length

        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs




@maybe_allow_in_graph
class ValueTransformerBlock(nn.Module):
    r"""
    Modified from Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        attention_class,
        attention_args,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        attn3_cross_attention_dim = 2048,
        num_latent_downsample_block = 0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = attention_class(
            **(attention_args[0]),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = attention_class(
            **(attention_args[1]),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self.num_latent_downsample_block = num_latent_downsample_block
        if self.num_latent_downsample_block > 0:
            self.latent_downsample_block = nn.ModuleList()
            for _i in range(self.num_latent_downsample_block):
                self.latent_downsample_block.append(
                    downsampling_block()
                )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attn3_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        
        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        

        return hidden_states



def add_value_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu",
    norm_eps: float = 1e-6,
    value_in_channels: int = 14,
    value_out_channels: int = None,
    value_num_attention_heads: int = 16,
    value_attention_head_dim: int = 32,
    value_rope_dim: int = None,
    value_final_embeddings: bool = True,
    learnable_value_state: bool = False,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    attention_class = None,
    attention_processor = None,
    **kwargs,
):

    if value_out_channels is None:
        value_out_channels = value_in_channels

    self.value_inner_dim = value_num_attention_heads * value_attention_head_dim

    self.learnable_value_state = learnable_value_state
    if self.learnable_value_state:
        self.value_state = nn.Parameter(torch.randn(1, 1, value_in_channels))

    self.value_proj_in = nn.Linear(value_in_channels, self.value_inner_dim)
    self.value_scale_shift_table = nn.Parameter(torch.randn(2, self.value_inner_dim) / self.value_inner_dim**0.5)
    self.value_time_embed = AdaLayerNormSingle(self.value_inner_dim, use_additional_conditions=False)

    if value_rope_dim is None:
        value_rope_dim = self.value_inner_dim
    # set to a fixed value currently, should adjust according to the action length
    self.value_rope = ValueRotaryPosEmbed(
        dim=value_rope_dim,
        base_seq_length=57,
        theta=10000.0,
    )

    attention_args = []
    attention_args.append(dict(
        query_dim=self.value_inner_dim,
        heads=value_num_attention_heads,
        kv_heads=value_num_attention_heads,
        dim_head=value_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=None,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))
    attention_args.append(dict(
        query_dim=self.value_inner_dim,
        heads=value_num_attention_heads,
        kv_heads=value_num_attention_heads,
        dim_head=value_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=inner_dim,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))

    self.value_blocks = nn.ModuleList(
        [
            ValueTransformerBlock(
                attention_class = attention_class,
                attention_args = attention_args,
                dim=self.value_inner_dim,
                num_attention_heads=value_num_attention_heads,
                attention_head_dim=value_attention_head_dim,
                cross_attention_dim=inner_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
            )
            for _ in range(num_layers)
        ]
    )

    self.value_proj_out = nn.Linear(self.value_inner_dim, value_out_channels) 
    self.value_final_embeddings = value_final_embeddings
    if not self.value_final_embeddings:
        self.value_proj_extra = nn.Linear(self.value_inner_dim, self.value_inner_dim)

    self.value_norm_out = nn.LayerNorm(self.value_inner_dim, eps=1e-6, elementwise_affine=False)


def preprocessing_value_states(
    self,
    value_states: torch.Tensor = None,
    value_timestep: torch.LongTensor = None, #[B,value_seq_length]
):

    assert self.value_expert == True
    assert value_states is not None and value_timestep is not None

    batch_size = value_states.shape[0]

    value_seq_length = value_states.shape[1]        
    if getattr(self, "learnable_value_state") and self.learnable_value_state:
        value_states = self.value_state.repeat(batch_size, value_seq_length, 1).to(dtype=value_states.dtype, device=value_states.device)

    value_rotary_emb = self.value_rope(value_states, value_seq_length)
    value_states = value_states.to(self.value_proj_in.weight.dtype)
    value_hidden_states = self.value_proj_in(value_states)

    value_temb, value_embedded_timestep = self.value_time_embed(
        value_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=value_hidden_states.dtype,
    )

    value_temb = value_temb.view(batch_size, -1, value_temb.size(-1))
    value_embedded_timestep = value_embedded_timestep.view(batch_size, -1, value_embedded_timestep.size(-1))
    
    return value_temb, value_embedded_timestep, value_rotary_emb, value_hidden_states
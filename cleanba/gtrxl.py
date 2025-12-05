# cleanba/gtrxl.py
import dataclasses
import math

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp

from cleanba.network import AtariCNNSpec, CNNEncoderSpec, IdentityNorm, NormConfig, PolicySpec, RMSNorm

# Default Atari-style encoder used unless overridden by Args based on the environment.
DEFAULT_GTRXL_ENCODER = AtariCNNSpec(channels=(16, 32, 32), strides=(2, 2, 2), mlp_hiddens=(256,))


@flax.struct.dataclass
class GTrXLState:
    mems: tuple[jax.Array, ...]  # one memory block per layer, shape (B, mem_len, d_model)


class _IdentityNorm(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


def _norm_from_cfg(cfg: NormConfig) -> nn.Module:
    if isinstance(cfg, RMSNorm):
        return nn.RMSNorm(
            epsilon=cfg.eps, use_scale=cfg.use_scale, reduction_axes=cfg.reduction_axes, feature_axes=cfg.feature_axes
        )
    if isinstance(cfg, IdentityNorm):
        return _IdentityNorm()
    # Fallback: call-through wrapper so we still get separate modules per site
    class _Wrapper(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array) -> jax.Array:
            return cfg(x)

    return _Wrapper()


def _split_heads(x: jax.Array, num_heads: int) -> jax.Array:
    b, t, d = x.shape
    d_head = d // num_heads
    x = x.reshape((b, t, num_heads, d_head))
    return jnp.swapaxes(x, 1, 2)  # (B, H, T, d_head)


def _merge_heads(x: jax.Array) -> jax.Array:
    b, h, t, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape((b, t, h * d))


def _rel_shift(x: jax.Array) -> jax.Array:
    # Standard Transformer-XL relative shift to align BD term
    b, h, q, k = x.shape
    zero_pad = jnp.zeros((b, h, q, 1), dtype=x.dtype)
    x_padded = jnp.concatenate([zero_pad, x], axis=3)
    x_padded = x_padded.reshape((b, h, k + 1, q))
    x = x_padded[:, :, 1:, :]
    return x.reshape((b, h, q, k))


def _relative_positional_encoding(k_len: int, d_model: int, dtype: jnp.dtype) -> jax.Array:
    # Sinusoidal relative positions as in Transformer-XL
    assert d_model % 2 == 0, "d_model must be even for sinusoidal relative positions"
    freq_seq = jnp.arange(0, d_model, 2, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (freq_seq / d_model))
    pos_seq = jnp.arange(k_len - 1, -1, -1, dtype=dtype)  # distances from current token backwards
    sinusoid = jnp.einsum("k,d->kd", pos_seq, inv_freq)
    pos_emb = jnp.concatenate([jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1)
    return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jax.Array, mem: jax.Array, mask: jax.Array, deterministic: bool) -> jax.Array:
        # x: (B, T, D), mem: (B, M, D), mask: (1, 1, T, M+T) with True=keep
        d_head = self.d_model // self.n_heads
        kv = jnp.concatenate([mem, x], axis=1)
        k_len = kv.shape[1]

        wq = nn.Dense(self.d_model, use_bias=False, name="wq")
        wk = nn.Dense(self.d_model, use_bias=False, name="wk")
        wv = nn.Dense(self.d_model, use_bias=False, name="wv")
        wr = nn.Dense(self.d_model, use_bias=False, name="wr")
        wo = nn.Dense(self.d_model, use_bias=False, name="wo")

        r_w_bias = self.param("r_w_bias", nn.initializers.zeros, (self.n_heads, d_head))
        r_r_bias = self.param("r_r_bias", nn.initializers.zeros, (self.n_heads, d_head))

        q = _split_heads(wq(x), self.n_heads)  # (B, H, T, d_head)
        k = _split_heads(wk(kv), self.n_heads)  # (B, H, k_len, d_head)
        v = _split_heads(wv(kv), self.n_heads)  # (B, H, k_len, d_head)

        rel = _relative_positional_encoding(k_len, self.d_model, x.dtype)
        rel = _split_heads(wr(rel[None, :, :]), self.n_heads)  # (1, H, k_len, d_head)
        rel = jnp.squeeze(rel, axis=0)  # (H, k_len, d_head)

        ac = jnp.einsum("bhqd,bhkd->bhqk", q + r_w_bias[:, None, :], k)  # content term
        bd = jnp.einsum("bhqd,hkd->bhqk", q + r_r_bias[:, None, :], rel)  # relative term
        bd = _rel_shift(bd)

        attn = (ac + bd) / math.sqrt(d_head)
        if mask is not None:
            neg_inf = jnp.array(-1e9, dtype=attn.dtype)
            attn = jnp.where(mask, attn, neg_inf)

        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = _merge_heads(out)
        out = wo(out)
        return out

class GRUGate(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        gate = nn.sigmoid(nn.Dense(self.d_model, bias_init=nn.initializers.constant(-1.5))(x) + nn.Dense(self.d_model, use_bias=False)(y))
        return gate * y + (1.0 - gate) * x


class GTrXLLayer(nn.Module):
    d_model: int
    n_heads: int
    mem_len: int
    ff_mult: int
    dropout: float
    norm_cfg: NormConfig

    def setup(self):
        self.attn_norm = _norm_from_cfg(self.norm_cfg)
        self.ff_norm = _norm_from_cfg(self.norm_cfg)
        self.attn = RelativeMultiHeadAttention(self.d_model, self.n_heads, self.dropout)
        self.ff_proj1 = nn.Dense(self.d_model * self.ff_mult)
        self.ff_proj2 = nn.Dense(self.d_model)
        self.ff_dropout = nn.Dropout(rate=self.dropout)
        # Define GRU gates as submodules instead of instantiating them in __call__
        self.attn_gate = GRUGate(self.d_model)
        self.ff_gate = GRUGate(self.d_model)

    def __call__(self, x: jax.Array, mem: jax.Array, mask: jax.Array, deterministic: bool):
        h = self.attn_norm(x)
        attn = self.attn(h, mem, mask, deterministic=deterministic)
        x = self.attn_gate(x, attn)

        h = self.ff_norm(x)
        ff = self.ff_proj1(h)
        ff = nn.gelu(ff)
        ff = self.ff_dropout(ff, deterministic=deterministic)
        ff = self.ff_proj2(ff)
        x = self.ff_gate(x, ff)

        if self.mem_len > 0:
            new_mem = jnp.concatenate([mem, x], axis=1)
            new_mem = new_mem[:, -self.mem_len:, :]
        else:
            new_mem = mem
        return x, new_mem


@dataclasses.dataclass(frozen=True)
class GTrXLConfig(PolicySpec):
    encoder: CNNEncoderSpec = dataclasses.field(
        default_factory=lambda: AtariCNNSpec(channels=(16, 32, 32), strides=(2, 2, 2), mlp_hiddens=(256,))
    )
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    mem_len: int = 256
    ff_mult: int = 4
    dropout: float = 0.0  # keep 0.0 unless you plumb through a deterministic flag from the caller
    mlp_hiddens: tuple[int, ...] = (256,)
    norm: NormConfig = RMSNorm()

    def make(self) -> "GTrXL":
        return GTrXL(self)


class GTrXL(nn.Module):
    cfg: GTrXLConfig

    def setup(self):
        self.encoder = self.cfg.encoder.make()
        self.encoder_norm = _norm_from_cfg(self.cfg.norm)
        self.input_proj = nn.Dense(self.cfg.d_model)
        self.layers = [
            GTrXLLayer(
                self.cfg.d_model,
                self.cfg.n_heads,
                self.cfg.mem_len,
                self.cfg.ff_mult,
                self.cfg.dropout,
                self.cfg.norm,
                name=f"layer_{i}",
            )
            for i in range(self.cfg.n_layers)
        ]
        self.head_norms = [_norm_from_cfg(self.cfg.norm) for _ in self.cfg.mlp_hiddens]
        self.head_mlp = [nn.Dense(h, name=f"mlp_{i}") for i, h in enumerate(self.cfg.mlp_hiddens)]

    def _causal_mask(self, q_len: int, mem_len: int) -> jax.Array:
        k_len = mem_len + q_len
        idx = jnp.arange(k_len)
        causal = idx[None, :] <= (mem_len + jnp.arange(q_len))[:, None]
        return causal[None, None, :, :]  # (1, 1, q, k)

    def _reset_mems(self, mems: tuple[jax.Array, ...], episode_starts: jax.Array) -> tuple[jax.Array, ...]:
        not_reset = (~episode_starts)[:, None, None]
        return tuple(m * not_reset for m in mems)

    def _encode(self, obs: jax.Array) -> jax.Array:
        x = self.encoder(obs)
        x = self.encoder_norm(x)
        return self.input_proj(x)

    def step(
        self, carry: GTrXLState, observations: jax.Array, episode_starts: jax.Array, *, deterministic: bool = True
    ) -> tuple[GTrXLState, jax.Array]:
        x = self._encode(observations).astype(jnp.float32)  # (B, d_model)
        x = x[:, None, :]                                   # add time axis -> (B, 1, d_model)
        mems = tuple(m.astype(jnp.float32) for m in self._reset_mems(carry.mems, episode_starts))
        mem_len = mems[0].shape[1] if mems else 0
        mask = self._causal_mask(x.shape[1], mem_len) if mem_len or x.shape[1] else None

        new_mems: list[jax.Array] = []
        h = x
        for layer, mem in zip(self.layers, mems):
            h, mem = layer(h, mem, mask, deterministic=deterministic)
            new_mems.append(mem)

        h = h.squeeze(1)  # (B, d_model)
        for dense, norm in zip(self.head_mlp, self.head_norms):
            h = norm(h)
            h = nn.relu(dense(h))
        return GTrXLState(mems=tuple(new_mems)), h

    
    def scan(
        self, carry: GTrXLState, observations: jax.Array, episode_starts: jax.Array, *, deterministic: bool = True
    ) -> tuple[GTrXLState, jax.Array]:
        # Detach carried memories at segment boundary (TXL/GTrXL convention)
        carry = GTrXLState(mems=tuple(jax.lax.stop_gradient(m) for m in carry.mems))

        # Step over one time slice: obs_t, eps_t have shape (B, …)
        def _step(module, carry, obs_t, eps_t):
            return module.step(carry, obs_t, eps_t, deterministic=deterministic)

        # Scan along axis 0 (time) for both observation and episode flags
        scan_fn = nn.scan(
            _step,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=(0, 0),     # time axis for both inputs
            out_axes=0,         # return stacked outputs along the time axis
        )
        return scan_fn(self, carry, observations, episode_starts)

    def __call__(self, observations: jax.Array) -> jax.Array:
        zero_state = GTrXLState(
            mems=tuple(
                jnp.zeros((observations.shape[0], self.cfg.mem_len, self.cfg.d_model), observations.dtype)
                for _ in range(self.cfg.n_layers)
            )
        )
        _, h = self.step(zero_state, observations, jnp.zeros((observations.shape[0],), dtype=jnp.bool_))
        return h

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> GTrXLState:
        batch = input_shape[0]
        mems = tuple(
            jnp.zeros((batch, self.cfg.mem_len, self.cfg.d_model), dtype=jnp.float32) for _ in range(self.cfg.n_layers)
        )
        return GTrXLState(mems=mems)

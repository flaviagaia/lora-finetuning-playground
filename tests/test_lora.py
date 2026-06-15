"""Cada lição do LoRA vira um invariante testado."""

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from lora import LoRALinear, train_adapter  # noqa: E402
from tasks import frozen_base_mse, low_rank_task, pretrained_base  # noqa: E402


def test_contagem_de_parametros():
    """trainable = r*(d_in+d_out) e é uma fração minúscula de d_out*d_in."""
    layer = LoRALinear(pretrained_base(4096, 4096), rank=8)
    assert layer.trainable_params == 8 * (4096 + 4096)
    assert layer.base_params == 4096 * 4096
    assert layer.trainable_fraction < 0.005  # < 0.5%


def test_adaptador_inicial_eh_zero():
    """Com B=0, o forward do LoRA é idêntico ao do base puro."""
    W = pretrained_base(64, 64)
    layer = LoRALinear(W, rank=4)
    X = np.random.default_rng(0).normal(size=(64, 16))
    assert np.allclose(layer.forward(X), W @ X)


def test_base_permanece_congelada():
    """Treinar o adaptador NÃO altera os pesos base W."""
    W = pretrained_base(128, 128)
    W_antes = W.copy()
    layer = LoRALinear(W, rank=4)
    X, Y, _ = low_rank_task(W, task_rank=4)
    train_adapter(layer, X, Y, steps=50)
    assert np.array_equal(layer.W, W_antes)


def test_adaptador_aprende_e_vence_base():
    """Base+LoRA reduz drasticamente o erro do base congelado."""
    W = pretrained_base(128, 128)
    X, Y, _ = low_rank_task(W, task_rank=4)
    baseline = frozen_base_mse(W, X, Y)
    layer = LoRALinear(W, rank=4, alpha=8)
    hist = train_adapter(layer, X, Y, steps=400, lr=0.05)
    assert hist[-1] < hist[0]                 # aprendeu
    assert hist[-1] < 0.01 * baseline         # erro < 1% do base congelado


def test_merge_tem_saida_identica():
    """O forward fundido bate com o forward LoRA (custo de inferência zero)."""
    W = pretrained_base(96, 96)
    layer = LoRALinear(W, rank=6, alpha=12)
    X, Y, _ = low_rank_task(W, task_rank=6)
    train_adapter(layer, X, Y, steps=100)
    Xtest = np.random.default_rng(7).normal(size=(96, 24))
    assert np.allclose(layer.forward(Xtest), layer.forward_merged(Xtest), atol=1e-10)


def test_adaptadores_sao_especializados():
    """Um base, dois adaptadores: cada um é melhor na sua própria tarefa."""
    W = pretrained_base(128, 128)
    Xa, Ya, _ = low_rank_task(W, task_rank=4, seed=10)
    Xb, Yb, _ = low_rank_task(W, task_rank=4, seed=20)
    la = LoRALinear(W, rank=4, alpha=8)
    lb = LoRALinear(W, rank=4, alpha=8)
    train_adapter(la, Xa, Ya, steps=400)
    train_adapter(lb, Xb, Yb, steps=400)
    err_a_na_a = np.mean((la.forward(Xa) - Ya) ** 2)
    err_a_na_b = np.mean((la.forward(Xb) - Yb) ** 2)
    assert err_a_na_a < err_a_na_b   # adaptador A é especializado na tarefa A


def test_rank_invalido_falha():
    W = pretrained_base(32, 64)
    try:
        LoRALinear(W, rank=33)  # > min(d_in, d_out)
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass

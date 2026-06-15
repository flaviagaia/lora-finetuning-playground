"""Tarefas sintéticas para o playground.

A história: existe uma matriz "pré-treinada" W_base (congelada). Uma tarefa
nova é a MESMA transformação com um deslocamento de POSTO BAIXO:

    W_alvo = W_base + (B_t @ A_t)     # B_t, A_t de posto r_alvo

Isso é exatamente o cenário em que LoRA brilha: a adaptação necessária mora
num subespaço de baixa dimensão, então um adaptador de posto r consegue
recuperá-la — enquanto o modelo base congelado, sozinho, erra.

Tudo é determinístico (seed fixa) para os testes serem estáveis.
"""

from __future__ import annotations

import numpy as np


def pretrained_base(d_out: int, d_in: int, seed: int = 1) -> np.ndarray:
    """Matriz de pesos 'pré-treinada' e congelada."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_out, d_in))


def low_rank_task(
    W_base: np.ndarray,
    task_rank: int,
    n_samples: int = 256,
    seed: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera (X, Y, W_alvo) para uma tarefa = base + deslocamento posto-baixo.

    Retorna entradas X (d_in, N), alvos Y (d_out, N) e a matriz alvo verdadeira.
    """
    d_out, d_in = W_base.shape
    rng = np.random.default_rng(seed)
    B_t = rng.normal(0.0, 1.0, size=(d_out, task_rank))
    A_t = rng.normal(0.0, 1.0, size=(task_rank, d_in))
    shift = (B_t @ A_t) / np.sqrt(d_in)
    W_target = W_base + shift
    X = rng.normal(0.0, 1.0, size=(d_in, n_samples))
    Y = W_target @ X
    return X, Y, W_target


def frozen_base_mse(W_base: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    """Erro do modelo base congelado SEM adaptação — a linha de base a vencer."""
    pred = W_base @ X
    return float(np.mean((pred - Y) ** 2))

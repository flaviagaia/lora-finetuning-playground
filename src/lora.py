"""LoRA do zero, em numpy puro — sem PyTorch, sem GPU, sem API key.

LoRA (Low-Rank Adaptation) congela a matriz de pesos pré-treinada W e
aprende um ajuste de POSTO BAIXO:

    W_efetivo = W + (alpha / r) * B @ A

onde A tem shape (r, d_in) e B tem shape (d_out, r), com r << min(d_in, d_out).
Em vez de treinar os d_out * d_in pesos de W, treinamos só
r * (d_in + d_out) pesos — uma fração minúscula.

Convenção de inicialização (a do paper): A ~ aleatório pequeno, B = 0.
Assim, no início do treino, B @ A = 0 e o modelo se comporta EXATAMENTE
como o modelo base — o adaptador só adiciona o que aprender.

Tudo aqui é diferenciável na mão (gradientes fechados para perda MSE),
o que torna o experimento 100% transparente e reprodutível.
"""

from __future__ import annotations

import numpy as np


class LoRALinear:
    """Camada linear y = W x com adaptador LoRA. W fica CONGELADA."""

    def __init__(
        self,
        weight: np.ndarray,  # (d_out, d_in) — pesos pré-treinados, congelados
        rank: int,
        alpha: float | None = None,
        seed: int = 0,
    ) -> None:
        self.W = weight.astype(np.float64)
        self.d_out, self.d_in = weight.shape
        if not 0 < rank <= min(self.d_in, self.d_out):
            raise ValueError("rank deve estar em (0, min(d_in, d_out)]")
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank

        rng = np.random.default_rng(seed)
        # A pequena e aleatória, B zerada -> adaptador inicial = 0
        self.A = rng.normal(0.0, 1.0 / self.d_in, size=(rank, self.d_in))
        self.B = np.zeros((self.d_out, rank))

    # ---- contagem de parâmetros (a tese de eficiência) ----
    @property
    def base_params(self) -> int:
        return self.d_out * self.d_in

    @property
    def trainable_params(self) -> int:
        return self.rank * (self.d_in + self.d_out)

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / self.base_params

    # ---- forward ----
    def delta(self) -> np.ndarray:
        """O ajuste de posto baixo, já com o scaling aplicado."""
        return self.scaling * (self.B @ self.A)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (d_in, N) -> y: (d_out, N). Usa W congelada + adaptador."""
        return self.W @ x + self.delta() @ x

    # ---- backward (apenas A e B; W congelada) ----
    def grads_mse(self, x: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Gradientes de MSE em relação a A e B. Retorna (gA, gB, loss)."""
        n = x.shape[1]
        pred = self.forward(x)
        diff = pred - y_true
        loss = float(np.mean(diff ** 2))
        dL_dpred = (2.0 / n) * diff                     # (d_out, N)
        dL_dM = self.scaling * (dL_dpred @ x.T)         # (d_out, d_in)
        gB = dL_dM @ self.A.T                           # (d_out, r)
        gA = self.B.T @ dL_dM                           # (r, d_in)
        return gA, gB, loss

    # ---- merge: dobra o adaptador de volta em W (custo de inferência zero) ----
    def merged_weight(self) -> np.ndarray:
        """W + adaptador. Em produção você implanta SÓ esta matriz."""
        return self.W + self.delta()

    def forward_merged(self, x: np.ndarray) -> np.ndarray:
        return self.merged_weight() @ x


class Adam:
    """Otimizador Adam mínimo para os tensores A e B."""

    def __init__(self, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self._state: dict[int, tuple] = {}
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> None:
        self.t += 1
        key = id(param)
        if key not in self._state:
            self._state[key] = (np.zeros_like(param), np.zeros_like(param))
        m, v = self._state[key]
        m[:] = self.b1 * m + (1 - self.b1) * grad
        v[:] = self.b2 * v + (1 - self.b2) * (grad ** 2)
        m_hat = m / (1 - self.b1 ** self.t)
        v_hat = v / (1 - self.b2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_adapter(
    layer: LoRALinear,
    x: np.ndarray,
    y: np.ndarray,
    steps: int = 400,
    lr: float = 0.05,
) -> list[float]:
    """Treina SÓ o adaptador (A, B) por gradiente. Retorna a curva de perda."""
    opt = Adam(lr=lr)
    history = []
    for _ in range(steps):
        gA, gB, loss = layer.grads_mse(x, y)
        opt.step(layer.A, gA)
        opt.step(layer.B, gB)
        history.append(loss)
    return history

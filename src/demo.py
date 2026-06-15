"""Demo executável do LoRA playground — roda em ~1 segundo, sem dependências
além de numpy.

Mostra as quatro lições do post B26:
  1. eficiência de parâmetros (LoRA treina uma fração minúscula)
  2. o adaptador APRENDE (vence o base congelado)
  3. merge tem custo de inferência ZERO (saída idêntica)
  4. um base, vários adaptadores (troca de tarefa sem tocar no base)

Uso:
    python src/demo.py
"""

from __future__ import annotations

from lora import LoRALinear, train_adapter
from tasks import frozen_base_mse, low_rank_task, pretrained_base


def tabela_eficiencia() -> None:
    print("\n1) EFICIÊNCIA DE PARÂMETROS")
    print(f"   {'camada (d_out x d_in)':>22} | {'rank':>4} | {'base':>12} | {'LoRA':>9} | {'%':>7}")
    casos = [
        (4096, 4096, 8),    # peso de atenção típico de um LLM
        (4096, 4096, 16),
        (4096, 11008, 16),  # projeção MLP (ex.: LLaMA)
        (256, 256, 8),      # a escala deste demo
    ]
    for d_out, d_in, r in casos:
        layer = LoRALinear(pretrained_base(d_out, d_in), rank=r)
        print(
            f"   {f'{d_out} x {d_in}':>22} | {r:>4} | {layer.base_params:>12,} | "
            f"{layer.trainable_params:>9,} | {100*layer.trainable_fraction:>6.2f}%"
        )


def treina_e_avalia() -> LoRALinear:
    print("\n2) O ADAPTADOR APRENDE (base congelado vs base + LoRA)")
    d_out, d_in = 256, 256
    W_base = pretrained_base(d_out, d_in)
    X, Y, _ = low_rank_task(W_base, task_rank=4)

    baseline = frozen_base_mse(W_base, X, Y)
    layer = LoRALinear(W_base, rank=4, alpha=8)
    hist = train_adapter(layer, X, Y, steps=500, lr=0.05)

    print(f"   MSE base congelado (sem adaptação): {baseline:.4f}")
    print(f"   MSE inicial do LoRA (B=0, = base) : {hist[0]:.4f}")
    print(f"   MSE final do LoRA                 : {hist[-1]:.6f}")
    print(f"   Redução sobre o base congelado    : {100*(1 - hist[-1]/baseline):.2f}%")
    print(f"   Parâmetros treinados: {layer.trainable_params:,} de "
          f"{layer.base_params:,} ({100*layer.trainable_fraction:.2f}%)")
    return layer


def verifica_merge(layer: LoRALinear) -> None:
    print("\n3) MERGE = CUSTO DE INFERÊNCIA ZERO")
    import numpy as np
    X = np.random.default_rng(99).normal(size=(layer.d_in, 32))
    y_lora = layer.forward(X)          # W congelada + B@A em duas operações
    y_merged = layer.forward_merged(X)  # uma única matriz já fundida
    max_diff = float(np.max(np.abs(y_lora - y_merged)))
    print(f"   Maior diferença entre forward LoRA e forward fundido: {max_diff:.2e}")
    print("   -> idêntico: implante uma só matriz, sem overhead de runtime.")


def troca_de_adaptador() -> None:
    print("\n4) UM BASE, VÁRIOS ADAPTADORES")
    d_out, d_in = 256, 256
    W_base = pretrained_base(d_out, d_in)
    Xa, Ya, _ = low_rank_task(W_base, task_rank=4, seed=10)
    Xb, Yb, _ = low_rank_task(W_base, task_rank=4, seed=20)

    layer_a = LoRALinear(W_base, rank=4, alpha=8, seed=0)
    layer_b = LoRALinear(W_base, rank=4, alpha=8, seed=0)
    train_adapter(layer_a, Xa, Ya, steps=500)
    train_adapter(layer_b, Xb, Yb, steps=500)

    import numpy as np
    # o adaptador A é ótimo na tarefa A, ruim na B, e vice-versa
    aa = float(np.mean((layer_a.forward(Xa) - Ya) ** 2))
    ab = float(np.mean((layer_a.forward(Xb) - Yb) ** 2))
    print(f"   Adaptador A na tarefa A: {aa:.4f}  | na tarefa B: {ab:.4f}")
    print("   Mesma matriz base de 65.536 pesos; só trocamos 2.048 pesos de adaptador.")


def main() -> None:
    print("=" * 64)
    print("LoRA PLAYGROUND — adaptação de posto baixo do zero (numpy)")
    print("=" * 64)
    tabela_eficiencia()
    layer = treina_e_avalia()
    verifica_merge(layer)
    troca_de_adaptador()


if __name__ == "__main__":
    main()

# LoRA do Zero — Low-Rank Adaptation em numpy puro

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · só numpy · 100% offline, sem PyTorch, sem GPU, sem API key · MIT License

---

## 🇧🇷 Português

### A tese

Fine-tuning completo de um LLM treina **todos** os pesos — caro, pesado e difícil
de versionar. LoRA (Low-Rank Adaptation) congela a matriz pré-treinada `W` e
aprende um ajuste de **posto baixo**:

```
W_efetivo = W + (alpha / r) · B · A
```

com `A` de shape `(r, d_in)` e `B` de shape `(d_out, r)`, `r ≪ min(d_in, d_out)`.
Em vez de `d_out · d_in` pesos, treinamos `r · (d_in + d_out)` — uma fração minúscula.

Este repositório implementa LoRA **do zero, em numpy**, com gradientes derivados na
mão, para que cada peça seja transparente. Roda em ~1 segundo.

### As quatro lições (todas com números reais deste repo)

**1. Eficiência de parâmetros.** Numa matriz de atenção típica de LLM (4096×4096),
LoRA rank 8 treina **0,39%** dos pesos:

| camada (d_out × d_in) | rank | pesos base | pesos LoRA | % treinado |
| --------------------- | ---- | ---------- | ---------- | ---------- |
| 4096 × 4096 | 8 | 16.777.216 | 65.536 | 0,39% |
| 4096 × 4096 | 16 | 16.777.216 | 131.072 | 0,78% |
| 4096 × 11008 (MLP) | 16 | 45.088.768 | 241.664 | 0,54% |

**2. O adaptador aprende.** Numa tarefa cujo alvo é o base + um deslocamento de
posto baixo, o base congelado erra (MSE 4,23) e o base + LoRA (treinando **3,12%**
dos pesos) chega a MSE **0,00003** — redução de ~100%.

**3. Merge = custo de inferência zero.** Como `W + (alpha/r)·B·A` é uma soma de
matrizes, o adaptador pode ser **fundido** de volta em `W` antes do deploy. A saída
do forward fundido bate com a do forward LoRA até `1e-14` — zero overhead em runtime.

**4. Um base, vários adaptadores.** A mesma matriz base de 65.536 pesos serve duas
tarefas; só trocamos os **2.048** pesos do adaptador. O adaptador da tarefa A acerta
a tarefa A (MSE 0,00) e erra a B (MSE 8,00) — especialização real, base intocado.

### Por que numpy e não PyTorch?

A mecânica do LoRA é uma soma de matrizes e dois gradientes fechados — não precisa
de framework. Implementar à mão deixa claro **onde** os parâmetros vivem, **por que**
`B` começa zerada (para o adaptador inicial ser nulo) e **como** o merge funciona.
Para escala real, a mesma lógica vira três linhas com a biblioteca `peft` da Hugging
Face sobre um modelo Transformer — o conceito é idêntico.

### Execução

```
pip install -r requirements.txt
pytest tests/ -v        # 7 testes
python src/demo.py      # roda as 4 lições com números reais (~1s)
```

### Estrutura

```
src/
├── lora.py    # LoRALinear (forward, backward, merge) + Adam, do zero
├── tasks.py   # tarefas sintéticas posto-baixo (teacher-student)
└── demo.py    # roda e imprime as 4 lições
tests/
└── test_lora.py   # um invariante por lição
```

### Limitações honestas

Demo de regressão linear com adaptação exatamente de posto baixo — o cenário ideal
para LoRA brilhar. Em modelos reais a adaptação não é perfeitamente de posto baixo,
então o rank vira hiperparâmetro (e dialoga com o `optuna-rag-tuning`). Não há
não-linearidades nem atenção aqui: o objetivo é isolar a **mecânica** do LoRA, não
reproduzir um Transformer.

---

## 🇺🇸 English

### The thesis

Full fine-tuning trains **every** weight — expensive and hard to version. LoRA
freezes the pretrained matrix `W` and learns a **low-rank** update
`W_eff = W + (alpha/r)·B·A`, with `r ≪ min(d_in, d_out)`. Instead of `d_out·d_in`
weights we train `r·(d_in + d_out)` — a tiny fraction. This repo implements LoRA
**from scratch in numpy**, with hand-derived gradients, running in ~1 second.

### Four lessons (with real numbers from this repo)

1. **Parameter efficiency** — on a 4096×4096 attention matrix, LoRA rank 8 trains
   **0.39%** of the weights (65,536 of 16,777,216).
2. **The adapter learns** — frozen base MSE 4.23 → base+LoRA MSE 0.00003, training
   only **3.12%** of the weights.
3. **Merge = zero inference cost** — the fused forward matches the LoRA forward to
   `1e-14`; deploy a single matrix with no runtime overhead.
4. **One base, many adapters** — same 65,536-weight base serves two tasks; we swap
   only the 2,048 adapter weights. Task-A adapter nails task A (0.00) and fails task
   B (8.00) — real specialization, base untouched.

### Why numpy instead of PyTorch?

LoRA's mechanics are a matrix sum and two closed-form gradients — no framework
needed. Doing it by hand shows where the parameters live, why `B` starts at zero,
and how merging works. At real scale this becomes three lines with Hugging Face
`peft` over a Transformer — the concept is identical.

### Running

```
pip install -r requirements.txt
pytest tests/ -v        # 7 tests
python src/demo.py      # runs the 4 lessons with real numbers (~1s)
```

### Honest limitations

Linear-regression demo with an exactly low-rank target — the ideal case for LoRA.
Real models aren't perfectly low-rank, so rank becomes a hyperparameter (which ties
back to `optuna-rag-tuning`). No nonlinearities or attention here: the goal is to
isolate LoRA's **mechanics**, not to reproduce a Transformer.

---

Part of my LinkedIn series on efficient LLMs → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)

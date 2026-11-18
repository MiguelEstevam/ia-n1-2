# Perceptron com sigmoid: detectando fraude em cartão

Programa didático em Python que treina um único neurônio (perceptron) com função de ativação sigmoid para classificar compras de cartão como **fraudulentas (1)** ou **legítimas (0)**. Ele foi feito para acompanhar um vídeo explicando como uma rede neural atualiza seus pesos e reduz o erro.

## Requisitos

- Python 3.8 ou superior
- Nenhuma biblioteca externa (usa apenas `math`)

## Como executar

```bash
python fraude_perceptron.py
```

O programa faz três coisas, nesta ordem:

1. **Passo a passo da primeira compra:** mostra `z`, a saída da sigmoid, o erro, o gradiente de cada peso e os novos pesos.
2. **Treinamento:** roda 2000 épocas e imprime o erro total em algumas delas.
3. **Resultado final:** mostra a probabilidade de fraude para cada compra, os pesos finais e a previsão para uma compra nova.

## Entradas e saída

Cada compra tem três características, todas normalizadas entre 0 e 1, mais o viés fixo `x0 = 1`:

| Entrada | Significado |
| --- | --- |
| `x1` | Valor da compra em relação ao gasto habitual do cliente |
| `x2` | Distância do local habitual de compra |
| `x3` | Horário incomum (0 = normal, 1 = madrugada) |

A saída é a probabilidade de fraude. Valores a partir de 0,5 são classificados como fraude.

## Como funciona

**1. Soma ponderada**

```
z = w0·1 + w1·x1 + w2·x2 + w3·x3
```

**2. Ativação (sigmoid)**

```
ŷ = 1 / (1 + e^(-z))
```

**3. Erro quadrático**

```
E = ½ · (y − ŷ)²
```

**4. Atualização dos pesos (backpropagation)**

```
δ = (ŷ − y) · ŷ · (1 − ŷ)
w ← w − η · δ · x
```

O sinal negativo move cada peso na direção que diminui o erro. O termo `δ` é comum a todos os pesos, e cada peso é ajustado em proporção à sua entrada `x`.

## Parâmetros

Ficam no topo do arquivo e podem ser alterados:

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| `PESOS_INICIAIS` | `[-0.5, 0.3, 0.2, -0.1]` | w0 (viés), w1, w2, w3 |
| `ETA` | `0.5` | Taxa de aprendizado |
| `EPOCAS` | `2000` | Passadas completas pelo conjunto de treino |
| `DADOS` | 10 compras | Lista de `([1, x1, x2, x3], y)` |

## Exemplo de resultado

Primeira compra (`x = 0,9 / 0,7 / 0,2`, fraude), antes e depois de um único passo de atualização:

| | Antes | Depois |
| --- | --- | --- |
| Saída ŷ | 0,4725 | 0,5110 |
| Erro E | 0,1391 | 0,1196 |

Erro total ao longo do treinamento:

| Época | Erro total |
| --- | --- |
| 0 | 1,0877 |
| 10 | 0,5430 |
| 100 | 0,0898 |
| 1000 | 0,0091 |
| 2000 | 0,0045 |

Ao final, a rede acerta as 10 compras do conjunto de treino.

## Limitações

- Os dados são inventados e há apenas 10 compras, o que serve para explicar o mecanismo, mas não para detectar fraude de verdade.
- Não há separação entre dados de treino e de teste, então o acerto de 10/10 não mede a capacidade de generalizar.
- Um único neurônio só separa classes com uma fronteira linear. Problemas mais complexos exigem camadas ocultas.
- Foi usado o erro quadrático por ser simples de explicar. Para classificação 0/1, a entropia cruzada (cross-entropy) costuma ser mais adequada com a sigmoid.

## Estrutura

```
fraude_perceptron.py   # programa completo
README.md              # este arquivo
```

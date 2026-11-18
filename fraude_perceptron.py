"""Perceptron com sigmoid para detectar fraude em cartão.

Entradas de cada compra (todas entre 0 e 1):
  x1 = valor da compra em relação ao gasto habitual do cliente
  x2 = distância do local habitual de compra
  x3 = horário incomum (0 = normal, 1 = madrugada)
Saída esperada: 1 = fraude, 0 = legítima.
"""
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def prever(pesos, x):
    """Soma ponderada (z = soma de wi*xi) seguida da sigmoid."""
    z = sum(w * xi for w, xi in zip(pesos, x))
    return sigmoid(z)


def erro(y, y_hat):
    """Erro quadrático: E = 1/2 * (y - y_hat)^2."""
    return 0.5 * (y - y_hat) ** 2


def atualizar(pesos, x, y, eta):
    """Um passo de backpropagation: w <- w - eta * dE/dw."""
    y_hat = prever(pesos, x)
    delta = (y_hat - y) * y_hat * (1 - y_hat)  # termo comum a todos os pesos
    return [w - eta * delta * xi for w, xi in zip(pesos, x)]


# Cada compra: ([x0=1, x1, x2, x3], y)
DADOS = [
    ([1, 0.90, 0.70, 0.20], 1),  # valor alto e longe de casa -> fraude
    ([1, 0.80, 0.90, 0.90], 1),  # valor alto, longe, madrugada -> fraude
    ([1, 0.95, 0.40, 0.80], 1),  # valor muito alto, madrugada -> fraude
    ([1, 0.70, 0.80, 0.70], 1),  # longe, madrugada -> fraude
    ([1, 0.10, 0.10, 0.10], 0),  # compra pequena e perto -> legítima
    ([1, 0.20, 0.05, 0.20], 0),
    ([1, 0.30, 0.20, 0.00], 0),
    ([1, 0.15, 0.30, 0.10], 0),
    ([1, 0.50, 0.10, 0.20], 0),
    ([1, 0.85, 0.10, 0.10], 0),  # valor alto, mas perto e de dia -> legítima
]

PESOS_INICIAIS = [-0.5, 0.3, 0.2, -0.1]  # w0, w1, w2, w3
ETA = 0.5  # taxa de aprendizado
EPOCAS = 2000


def passo_a_passo():
    """Mostra, para a primeira compra, os cálculos do vídeo."""
    x, y = DADOS[0]
    pesos = PESOS_INICIAIS
    z = sum(w * xi for w, xi in zip(pesos, x))
    y_hat = sigmoid(z)
    delta = (y_hat - y) * y_hat * (1 - y_hat)
    print("--- Passo a passo (1ª compra) ---")
    print(f"z = {z:.4f}")
    print(f"y_hat = sigmoid(z) = {y_hat:.4f}  (esperado: {y})")
    print(f"erro = {erro(y, y_hat):.4f}")
    print(f"delta = {delta:.4f}")
    novos = atualizar(pesos, x, y, ETA)
    for i, (antigo, novo) in enumerate(zip(pesos, novos)):
        gradiente = delta * x[i]
        print(f"w{i}: gradiente = {gradiente:.4f}  {antigo:.4f} -> {novo:.4f}")
    y_novo = prever(novos, x)
    print(f"nova saída = {y_novo:.4f}  novo erro = {erro(y, y_novo):.4f}\n")


def treinar():
    pesos = PESOS_INICIAIS
    print("--- Treinamento ---")
    for epoca in range(EPOCAS + 1):
        erro_total = sum(erro(y, prever(pesos, x)) for x, y in DADOS)
        if epoca in (0, 1, 10, 100, 500, 1000, EPOCAS):
            print(f"época {epoca:5d} | erro total = {erro_total:.4f}")
        for x, y in DADOS:
            pesos = atualizar(pesos, x, y, ETA)
    return pesos


def avaliar(pesos):
    print("\n--- Resultado final ---")
    acertos = 0
    for x, y in DADOS:
        p = prever(pesos, x)
        classe = 1 if p >= 0.5 else 0
        acertos += classe == y
        print(f"compra {x[1:]} -> prob. fraude = {p:.3f} | previsto = {classe} | real = {y}")
    print(f"acertos: {acertos}/{len(DADOS)}")
    print("pesos finais:", [round(w, 3) for w in pesos])

    nova = [1, 0.95, 0.85, 0.90]  # compra nova, nunca vista no treino
    print(f"\nCompra nova {nova[1:]} -> prob. fraude = {prever(pesos, nova):.3f}")


if __name__ == "__main__":
    passo_a_passo()
    pesos_finais = treinar()
    avaliar(pesos_finais)

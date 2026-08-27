


# ATIVIDADE PRÁTICA: TRATAMENTO DE DADOS, ÁRVORE DE DECISÃO E KNN

**Professora Karize Viecelli** @karizeviecelli — 2026

**Disciplina:** Inteligência Artificial | **Turma:** _____ | **Data:** _____

**Objetivo:** aplicar o pipeline completo de ML supervisionado — tratamento de dados,
modelagem com Árvore de Decisão e KNN, geração de gráficos e relatório técnico.

---

## 1. Recapitulação Ativa (Guia de Estudos)

### 1.1 Tratamento de Dados (Pré-processamento)
> Analogia: **"limpar a cozinha antes de cozinhar"** — nenhuma receita fica boa com
> ingredientes sujos ou misturados. O modelo é tão bom quanto os dados que o alimentam.

1. **Limpeza:** remover duplicatas e corrigir inconsistências.
2. **Valores ausentes:** remover linhas ou preencher (média, mediana, moda).
3. **Normalização/Padronização:** colocar variáveis numéricas na mesma escala (0–1 ou z-score) — essencial para o KNN.
4. **Encoding:** transformar variáveis categóricas em números (One-Hot ou Label Encoding).
5. **Divisão treino/teste:** ex. 70/30 — testar o modelo em dados que ele nunca viu.

### 1.2 Árvore de Decisão
> Analogia: **"jogo de 20 perguntas"** — cada pergunta (nó) divide o grupo até chegar
> à resposta (folha).

- Estrutura: nó raiz → nós internos → ramos → folhas.
- Critérios de divisão: **Índice de Gini** e **Entropia** (medem a pureza da divisão).
- Vantagem: **alta interpretabilidade** — é possível explicar o porquê de cada decisão.

### 1.3 KNN (K-Vizinhos Mais Próximos)
> Analogia: **"diga-me com quem andas que te direi quem és"**.

1. Escolher **k** (nº de vizinhos).
2. Calcular a **distância euclidiana** do novo ponto até todos os outros.
3. Selecionar os **k vizinhos** mais próximos.
4. Aplicar **voto majoritário**: a classe mais frequente vence.

### 1.4 Gráficos e Relatórios
- Gráficos: histograma do target, barras, **scatter plot** colorido pelo target, **matriz de confusão**.
- Métricas: **Acurácia, Precisão, Recall, F1-Score**.

---

## 2. Estudos de Caso e Exemplos Reais

| Caso | Algoritmo típico | Features | Target |
|------|------------------|----------|--------|
| **Netflix** — recomendação | KNN (usuários vizinhos) | histórico, gêneros, avaliações | vai gostar ou não |
| **Bancos** — análise de crédito | Árvore de Decisão | renda, idade, histórico | adimplente/inadimplente |
| **Medicina** — diagnóstico | Árvore + KNN | sintomas, exames, idade | normal/alterado |

---

## 3. Exercícios de Fixação

### 3.1 Conceituais (discursivas)
a) Explique, com uma analogia própria, por que o tratamento de dados é a etapa mais importante de um projeto de ML.
b) Diferencie Árvore de Decisão e KNN quanto à interpretabilidade. Quando escolher cada um?
c) O que acontece com o KNN com k=1 e com k=100 (dataset de 120 amostras)? Justifique.

### 3.2 Análise de dataset
| horas_estudo | horas_sono | aprovado |
|:---:|:---:|:---:|
| 2 | 8 | Não |
| 3 | 7 | Não |
| 5 | 6 | Sim |
| 6 | 5 | Sim |

Identifique: (a) nº de instâncias; (b) features; (c) target; (d) supervisionado ou não? Justifique.

### 3.3 Classificação de cenários
a) Agrupar clientes por comportamento de compra sem rótulos → ?
b) Prever se e-mail é spam com exemplos rotulados → ?
c) Robô aprende a andar recebendo recompensas → ?
d) Classificar tumores benignos/malignos com exames rotulados → ?

### 3.4 Prática KNN (cálculo manual)
Classifique a nova instância **horas_estudo=4, horas_sono=6** com **k=3**, usando o dataset da 3.2.

$$d(P,Q) = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$$

### 3.5 Prática Árvore de Decisão
Dataset "Jogar Tênis" (14 instâncias): Clima, Temperatura, Umidade, Vento → Jogar.
Comece pelo atributo **Clima** e construa a árvore. Descreva as regras em linguagem natural.

### 3.6 Desafio de Impacto Social / Inclusão Digital
Proponha um projeto de IA supervisionada que ajude pessoas com **TDAH ou autismo**.
Descreva: problema, features, target e como a interpretabilidade ajuda educadores e famílias.

---

## 4. Gabarito Comentado (exclusivo do professor)

**4.1** a) "Lixo entra, lixo sai" — sem limpeza o modelo aprende ruídos. b) Árvore = "caixa branca" (regras claras); KNN = vizinhança local, menos explicável. c) k=1 → *overfitting* (sensível a ruído); k=100 → *underfitting* (classe majoritária domina).

**4.2** a) 4 instâncias; b) features: horas_estudo, horas_sono; c) target: aprovado; d) **supervisionado**, pois há rótulos históricos.

**4.3** a) Não supervisionado; b) Supervisionado; c) Reforço; d) Supervisionado.

**4.4** Distâncias de (4,6):
- (2,8): √8 ≈ **2,83** → Não
- (3,7): √2 ≈ **1,41** → Não
- (5,6): √1 = **1,00** → Sim
- (6,5): √5 ≈ **2,24** → Sim

Ordenadas: 1,00 (Sim) < 1,41 (Não) < 2,24 (Sim) < 2,83 (Não).
**k=3** → vizinhos: Sim, Não, Sim → voto 2×Sim × 1×Não → **APROVADO**. 
*Dica: usar k ímpar evita empates.*

**4.5** Árvore:
```
Clima
├── Nublado → Sim
├── Sol → Umidade
│   ├── Alta → Não
│   └── Normal → Sim
└── Chuva → Vento
    ├── Fraco → Sim
    └── Forte → Não
```
Regra: "Joga-se tênis se nublado; com sol, apenas se umidade normal; com chuva, apenas se vento fraco."

**4.6** Resposta aberta. Critérios: problema relevante p/ neurodiversidade; features plausíveis (tempo de foco, pausas, erros); target claro (nível de atenção); justificar árvore pela interpretabilidade. Debater ética e privacidade dos dados.

---

## 5. Entrega do Relatório Final
1. Dataset tratado e etapas de pré-processamento.
2. Código/passos de modelagem.
3. Gráficos: distribuição do target, scatter plot, matriz de confusão.
4. Tabela comparativa de métricas (Acurácia, Precisão, Recall) entre os dois modelos.
5. Conclusão crítica sobre o melhor modelo para o problema.
```

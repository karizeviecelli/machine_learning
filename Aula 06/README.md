# AULA 06 --- Como avaliar um modelo de classificação?

**Trilha de IA Aplicada --- Base Comum dos Projetos**\
**Carga horária sugerida:** 4 horas\
**Pré-requisitos:** preparação de dados, features, target, treino/teste,
Árvore de Decisão, KNN e acurácia.

**[Click aqui - Link Colab](https://colab.research.google.com/drive/1sFgVNPYzDT5Qr_BsDkU1dyDyibZNi6FL?usp=sharing)**

------------------------------------------------------------------------

## 1. Onde estamos?

Até agora aprendemos a:

``` text
Preparar dados
      ↓
Escolher features e target
      ↓
Separar treino e teste
      ↓
Treinar um modelo
      ↓
Fazer previsões
      ↓
Calcular acurácia
```

Agora surge uma pergunta muito importante:

> **A acurácia sozinha é suficiente para dizer que um modelo é bom?**

Nesta aula vamos aprender a olhar os **acertos e erros** do modelo com
mais cuidado.

------------------------------------------------------------------------

# 2. Situação-problema

Imagine um dataset com 100 registros:

``` text
90 → classe NÃO
10 → classe SIM
```

Um modelo muito ruim decide responder sempre:

``` text
NÃO
```

Resultado:

``` text
90 acertos
10 erros
```

Acurácia:

``` text
90%
```

Parece excelente.

Mas o modelo encontrou algum caso da classe `SIM`?

``` text
NÃO.
```

Esse exemplo mostra que:

> **Uma acurácia alta não garante que o modelo esteja resolvendo bem o
> problema.**

------------------------------------------------------------------------

# 3. Por que isso importa nos projetos?

## Projeto Cuidadores

O modelo pode tentar classificar:

``` text
atrasou / não atrasou
```

## Projeto NR-1

Pode classificar indicadores em níveis de atenção:

``` text
baixo / atenção / elevado / crítico
```

## Fiscalização Colaborativa

Pode classificar:

``` text
baixa / média / alta prioridade
```

## Agenda Inteligente

Pode classificar tarefas:

``` text
baixa / média / alta prioridade
```

Em todos os casos precisamos saber **que tipo de erro o modelo está
cometendo**.

------------------------------------------------------------------------

# 4. Relembrando a acurácia

A acurácia responde:

> **De todas as previsões realizadas, quantas o modelo acertou?**

``` text
Acurácia = acertos / total
```

Exemplo:

``` text
80 previsões
72 corretas
```

``` text
72 / 80 = 0,90
```

Acurácia:

``` text
90%
```

Ela continua sendo útil.

O problema é utilizá-la **sozinha**.

------------------------------------------------------------------------

# 5. Matriz de Confusão

A Matriz de Confusão mostra como as previsões se distribuíram entre as
classes reais.

Para começar, usaremos classificação binária:

``` text
0 = NÃO
1 = SIM
```

Exemplo:

  |             |Previsto NÃO |  Previsto SIM|
  |----------|--------------|--------------|
  |Real NÃO  |             50 |             5|
  |Real SIM  |             10  |           35|
  ----------------------------------------------

Ela permite enxergar não apenas quantos erros aconteceram, mas **quais
erros** aconteceram.

------------------------------------------------------------------------

# 6. Quatro resultados possíveis

Em uma classificação binária temos quatro situações.

## Verdadeiro Positivo --- VP

O modelo previu:

``` text
SIM
```

e o resultado real era:

``` text
SIM
```

Acertou.

## Verdadeiro Negativo --- VN

Previu:

``` text
NÃO
```

e o real era:

``` text
NÃO
```

Acertou.

## Falso Positivo --- FP

Previu:

``` text
SIM
```

mas o real era:

``` text
NÃO
```

Errou.

## Falso Negativo --- FN

Previu:

``` text
NÃO
```

mas o real era:

``` text
SIM
```

Errou.

------------------------------------------------------------------------

# 7. Analogia: detector de chuva

Imagine um sistema que prevê:

``` text
Vai chover? SIM / NÃO
```

### Verdadeiro Positivo

Previu chuva e choveu.

### Verdadeiro Negativo

Previu que não choveria e realmente não choveu.

### Falso Positivo

Previu chuva, mas não choveu.

Você carregou guarda-chuva à toa.

### Falso Negativo

Previu que não choveria, mas choveu.

Você chegou ensopado. 😄

Os dois erros são diferentes.

Dependendo do problema, um deles pode ser mais importante.

------------------------------------------------------------------------

# 8. Matriz de Confusão no Scikit-Learn

``` python
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(
    y_teste,
    previsoes
)

print(matriz)
```

Exemplo:

``` text
[[50  5]
 [10 35]]
```

A leitura padrão é:

``` text
[[VN FP]
 [FN VP]]
```

------------------------------------------------------------------------

# 9. Visualizando a matriz

``` python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_teste,
    previsoes
)

plt.show()
```

A visualização ajuda a perceber onde o modelo está confundindo as
classes.

------------------------------------------------------------------------

# 10. Precisão

A **precisão** responde:

> **Quando o modelo disse SIM, quantas vezes ele estava correto?**

Exemplo:

O modelo disse `SIM` 20 vezes.

Dessas:

``` text
15 estavam corretas
5 estavam erradas
```

Precisão:

``` text
15 / 20 = 75%
```

Conceitualmente:

``` text
Precisão = VP / (VP + FP)
```

------------------------------------------------------------------------

# 11. Analogia da precisão

Imagine um detector de spam.

Ele marcou 20 mensagens como spam.

Mas apenas 10 realmente eram spam.

A precisão é baixa porque:

> O modelo acusa muitos casos que não pertencem à classe procurada.

------------------------------------------------------------------------

# 12. Recall

O **recall** responde:

> **De todos os casos SIM que realmente existiam, quantos o modelo
> conseguiu encontrar?**

Exemplo:

Existiam 20 casos reais da classe `SIM`.

O modelo encontrou 18.

``` text
Recall = 18 / 20 = 90%
```

Conceitualmente:

``` text
Recall = VP / (VP + FN)
```

------------------------------------------------------------------------

# 13. Analogia do recall

Imagine que existem 100 bolas vermelhas em uma caixa.

Seu sistema encontrou apenas 60.

Mesmo que quase tudo o que ele encontrou seja realmente vermelho, ele
deixou muitas bolas vermelhas para trás.

Seu recall seria baixo.

------------------------------------------------------------------------

# 14. Precisão × Recall

Guarde estas duas perguntas:

``` text
PRECISÃO
Quando o modelo disse SIM,
quanto podemos confiar nesse SIM?
```

``` text
RECALL
Dos SIM que realmente existiam,
quantos o modelo encontrou?
```

Não precisamos decorar apenas fórmulas.

Precisamos compreender as perguntas.

------------------------------------------------------------------------

# 15. F1-score

Às vezes queremos uma medida que considere precisão e recall juntas.

Para isso existe o:

> **F1-score**

Ele combina as duas métricas.

``` text
Precisão alta + Recall alto
            ↓
       F1 tende a ser alto
```

Se uma das duas for muito baixa, o F1 também será prejudicado.

Para esta aula, o mais importante é:

> **F1 ajuda a observar o equilíbrio entre precisão e recall.**

------------------------------------------------------------------------

# 16. Classification Report

O Scikit-Learn consegue calcular várias métricas:

``` python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_teste,
        previsoes
    )
)
```

Resultado semelhante a:

``` text
              precision   recall   f1-score

0                0.85      0.91       0.88
1                0.88      0.78       0.82
```

Não se assuste com a tabela.

Vamos lê-la coluna por coluna.

------------------------------------------------------------------------

# 17. Classes desbalanceadas

Quando uma classe aparece muito mais que outra, temos um problema de:

> **desbalanceamento de classes**

Exemplo:

``` text
Classe 0 → 950 registros
Classe 1 → 50 registros
```

Um modelo que respondesse sempre `0` teria:

``` text
95% de acurácia
```

mas seria inútil para encontrar a classe `1`.

Por isso, antes de comemorar uma acurácia alta, verifique:

``` python
y.value_counts()
```

------------------------------------------------------------------------

# 18. Qual métrica é mais importante?

Não existe uma resposta universal.

Depende do problema.

Pergunte:

> **Qual erro custa mais para a aplicação?**

Essa pergunta é mais importante do que decorar métricas.

------------------------------------------------------------------------

# 19. Aplicação aos projetos

## Cuidadores

Se o modelo estiver estudando atrasos:

``` text
FN = havia atraso, mas o modelo classificou como não atraso
```

## NR-1

Em um modelo educacional de classificação agregada de indicadores:

``` text
FN = setor em classe de atenção que o modelo não identificou
```

A análise deve permanecer agregada, sem diagnóstico individual.

## Fiscalização

``` text
FN = ocorrência prioritária classificada como não prioritária
```

## Agenda

``` text
FP = tarefa simples classificada como prioridade alta
```

Perceba que o significado do erro muda com o problema.

------------------------------------------------------------------------

# 20. Código completo básico

``` python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Acurácia
acuracia = accuracy_score(
    y_teste,
    previsoes
)

print("Acurácia:", acuracia)

# Matriz de confusão
matriz = confusion_matrix(
    y_teste,
    previsoes
)

print("Matriz de confusão:")
print(matriz)

# Relatório
print(
    classification_report(
        y_teste,
        previsoes
    )
)
```

------------------------------------------------------------------------

# 21. Atividade de fixação 1

Considere:

``` text
VP = 30
VN = 50
FP = 10
FN = 10
```

Responda:

1.  Quantos registros existem no total?
2.  Quantos foram classificados corretamente?
3.  Quantos foram classificados incorretamente?
4.  Qual erro representa um `SIM` previsto incorretamente?
5.  Qual erro representa um `SIM` real que não foi encontrado?

------------------------------------------------------------------------

# 22. Atividade de fixação 2

Um modelo apresenta:

``` text
Acurácia = 94%
Recall da classe importante = 20%
```

Responda:

1.  Podemos olhar apenas para os 94%?
2.  O que o recall de 20% sugere?
3.  Que informação você procuraria no dataset?
4.  A distribuição das classes pode explicar esse comportamento?

------------------------------------------------------------------------

# 23. Prática principal

Utilize o modelo criado na Aula 01.

Calcule:

-   acurácia;
-   matriz de confusão;
-   precisão;
-   recall;
-   F1-score.

Depois compare:

``` text
KNN
×
Árvore de Decisão
```

Tabela:

  Modelo     Acurácia   Precisão   Recall   F1
  -------- ---------- ---------- -------- ----
  KNN                                     
  Árvore                                  

------------------------------------------------------------------------

# 24. Perguntas para a conclusão

O grupo deverá responder:

1.  Qual modelo teve maior acurácia?
2.  Qual teve maior precisão?
3.  Qual teve maior recall?
4.  Qual teve maior F1?
5.  Os modelos cometeram os mesmos tipos de erro?
6.  Qual modelo vocês escolheriam para o experimento?
7.  Por quê?

------------------------------------------------------------------------

# 25. Erros comuns

## Erro 1

Dizer:

> "Meu modelo tem 95% de acurácia, então está perfeito."

## Erro 2

Ignorar a quantidade de exemplos de cada classe.

## Erro 3

Confundir precisão com acurácia.

## Erro 4

Decorar VP, VN, FP e FN sem relacioná-los ao problema.

## Erro 5

Escolher o modelo apenas porque uma métrica ficou alguns pontos maior.

------------------------------------------------------------------------

# 26. Questões de revisão

1.  O que mede a acurácia?
2.  O que é uma matriz de confusão?
3.  O que significa VP?
4.  O que significa VN?
5.  O que significa FP?
6.  O que significa FN?
7.  O que a precisão pergunta?
8.  O que o recall pergunta?
9.  O que o F1-score combina?
10. O que são classes desbalanceadas?
11. Por que 95% de acurácia pode esconder um modelo ruim?
12. Qual métrica devemos usar sempre? Explique por que a resposta não é
    simplesmente uma única métrica.

------------------------------------------------------------------------

# 27. Desafio final

Crie dois modelos diferentes para o mesmo dataset.

Compare:

``` text
Acurácia
Matriz de Confusão
Precisão
Recall
F1-score
```

Depois escreva uma conclusão:

> **Qual modelo você escolheria e quais evidências sustentam sua
> escolha?**

------------------------------------------------------------------------

# 28. Entrega

Notebook:

``` text
02_metricas_classificacao.ipynb
```

Deve conter:

-   modelo treinado;
-   previsões;
-   distribuição das classes;
-   acurácia;
-   matriz de confusão;
-   classification report;
-   comparação entre dois modelos;
-   interpretação escrita.

------------------------------------------------------------------------

# 29. Resumo da aula

``` text
ACURÁCIA
   ↓
quantos acertos?

MATRIZ DE CONFUSÃO
   ↓
onde acertou e errou?

PRECISÃO
   ↓
quando disse SIM, acertou?

RECALL
   ↓
dos SIM reais, quantos encontrou?

F1
   ↓
equilíbrio entre precisão e recall
```

Na próxima aula entraremos em um tipo de dado diferente:

> **texto.**

E a pergunta será:

> **Como transformar linguagem humana em números para que um algoritmo
> possa trabalhar com ela?**

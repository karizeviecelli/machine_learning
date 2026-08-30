# AULA 05 --- KNN: Classificação por Vizinhos Mais Próximos

**Trilha de IA Aplicada --- Projeto Cuidadores**\
**Carga horária sugerida:** 4 horas\
**Nível:** introdutório, após Árvore de Decisão e acurácia

**[Click aqui - Link Colab](https://colab.research.google.com/drive/1L1kqqjCdKLxwUOG6_E3UmQPqijEpHIgL?usp=sharing)**
------------------------------------------------------------------------

## 1. Onde estamos na trilha?

Até aqui, já estudamos:

-   preparação e tratamento de dados;
-   dataset;
-   features e target;
-   separação entre treino e teste;
-   Árvore de Decisão;
-   treinamento de um modelo;
-   `predict()`;
-   acurácia.

Nesta aula vamos conhecer um **novo algoritmo de Machine Learning
supervisionado**: o **KNN --- K-Nearest Neighbors**.

A ideia não é abandonar a Árvore de Decisão. Pelo contrário: vamos
aprender que um mesmo problema pode ser resolvido por algoritmos
diferentes e que podemos **comparar seus resultados**.

------------------------------------------------------------------------

# 2. Situação-problema

No aplicativo de apoio aos cuidadores, os medicamentos possuem horários
programados.

Com o passar do tempo, o sistema acumula registros como:

 | Dia    |   Turno   |  Lembretes  | Atraso médio |  anterior Atrasou?  |
 |--------|-----------|-------------|--------------|---------------------|
 |Segunda |  Manhã    |      1      |  2 min       |      Não            |
 |Segunda |  Noite    |      2      | 15 min       |      Sim            |
 |Terça   |  Manhã    |      1      |  0 min       |      Não            |
 |Sexta   |  Noite    |      3      | 24 min       |      Sim            |
 |Domingo |  Tarde    |      2      | 12 min       |      Sim            |
---------------------------------------------------------------------------
Queremos investigar:

> **A partir de registros anteriores, é possível classificar uma nova
> situação como maior ou menor possibilidade de atraso?**

Para fins didáticos, vamos representar:

``` text
0 → não atrasou
1 → atrasou
```

> **Importante:** o modelo é uma atividade educacional. Um aplicativo
> real precisaria de dados adequados, validação e avaliação antes de
> utilizar previsões no cuidado de pessoas.

------------------------------------------------------------------------

# 3. Antes do KNN: o que é classificação?

Machine Learning pode ser utilizado para diferentes tipos de problemas.

Quando queremos prever uma **categoria**, temos um problema de
**classificação**.

Exemplos:

``` text
E-mail → spam / não spam

Imagem → gato / cachorro

Cliente → compra / não compra

Medicamento → atrasou / não atrasou
```

No nosso exemplo:

``` text
TARGET = medicamento_atrasado
```

As classes são:

``` text
0 = não
1 = sim
```

Portanto, estamos diante de um problema de:

> **Classificação supervisionada.**

------------------------------------------------------------------------

# 4. O que significa aprendizado supervisionado?

No aprendizado supervisionado, o algoritmo aprende utilizando exemplos
em que a resposta correta já é conhecida.

Imagine:

   | Lembretes |  Atraso médio | Resultado |
   |-----------|---------------|-----------|
   |    1      |        2      | Não atrasou |
   |    3      |       22      |  Atrasou  |
   |    1      |        1      |Não atrasou|
   |    2      |      18       |Atrasou  |
   ------------------------------------------

O modelo recebe exemplos contendo:

``` text
ENTRADAS + RESPOSTA
```

ou:

``` text
FEATURES + TARGET
```

Durante o treinamento, ele procura padrões que relacionem essas
informações.

Depois podemos apresentar um novo registro:

``` text
Lembretes = 3
Atraso médio = 20
```

e solicitar uma previsão.

------------------------------------------------------------------------

# 5. Relembrando Features e Target

## Features

São as características usadas como entrada do modelo.

No projeto:

``` python
X = dados[
    [
        "quantidade_lembretes",
        "atraso_medio"
    ]
]
```

## Target

É a variável que queremos prever.

``` python
y = dados["medicamento_atrasado"]
```

Podemos visualizar assim:

``` text
FEATURES                         TARGET
   ↓                               ↓
quantidade_lembretes ───────┐
                            ├──► medicamento_atrasado
atraso_medio ───────────────┘
```

------------------------------------------------------------------------

# 6. O que é KNN?

KNN significa:

> **K-Nearest Neighbors**

Em português:

> **K Vizinhos Mais Próximos**

É um algoritmo que classifica um novo exemplo observando os exemplos
conhecidos que estão **mais próximos** dele.

A lógica básica é:

``` text
Novo registro
     ↓
Encontrar os K registros mais próximos
     ↓
Observar as classes desses vizinhos
     ↓
A classe mais frequente vence
     ↓
Previsão
```

------------------------------------------------------------------------

# 7. Analogia: escolhendo pelo grupo mais parecido

Imagine que uma pessoa chega a uma turma e você quer descobrir se ela
provavelmente prefere estudar pela manhã ou à noite.

Você procura alunos com características semelhantes.

Os três mais parecidos dizem:

``` text
Aluno A → manhã
Aluno B → manhã
Aluno C → noite
```

A maioria é:

``` text
MANHÃ
```

O KNN trabalha com uma ideia semelhante.

A diferença é que ele usa **distância matemática** para determinar quem
são os vizinhos mais próximos.

------------------------------------------------------------------------

# 8. O que é o K?

A letra **K** indica quantos vizinhos serão considerados.

Se:

``` text
K = 3
```

o algoritmo observa os **3 vizinhos mais próximos**.

Exemplo:

``` text
Vizinho 1 → ATRASOU
Vizinho 2 → ATRASOU
Vizinho 3 → NÃO ATRASOU
```

Resultado:

``` text
2 votos → ATRASOU
1 voto  → NÃO ATRASOU
```

Previsão:

``` text
ATRASOU
```

------------------------------------------------------------------------

# 9. O valor de K muda o resultado?

Pode mudar.

Imagine os vizinhos ordenados pela proximidade:

``` text
1º → SIM
2º → SIM
3º → NÃO
4º → NÃO
5º → NÃO
```

Com:

``` text
K = 3
```

temos:

``` text
SIM, SIM, NÃO
```

Resultado:

``` text
SIM
```

Com:

``` text
K = 5
```

temos:

``` text
SIM, SIM, NÃO, NÃO, NÃO
```

Resultado:

``` text
NÃO
```

Portanto:

> **A escolha de K influencia o comportamento do modelo.**

------------------------------------------------------------------------

# 10. K muito pequeno × K muito grande

## K muito pequeno

Exemplo:

``` text
K = 1
```

O modelo considera apenas um vizinho.

Pode ficar muito sensível a casos específicos ou ruídos nos dados.

## K muito grande

Se K for muito grande, o modelo pode considerar muitos registros pouco
semelhantes ao novo exemplo.

### Ideia central

Não existe um valor mágico de K que seja sempre o melhor.

Podemos experimentar diferentes valores e avaliar o modelo.

------------------------------------------------------------------------

# 11. Por que normalmente usamos K ímpar?

Em classificação binária, valores ímpares ajudam a reduzir empates.

Por exemplo:

Com:

``` text
K = 4
```

poderíamos obter:

``` text
SIM
SIM
NÃO
NÃO
```

Temos empate.

Com:

``` text
K = 3
```

sempre haverá maioria entre duas classes.

> Isso não significa que K sempre precise ser ímpar, mas é uma escolha
> didática útil em problemas binários.

------------------------------------------------------------------------

# 12. Como o KNN sabe quem está perto?

Precisamos falar de **distância**.

Imagine dois registros:

``` text
Registro A
lembretes = 1
atraso médio = 2

Registro B
lembretes = 2
atraso médio = 4
```

E um novo registro:

``` text
lembretes = 2
atraso médio = 3
```

Visualmente, o novo registro parece mais próximo desses casos do que de
um registro como:

``` text
lembretes = 5
atraso médio = 40
```

O KNN calcula matematicamente essas distâncias.

------------------------------------------------------------------------

# 13. Distância Euclidiana

Uma das distâncias mais conhecidas é a **distância Euclidiana**.

Ela vem da mesma ideia usada para calcular a distância entre pontos em
um plano.

Para dois pontos:

``` text
A = (x1, y1)
B = (x2, y2)
```

a distância é:

``` text
d = √((x2 - x1)² + (y2 - y1)²)
```

Não precisamos decorar a fórmula para utilizar o KNN.

O importante é compreender:

> Quanto menor a distância, mais próximos os registros são considerados.

------------------------------------------------------------------------

# 14. Exemplo visual

Considere duas features:

``` text
X → quantidade de lembretes
Y → atraso médio
```

Podemos imaginar:

``` text
atraso
 médio

 25 |                    ● SIM
 20 |                 ● SIM
 15 |              ? NOVO
 10 |
  5 |   ○ NÃO
  0 | ○ NÃO
    +-------------------------
       1   2   3   4
          lembretes
```

O novo ponto será classificado observando os pontos mais próximos.

------------------------------------------------------------------------

# 15. Um problema importante: escalas diferentes

Considere estas features:

``` text
quantidade_lembretes = 2

atraso_medio = 35
```

Uma feature varia aproximadamente entre:

``` text
0 e 5
```

A outra pode variar entre:

``` text
0 e 60
```

A segunda possui números muito maiores.

Como o KNN utiliza distância, essa diferença pode fazer uma variável ter
influência muito maior simplesmente por causa da sua escala.

------------------------------------------------------------------------

# 16. Padronização

Para resolver esse problema, podemos **padronizar** os dados.

A padronização transforma as features para que trabalhem em escalas
comparáveis.

No Scikit-Learn podemos utilizar:

``` python
from sklearn.preprocessing import StandardScaler
```

Criamos o objeto:

``` python
scaler = StandardScaler()
```

------------------------------------------------------------------------

# 17. fit_transform() e transform()

No conjunto de treino:

``` python
X_treino_escalado = scaler.fit_transform(X_treino)
```

Aqui acontecem duas coisas:

``` text
FIT       → aprende os parâmetros do conjunto de treino
TRANSFORM → transforma os dados
```

No teste:

``` python
X_teste_escalado = scaler.transform(X_teste)
```

Observe que não fazemos outro `fit`.

### Por quê?

Porque o conjunto de teste representa dados que o modelo não deveria
conhecer durante a preparação do treinamento.

------------------------------------------------------------------------

# 18. Analogia do StandardScaler

Imagine duas réguas:

``` text
Régua A → centímetros
Régua B → metros
```

Antes de comparar as medidas, precisamos colocá-las em uma referência
compatível.

O `StandardScaler` exerce um papel parecido para as features numéricas.

------------------------------------------------------------------------

# 19. KNN × Árvore de Decisão

Vocês já conhecem a Árvore de Decisão.

Agora podemos comparar os conceitos.

| Característica | Árvore de Decisão | KNN |
|---|---|---|
| Ideia principal | Cria decisões e regras a partir das features | Procura exemplos próximos |
| Usa vizinhos? | Não | Sim |
| A distância é importante? | Não da mesma forma | Sim |
| A escala costuma ser crítica? | Menos sensível | Muito importante |
| Possui o parâmetro K? | Não | Sim |
| Pode classificar? | Sim | Sim |
| É supervisionado? | Sim | Sim |

### Importante

Não estamos perguntando:

> "Qual algoritmo é melhor no mundo?"

Estamos perguntando:

> **"Qual funciona melhor para este problema e estes dados?"**

------------------------------------------------------------------------

# 20. Preparando o ambiente

``` python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

------------------------------------------------------------------------

# 21. Carregando os dados

``` python
dados = pd.read_csv("atrasos_medicamentos.csv")

dados.head()
```

Antes de treinar qualquer modelo:

``` python
dados.info()
```

E:

``` python
dados.isnull().sum()
```

Mesmo que a turma já tenha estudado preparação de dados, devemos sempre
conferir o dataset.

------------------------------------------------------------------------

# 22. Selecionando Features e Target

Para começar, utilizaremos apenas features numéricas:

``` python
X = dados[
    [
        "quantidade_lembretes",
        "atraso_medio"
    ]
]

y = dados["medicamento_atrasado"]
```

### Por que começar com poucas features?

Porque queremos enxergar claramente o funcionamento do KNN antes de
aumentar a complexidade.

------------------------------------------------------------------------

# 23. Separando treino e teste

``` python
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Relembrando:

``` text
TREINO → modelo aprende

TESTE → verificamos como ele se comporta em dados separados
```

------------------------------------------------------------------------

# 24. Padronizando as Features

``` python
scaler = StandardScaler()

X_treino_escalado = scaler.fit_transform(X_treino)

X_teste_escalado = scaler.transform(X_teste)
```

------------------------------------------------------------------------

# 25. Criando o modelo KNN

``` python
modelo_knn = KNeighborsClassifier(
    n_neighbors=3
)
```

`n_neighbors=3` significa:

``` text
K = 3
```

------------------------------------------------------------------------

# 26. Treinando

``` python
modelo_knn.fit(
    X_treino_escalado,
    y_treino
)
```

Mesmo que o KNN tenha um funcionamento diferente da Árvore de Decisão,
usamos a interface padrão do Scikit-Learn:

``` text
modelo.fit()
```

------------------------------------------------------------------------

# 27. Fazendo previsões

``` python
previsoes = modelo_knn.predict(
    X_teste_escalado
)

print(previsoes)
```

------------------------------------------------------------------------

# 28. Avaliando com acurácia

``` python
acuracia = accuracy_score(
    y_teste,
    previsoes
)

print("Acurácia:", acuracia)
```

Se obtivermos:

``` text
0.85
```

podemos interpretar como:

``` text
85%
```

de acertos naquele conjunto de teste.

------------------------------------------------------------------------

# 29. Cuidado com a interpretação da acurácia

Uma acurácia alta não significa:

> "Nosso sistema está pronto para uso real."

Significa apenas que:

> O modelo acertou determinada proporção dos exemplos daquele conjunto
> de teste.

A qualidade de um modelo depende também de:

-   qualidade dos dados;
-   quantidade de dados;
-   representatividade;
-   distribuição das classes;
-   outras métricas;
-   validação adequada;
-   contexto de utilização.

------------------------------------------------------------------------

# 30. Testando diferentes valores de K

``` python
for k in [1, 3, 5, 7]:

    modelo = KNeighborsClassifier(
        n_neighbors=k
    )

    modelo.fit(
        X_treino_escalado,
        y_treino
    )

    previsoes = modelo.predict(
        X_teste_escalado
    )

    acuracia = accuracy_score(
        y_teste,
        previsoes
    )

    print(
        "K =", k,
        "| Acurácia =", acuracia
    )
```

Agora podemos comparar.

------------------------------------------------------------------------

# 31. Atividade de fixação 1 --- Entendendo o conceito

Considere os cinco vizinhos mais próximos:

``` text
Vizinho 1 → NÃO
Vizinho 2 → SIM
Vizinho 3 → SIM
Vizinho 4 → NÃO
Vizinho 5 → SIM
```

Responda:

1.  Qual seria a classificação com `K = 1`?
2.  Qual seria a classificação com `K = 3`?
3.  Qual seria a classificação com `K = 5`?
4.  O resultado mudou?
5.  O que isso demonstra sobre K?

------------------------------------------------------------------------

# 32. Atividade de fixação 2 --- Escala

Considere:

``` text
idade = 75
quantidade_medicamentos = 4
atraso_segundos = 1800
```

Responda:

1.  Qual feature possui valores numericamente maiores?
2.  Por que isso pode influenciar um algoritmo baseado em distância?
3.  Que ferramenta do Scikit-Learn podemos utilizar?
4.  Por que a Árvore de Decisão não apresenta exatamente a mesma
    necessidade?

------------------------------------------------------------------------

# 33. Prática principal --- Modelo de atraso

## Objetivo

Construir um classificador KNN para o target:

``` text
medicamento_atrasado
```

## Etapas

### Etapa 1

Carregue o dataset.

### Etapa 2

Observe:

``` python
dados.head()
dados.info()
dados.describe()
```

### Etapa 3

Defina X e y.

### Etapa 4

Faça a separação treino/teste.

### Etapa 5

Utilize `StandardScaler`.

### Etapa 6

Treine:

``` text
K = 3
```

### Etapa 7

Faça previsões.

### Etapa 8

Calcule a acurácia.

### Etapa 9

Teste:

``` text
K = 1
K = 3
K = 5
K = 7
```

### Etapa 10

Registre os resultados.

    K   Acurácia
  --- ----------
    1 
    3 
    5 
    7 

------------------------------------------------------------------------

# 34. Comparando com a Árvore de Decisão

Treine novamente uma Árvore de Decisão com os mesmos conjuntos de treino
e teste.

``` python
from sklearn.tree import DecisionTreeClassifier

arvore = DecisionTreeClassifier(
    random_state=42
)

arvore.fit(
    X_treino,
    y_treino
)

previsoes_arvore = arvore.predict(
    X_teste
)

acuracia_arvore = accuracy_score(
    y_teste,
    previsoes_arvore
)

print(
    "Acurácia da árvore:",
    acuracia_arvore
)
```

> Observe que, para esta comparação introdutória, a Árvore de Decisão
> pode trabalhar com as features originais.

------------------------------------------------------------------------

# 35. Tabela de comparação

Preencha:

  Modelo          Configuração            Acurácia
  --------------- --------------------- ----------
  KNN             K = 1                 
  KNN             K = 3                 
  KNN             K = 5                 
  KNN             K = 7                 
  Decision Tree   padrão da atividade   

Depois responda:

> Qual configuração apresentou maior acurácia neste experimento?

E:

> Podemos afirmar que esse modelo será sempre o melhor?

**Não.**

Nossa conclusão é válida para o experimento realizado e deve ser
analisada dentro do contexto dos dados utilizados.

------------------------------------------------------------------------

# 36. Fazendo uma nova previsão

Depois de treinar o modelo, queremos prever um novo caso.

Exemplo:

``` python
novo_registro = [[
    3,    # quantidade de lembretes
    20    # atraso médio
]]
```

Antes de enviar ao KNN, precisamos utilizar o mesmo `scaler`:

``` python
novo_registro_escalado = scaler.transform(
    novo_registro
)
```

Agora:

``` python
resultado = modelo_knn.predict(
    novo_registro_escalado
)

print(resultado)
```

------------------------------------------------------------------------

# 37. Erro comum: esquecer o scaler na nova previsão

Errado:

``` python
modelo_knn.predict([[3, 20]])
```

se o modelo foi treinado com dados padronizados.

Correto:

``` python
novo = scaler.transform([[3, 20]])

modelo_knn.predict(novo)
```

### Regra

> A nova entrada deve passar pelas mesmas transformações usadas para
> preparar os dados de treinamento.

------------------------------------------------------------------------

# 38. Erros comuns no KNN

## Erro 1 --- Não padronizar features com escalas muito diferentes

Pode distorcer as distâncias.

## Erro 2 --- Fazer `fit_transform()` separadamente no teste

O scaler deve aprender com o treino.

## Erro 3 --- Usar `fit_transform()` em cada nova previsão

Novos dados devem usar apenas:

``` python
scaler.transform()
```

## Erro 4 --- Escolher K sem testar

O valor de K influencia o resultado.

## Erro 5 --- Concluir que maior acurácia significa modelo perfeito

Nenhuma métrica isolada conta toda a história.

## Erro 6 --- Colocar textos diretamente no KNN

Features categóricas/textuais precisam ser transformadas antes de serem
usadas numericamente.

------------------------------------------------------------------------

# 39. KNN é Inteligência Artificial?

Dentro do contexto de Machine Learning:

**Sim.**

O KNN é um algoritmo de aprendizado de máquina supervisionado.

Mas perceba que ele não "pensa" como uma pessoa.

Ele executa um procedimento matemático:

``` text
recebe dados
   ↓
calcula proximidades
   ↓
encontra vizinhos
   ↓
observa as classes
   ↓
produz uma classificação
```

------------------------------------------------------------------------

# 40. KNN "aprende" como a Árvore?

Essa é uma ótima pergunta.

Na Árvore de Decisão, durante o `fit()`, o algoritmo constrói uma
estrutura de decisões.

No KNN, o comportamento é diferente: ele depende fortemente dos exemplos
de treinamento armazenados e realiza grande parte do trabalho quando
precisa classificar um novo exemplo.

Por isso, costuma ser chamado de um algoritmo de **aprendizado baseado
em instâncias**.

Para esta aula, basta guardar:

> **Árvore cria uma estrutura de decisão; KNN compara o novo exemplo com
> exemplos conhecidos.**

------------------------------------------------------------------------

# 41. Limitações do KNN

O KNN é excelente para aprender conceitos, mas possui limitações.

-   pode ficar mais lento com muitos registros;
-   é sensível à escala;
-   pode sofrer com features irrelevantes;
-   a escolha de K influencia o resultado;
-   dados com muitas dimensões podem dificultar a noção de proximidade.

Não precisamos resolver todos esses problemas agora.

O objetivo é aprender a reconhecer que:

> **Cada algoritmo possui vantagens e limitações.**

------------------------------------------------------------------------

# 42. O que aprendemos sobre Machine Learning nesta aula?

Mais importante que aprender um novo comando, aprendemos que:

``` text
PROBLEMA
   ↓
DADOS
   ↓
FEATURES / TARGET
   ↓
ALGORITMO
   ↓
TREINAMENTO
   ↓
PREVISÃO
   ↓
AVALIAÇÃO
   ↓
COMPARAÇÃO
```

Machine Learning não termina quando executamos:

``` python
modelo.fit()
```

Precisamos avaliar e interpretar o resultado.

------------------------------------------------------------------------

# 43. Questões de revisão

1.  O que significa KNN?
2.  O KNN é supervisionado ou não supervisionado nesta aplicação?
3.  O que significa o valor K?
4.  O que são vizinhos?
5.  Como o algoritmo determina quais exemplos estão próximos?
6.  O que é distância Euclidiana?
7.  Por que features em escalas diferentes podem ser um problema?
8.  Para que serve o `StandardScaler`?
9.  Qual a diferença entre `fit_transform()` e `transform()`?
10. Por que utilizamos `fit_transform()` no treino?
11. Por que usamos somente `transform()` no teste?
12. O que acontece se mudarmos K?
13. Por que valores ímpares de K podem ser úteis em classificação
    binária?
14. Qual a diferença conceitual entre KNN e Árvore de Decisão?
15. O modelo com maior acurácia é necessariamente perfeito?
16. O que devemos fazer com um novo registro antes de chamar `predict()`
    se treinamos o KNN com dados padronizados?

------------------------------------------------------------------------

# 44. Desafio final

Construa um experimento que teste:

``` text
K = 1
K = 3
K = 5
K = 7
K = 9
```

Depois:

1.  registre a acurácia de cada modelo;
2.  identifique o melhor resultado;
3.  compare com a Árvore de Decisão;
4.  escreva uma conclusão de pelo menos cinco linhas;
5.  explique por que não podemos escolher um algoritmo apenas pelo nome;
6.  explique por que os resultados obtidos não significam que o modelo
    está validado para uso real em saúde.

------------------------------------------------------------------------

# 45. Entrega

Entregar um notebook chamado:

``` text
01_knn_atraso_medicamentos.ipynb
```

O notebook deverá conter:

-   contextualização do problema;
-   carregamento do dataset;
-   análise inicial;
-   definição de features e target;
-   treino/teste;
-   `StandardScaler`;
-   KNN;
-   testes com diferentes valores de K;
-   acurácia;
-   comparação com Árvore de Decisão;
-   nova previsão;
-   conclusão do grupo.

------------------------------------------------------------------------

# 46. Resumo da aula

Nesta aula aprendemos:

``` text
KNN
│
├── K = quantidade de vizinhos
│
├── proximidade
│
├── distância
│
├── escala dos dados
│
├── StandardScaler
│
├── classificação
│
├── predict()
│
└── comparação de modelos
```

A principal ideia para guardar é:

> **O KNN classifica um novo exemplo observando os exemplos conhecidos
> que estão mais próximos dele.**

Na próxima aula, mudaremos a pergunta. Em vez de tentar prever uma
classe a partir de vizinhos, vamos investigar:

> **Como identificar registros muito diferentes do comportamento
> observado nos dados?**

Esse será o ponto de partida para **outliers e detecção de anomalias**.

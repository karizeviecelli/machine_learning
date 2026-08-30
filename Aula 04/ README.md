# Aula 04 — Como os Dados Chegam ao Machine Learning e ao Deep Learning

**Curso:** Fundamentos de Machine Learning
**Carga horária:** 4 horas
**Modalidade:** Aula teórica e prática
**Ambiente recomendado:** Google Colab

© @karizeviecelli - 2026

https://colab.research.google.com/drive/1DtIxroblfbH1WlyJrid9-L09Ml1O9c2y?usp=sharing

```text
Atividade prática

```
https://colab.research.google.com/drive/1LFEpZikJB6pJNsVNKrPHlgkO6HQ3-wT3?usp=sharing

---

# 1. Introdução

Nas aulas anteriores, iniciamos nossa jornada pelo Machine Learning.

Aprendemos que uma máquina não “pensa” da mesma forma que uma pessoa. Ela aprende identificando padrões em dados.

Também vimos que os dados não chegam prontos para os algoritmos. Antes de treinar um modelo, precisamos:

* carregar os dados;
* analisar as informações;
* corrigir problemas;
* escolher as colunas que serão utilizadas;
* separar as entradas e as respostas;
* treinar o modelo;
* avaliar os resultados.

Na Aula 03, utilizamos uma Árvore de Decisão para construir nosso primeiro modelo de Machine Learning.

Nesta aula, vamos compreender com mais profundidade **como os dados são organizados antes de entrarem em um modelo**.

Também faremos uma conexão importante com o Deep Learning e com as Redes Neurais.

O objetivo não é construir uma Rede Neural nesta aula.

Nosso objetivo é entender a base que será utilizada tanto no Machine Learning quanto no Deep Learning.

---

# 2. Pergunta inicial

Considere o seguinte código utilizado na Aula 03:

```python
modelo.fit(X_treino, y_treino)
```

O que realmente existe dentro de `X_treino`?

Como o computador enxerga esses dados?

Como uma tabela do Pandas é entregue para um algoritmo?

Como os mesmos dados poderiam ser utilizados em uma Rede Neural?

Essas perguntas serão respondidas nesta aula.

---

# 3. Objetivos da aula

Ao final desta aula, você deverá ser capaz de:

* revisar o fluxo de um projeto de Machine Learning;
* diferenciar Inteligência Artificial, Machine Learning e Deep Learning;
* compreender como os dados são representados em memória;
* diferenciar listas, arrays, matrizes e tensores;
* criar arrays utilizando NumPy;
* verificar as dimensões de um conjunto de dados;
* acessar linhas, colunas e valores de uma matriz;
* realizar operações vetorizadas;
* utilizar máscaras booleanas para filtrar dados;
* converter um DataFrame do Pandas para NumPy;
* compreender por que o Scikit-Learn utiliza estruturas NumPy;
* identificar o formato de dados utilizado em Redes Neurais.

---

# 4. O caminho percorrido até agora

Antes de avançarmos, vamos revisar o que aprendemos nas três primeiras aulas.

---

## 4.1 Aula 01 — Introdução ao Machine Learning

Na primeira aula, conhecemos os principais conceitos de Inteligência Artificial e Machine Learning.

Aprendemos que Machine Learning permite que os computadores identifiquem padrões nos dados e realizem previsões.

Alguns exemplos de aplicação são:

* prever se um aluno será aprovado;
* identificar mensagens de spam;
* recomendar filmes;
* detectar fraudes;
* prever preços;
* classificar imagens;
* reconhecer padrões de comportamento.

Também aprendemos que um modelo precisa de exemplos anteriores para aprender.

Esses exemplos formam o conjunto de dados, também chamado de dataset.

---

## 4.2 Aula 02 — Limpeza e preparação dos dados

Na segunda aula, aprendemos que dados reais podem apresentar diversos problemas.

Por exemplo:

* valores ausentes;
* registros duplicados;
* textos digitados de formas diferentes;
* números armazenados como texto;
* dados incorretos;
* colunas desnecessárias;
* valores fora do padrão.

Antes de treinar um modelo, precisamos preparar os dados.

Esse processo pode incluir:

```python
df.isnull().sum()
```

```python
df.drop_duplicates()
```

```python
df.fillna()
```

```python
df.dropna()
```

```python
df["coluna"].astype()
```

A qualidade dos dados influencia diretamente a qualidade do modelo.

Uma frase muito utilizada na área de dados é:

> Garbage in, garbage out.

Em português:

> Se os dados de entrada forem ruins, os resultados também serão ruins.

---

## 4.3 Aula 03 — Primeiro modelo de Machine Learning

Na terceira aula, construímos nosso primeiro modelo utilizando uma Árvore de Decisão.

O fluxo utilizado foi semelhante ao seguinte:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

Depois, carregamos o dataset:

```python
df = pd.read_csv("alunos.csv")
```

Selecionamos as variáveis de entrada:

```python
X = df[["idade", "nota", "frequencia"]]
```

Selecionamos a variável que desejávamos prever:

```python
y = df["aprovado"]
```

Separamos os dados de treino e teste:

```python
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Criamos o modelo:

```python
modelo = DecisionTreeClassifier(random_state=42)
```

Treinamos o modelo:

```python
modelo.fit(X_treino, y_treino)
```

Realizamos previsões:

```python
previsoes = modelo.predict(X_teste)
```

Calculamos a acurácia:

```python
acuracia = accuracy_score(y_teste, previsoes)

print(acuracia)
```

Esse processo pode ser representado pelo seguinte fluxo:

```text
Dataset
   ↓
Análise
   ↓
Limpeza
   ↓
Seleção das features
   ↓
Seleção do target
   ↓
Separação entre treino e teste
   ↓
Treinamento do modelo
   ↓
Previsão
   ↓
Avaliação
```

---

# 5. Revisando features e target

Antes de continuar, precisamos reforçar dois conceitos importantes.

---

## 5.1 Features

As features são as informações utilizadas pelo modelo para aprender.

Também podem ser chamadas de:

* características;
* atributos;
* variáveis de entrada;
* variáveis independentes;
* dados de entrada.

Considere a seguinte tabela:

| idade | nota | frequência | aprovado |
| ----: | ---: | ---------: | -------- |
|    18 |  8,5 |         95 | Sim      |
|    19 |  5,5 |         70 | Não      |
|    20 |  9,0 |         98 | Sim      |

As features podem ser:

```text
idade
nota
frequência
```

No código:

```python
X = df[["idade", "nota", "frequencia"]]
```

Por convenção, utilizamos a letra maiúscula `X` para representar as entradas.

---

## 5.2 Target

O target é a informação que queremos prever.

Também pode ser chamado de:

* alvo;
* rótulo;
* variável de saída;
* resposta;
* variável dependente.

No exemplo anterior, o target é:

```text
aprovado
```

No código:

```python
y = df["aprovado"]
```

Por convenção, utilizamos a letra minúscula `y` para representar a saída.

---

## 5.3 Analogia

Imagine que um professor precisa decidir se um aluno está em risco de reprovação.

Para tomar a decisão, ele analisa:

* as notas;
* a frequência;
* as atividades entregues;
* a participação;
* o desempenho ao longo das aulas.

Essas informações são as features.

A decisão final, como “aprovado” ou “reprovado”, é o target.

No Machine Learning, o modelo tenta aprender a relação entre as features e o target.

---

# 6. Inteligência Artificial, Machine Learning e Deep Learning

Esses termos são relacionados, mas não significam exatamente a mesma coisa.

Podemos representar essa relação da seguinte forma:

```text
Inteligência Artificial
        │
        └── Machine Learning
                │
                └── Deep Learning
```

Deep Learning faz parte do Machine Learning.

Machine Learning faz parte da Inteligência Artificial.

---

## 6.1 Inteligência Artificial

Inteligência Artificial é uma área da computação que busca criar sistemas capazes de realizar tarefas associadas à inteligência humana.

Essas tarefas podem incluir:

* tomar decisões;
* reconhecer imagens;
* compreender textos;
* conversar com pessoas;
* planejar ações;
* resolver problemas;
* identificar padrões.

Nem toda Inteligência Artificial utiliza Machine Learning.

Um sistema baseado em regras, por exemplo, também pode ser considerado uma forma de Inteligência Artificial.

Exemplo:

```python
if temperatura > 38:
    print("Possível febre")
else:
    print("Temperatura normal")
```

Nesse caso, as regras foram programadas diretamente.

O computador não aprendeu com dados.

---

## 6.2 Machine Learning

Machine Learning é uma abordagem em que o computador aprende padrões a partir de exemplos.

Em vez de programarmos todas as regras, fornecemos dados para o algoritmo.

Exemplo:

```text
idade + nota + frequência
              ↓
       modelo treinado
              ↓
      aprovação prevista
```

Na Aula 03, utilizamos uma Árvore de Decisão.

A árvore analisou os exemplos disponíveis e criou regras para classificar os alunos.

---

## 6.3 Deep Learning

Deep Learning é uma área do Machine Learning baseada principalmente em Redes Neurais com várias camadas.

O termo “deep” significa “profundo”.

A profundidade está relacionada à quantidade de camadas utilizadas para processar os dados.

Exemplo simplificado:

```text
Dados de entrada
       ↓
Primeira camada
       ↓
Segunda camada
       ↓
Terceira camada
       ↓
Resultado
```

Deep Learning é muito utilizado em problemas como:

* reconhecimento facial;
* classificação de imagens;
* reconhecimento de voz;
* geração de textos;
* tradução automática;
* carros autônomos;
* análise de vídeos;
* processamento de linguagem natural.

---

# 7. Machine Learning e Deep Learning utilizam dados

Apesar de utilizarem algoritmos diferentes, Machine Learning e Deep Learning dependem de dados.

O fluxo básico é semelhante.

## Machine Learning

```text
Dados
  ↓
Preparação
  ↓
Features e target
  ↓
Algoritmo de Machine Learning
  ↓
Modelo
  ↓
Previsão
```

## Deep Learning

```text
Dados
  ↓
Preparação
  ↓
Tensores
  ↓
Rede Neural
  ↓
Modelo
  ↓
Previsão
```

Observe que as etapas iniciais são semelhantes.

Em ambos os casos, precisamos:

* coletar dados;
* limpar os dados;
* transformar informações;
* separar entradas e saídas;
* organizar os dados numericamente;
* treinar um modelo;
* avaliar resultados.

---

# 8. O que muda entre Machine Learning e Deep Learning?

Uma das principais diferenças está no tipo de modelo utilizado.

Na Aula 03, utilizamos:

```text
Árvore de Decisão
```

Nas aulas de Redes Neurais, os alunos utilizarão modelos formados por neurônios artificiais conectados.

Outra diferença é que o Deep Learning costuma trabalhar com grandes volumes de dados e estruturas mais complexas.

Exemplos:

* imagens;
* vídeos;
* áudios;
* textos;
* grandes conjuntos de documentos.

---

## 8.1 Comparação inicial

| Machine Learning clássico                       | Deep Learning                                 |
| ----------------------------------------------- | --------------------------------------------- |
| Utiliza algoritmos como árvore, KNN e regressão | Utiliza Redes Neurais                         |
| Pode funcionar com conjuntos menores de dados   | Normalmente precisa de mais dados             |
| Pode ser mais fácil de interpretar              | Pode ser mais difícil de interpretar          |
| Geralmente utiliza menos processamento          | Pode utilizar grande capacidade computacional |
| Muito utilizado em dados tabulares              | Muito utilizado em imagens, textos e áudios   |
| Pode exigir criação manual de features          | Pode aprender representações automaticamente  |

Essa comparação é geral.

Não significa que Machine Learning não possa trabalhar com imagens ou que Deep Learning não possa trabalhar com tabelas.

A escolha depende do problema, da quantidade de dados, dos recursos disponíveis e do objetivo do projeto.

---

# 9. O elemento em comum: números

Os algoritmos não compreendem os dados da mesma forma que nós.

Para uma pessoa, estas informações possuem significado:

```text
Nome: Maria
Curso: Desenvolvimento de Sistemas
Turno: Noturno
Resultado: Aprovada
```

Para um algoritmo, esses dados precisam ser representados numericamente.

Por exemplo:

```text
Curso:
0 = Desenvolvimento de Sistemas
1 = Administração
2 = Mecânica
```

```text
Turno:
0 = Matutino
1 = Vespertino
2 = Noturno
```

```text
Resultado:
0 = Reprovado
1 = Aprovado
```

Essa transformação permite que os dados sejam utilizados nos cálculos realizados pelos modelos.

---

# 10. Como o computador enxerga uma tabela

Considere este conjunto de dados:

| idade | nota | frequência |
| ----: | ---: | ---------: |
|    18 |  8,5 |         95 |
|    19 |  6,0 |         75 |
|    20 |  9,0 |         98 |

Uma pessoa enxerga uma tabela com linhas e colunas.

O computador pode representar essa tabela como uma matriz:

```text
[
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
]
```

Cada linha representa um aluno.

Cada coluna representa uma feature.

```text
Linha 0 → primeiro aluno
Linha 1 → segundo aluno
Linha 2 → terceiro aluno
```

```text
Coluna 0 → idade
Coluna 1 → nota
Coluna 2 → frequência
```

Essa organização será fundamental para compreender NumPy, Scikit-Learn e Redes Neurais.

---

# 11. Por que aprender NumPy?

NumPy significa:

```text
Numerical Python
```

É uma biblioteca utilizada para trabalhar com dados numéricos de maneira eficiente.

Ela fornece uma estrutura chamada:

```text
ndarray
```

Essa estrutura permite trabalhar com:

* arrays;
* vetores;
* matrizes;
* dados multidimensionais;
* operações matemáticas;
* estatísticas;
* manipulação numérica.

Importação:

```python
import numpy as np
```

O apelido `np` é uma convenção utilizada pela comunidade Python.

---

## 11.1 NumPy como base do ecossistema de dados

Diversas bibliotecas utilizadas em Ciência de Dados e Inteligência Artificial são construídas sobre conceitos do NumPy.

Exemplos:

```text
Pandas
   ↓
NumPy
   ↓
Scikit-Learn
   ↓
Machine Learning
```

Também encontramos estruturas semelhantes em frameworks de Deep Learning:

```text
NumPy arrays
      ↓
Tensores
      ↓
Redes Neurais
```

Aprender NumPy ajuda a compreender como os dados são organizados e processados por essas ferramentas.

---

# 12. Lista Python

Antes de estudar arrays, vamos revisar as listas.

Uma lista pode armazenar vários valores.

```python
idades = [18, 19, 20, 21]

print(idades)
```

Resultado:

```text
[18, 19, 20, 21]
```

Podemos acessar um elemento utilizando seu índice:

```python
print(idades[0])
```

Resultado:

```text
18
```

Lembre-se de que os índices começam em zero.

```text
Valor:   18   19   20   21
Índice:   0    1    2    3
```

---

## 12.1 Lista com diferentes tipos

Uma lista Python pode armazenar dados de tipos diferentes.

```python
dados = ["Maria", 19, 8.5, True]

print(dados)
```

Essa flexibilidade é útil em diversas situações.

Entretanto, em cálculos científicos e Machine Learning, normalmente trabalhamos com grandes coleções de números.

Nessas situações, os arrays NumPy são mais apropriados.

---

# 13. Criando o primeiro array NumPy

Para criar um array, utilizamos `np.array()`.

```python
import numpy as np

idades = np.array([18, 19, 20, 21])

print(idades)
```

Resultado:

```text
[18 19 20 21]
```

Observe que a exibição é um pouco diferente de uma lista Python.

Lista:

```text
[18, 19, 20, 21]
```

Array NumPy:

```text
[18 19 20 21]
```

O array não apresenta vírgulas entre os valores.

---

# 14. Comparando lista e array

```python
lista = [10, 20, 30]

array = np.array([10, 20, 30])
```

Agora vamos tentar somar o valor `5`.

```python
array + 5
```

Resultado:

```text
[15 25 35]
```

O NumPy adicionou `5` a todos os elementos.

Com uma lista Python, o mesmo comando não funciona:

```python
lista + 5
```

Esse código gera um erro.

Para realizar a mesma operação com uma lista, precisaríamos utilizar um laço:

```python
nova_lista = []

for valor in lista:
    nova_lista.append(valor + 5)

print(nova_lista)
```

Resultado:

```text
[15, 25, 35]
```

Com NumPy, a operação é mais direta:

```python
array + 5
```

---

# 15. Analogia: lista e array

Imagine uma sala com 40 alunos.

A professora deseja aumentar todas as notas em um ponto.

Com uma lista tradicional, podemos imaginar a professora passando por cada aluno:

```text
Aluno 1 → aumentar nota
Aluno 2 → aumentar nota
Aluno 3 → aumentar nota
...
Aluno 40 → aumentar nota
```

Com NumPy, podemos imaginar uma instrução aplicada ao conjunto inteiro:

```text
Todas as notas + 1
```

Essa forma de aplicar uma operação a vários valores é chamada de operação vetorizada.

---

# 16. Primeira prática

Execute o código:

```python
import numpy as np

notas = np.array([5.5, 7.0, 8.5, 9.0])

print("Notas originais:")
print(notas)
```

Agora aumente todas as notas em um ponto:

```python
notas_atualizadas = notas + 1

print("Notas atualizadas:")
print(notas_atualizadas)
```

Resultado esperado:

```text
Notas originais:
[5.5 7.  8.5 9. ]

Notas atualizadas:
[ 6.5  8.   9.5 10. ]
```

---

# 17. Observação importante

Ao aumentar as notas, uma delas passou de 9 para 10.

Em outros exemplos, uma operação poderia produzir uma nota acima de 10.

Por isso, mesmo utilizando operações automatizadas, precisamos verificar as regras do problema.

Exemplo:

```python
notas_atualizadas = np.minimum(notas + 1, 10)

print(notas_atualizadas)
```

A função `np.minimum()` limita os valores ao máximo informado.

Nesse caso, nenhuma nota será maior do que 10.

---

# 18. Atividade de reflexão

Considere as notas:

```python
notas = np.array([4.0, 5.5, 7.0, 8.5, 9.5])
```

Responda:

1. Como aumentar todas as notas em `0.5`?
2. Como multiplicar todas as notas por `2`?
3. Como diminuir todas as notas em `1`?
4. O que acontece se alguma nota ficar acima de `10`?
5. O que acontece se alguma nota ficar abaixo de `0`?
6. Como poderíamos limitar os valores entre `0` e `10`?

Uma possível solução para limitar os valores é:

```python
notas_ajustadas = np.clip(notas + 0.5, 0, 10)

print(notas_ajustadas)
```

A função `np.clip()` limita os dados entre um valor mínimo e um valor máximo.

---

# 19. Conexão com Machine Learning

Na Aula 03, trabalhamos com um DataFrame:

```python
X = df[["idade", "nota", "frequencia"]]
```

Apesar de visualizarmos uma tabela, o modelo precisa realizar cálculos sobre esses dados.

Internamente, bibliotecas como Scikit-Learn trabalham com estruturas numéricas semelhantes aos arrays NumPy.

Por isso, o seguinte código funciona:

```python
modelo.fit(X_treino, y_treino)
```

O Scikit-Learn recebe as informações, valida os dados e realiza os cálculos necessários para treinar o modelo.

Mais adiante, converteremos explicitamente o DataFrame para NumPy:

```python
X_numpy = X.to_numpy()
```

Depois, poderemos verificar o tipo:

```python
print(type(X))
print(type(X_numpy))
```

Resultados esperados:

```text
<class 'pandas.core.frame.DataFrame'>
<class 'numpy.ndarray'>
```

---

# 20. Conexão com Deep Learning

Em Redes Neurais, os dados também precisam ser organizados numericamente.

Em vez de utilizar apenas tabelas bidimensionais, podemos trabalhar com estruturas que possuem mais dimensões.

Exemplos:

```text
Lista de notas
1 dimensão
```

```text
Tabela de alunos
2 dimensões
```

```text
Imagem colorida
3 dimensões
```

```text
Conjunto de imagens coloridas
4 dimensões
```

Essas estruturas multidimensionais são frequentemente chamadas de tensores no contexto do Deep Learning.

Podemos pensar nos tensores como uma extensão dos arrays e das matrizes.

```text
Número → escalar
Lista de números → vetor
Tabela → matriz
Estrutura com várias dimensões → tensor
```

Essa relação será aprofundada ao longo da aula.

---

# 21. Resumo da Parte 1

Nesta primeira parte, revisamos:

* o fluxo de um projeto de Machine Learning;
* o significado de features e target;
* a relação entre Inteligência Artificial, Machine Learning e Deep Learning;
* a necessidade de representar dados numericamente;
* a forma como uma tabela pode ser representada como matriz;
* o papel do NumPy;
* a diferença inicial entre listas e arrays;
* o conceito de operação vetorizada;
* a conexão entre NumPy, Scikit-Learn e Redes Neurais.

Na próxima etapa da aula, estudaremos as propriedades de um array NumPy:

* `dtype`;
* `shape`;
* `ndim`;
* `size`;
* criação de arrays;
* tipos numéricos;
* vetores;
* matrizes.

© @karizeviecelli - 2026


# Aula 04 — Como os Dados Chegam ao Machine Learning e ao Deep Learning

## Parte 2 — Arrays, dimensões e matrizes com NumPy

© @karizeviecelli - 2026

---

# 22. Conhecendo um array NumPy

Na Parte 1, criamos nosso primeiro array:

```python
import numpy as np

idades = np.array([18, 19, 20, 21])

print(idades)
```

Resultado:

```text
[18 19 20 21]
```

Agora vamos investigar as características desse array.

O NumPy oferece propriedades que permitem descobrir:

* o tipo dos dados;
* o formato da estrutura;
* a quantidade de dimensões;
* a quantidade total de elementos.

As principais propriedades são:

```text
dtype
shape
ndim
size
```

Essas informações são muito importantes em Machine Learning e Deep Learning.

Quando um modelo apresenta erro de formato, normalmente uma das primeiras verificações deve ser:

```python
print(X.shape)
```

---

# 23. A propriedade `dtype`

A propriedade `dtype` informa o tipo dos dados armazenados no array.

Exemplo:

```python
import numpy as np

idades = np.array([18, 19, 20, 21])

print(idades.dtype)
```

Um possível resultado será:

```text
int64
```

A palavra `int` indica que os valores são números inteiros.

O número `64` indica a quantidade de bits utilizada para representar cada valor.

Dependendo do sistema e do ambiente utilizado, o resultado também pode aparecer como:

```text
int32
```

Isso não representa um erro.

---

## 23.1 Array com números decimais

```python
notas = np.array([7.5, 8.0, 9.2, 6.5])

print(notas.dtype)
```

Resultado provável:

```text
float64
```

A palavra `float` representa números com casas decimais.

---

## 23.2 Misturando inteiros e decimais

Observe o exemplo:

```python
valores = np.array([10, 20, 30.5, 40])

print(valores)
print(valores.dtype)
```

Resultado:

```text
[10.  20.  30.5 40. ]
float64
```

Mesmo que alguns valores tenham sido escritos como inteiros, o NumPy converteu todos para o mesmo tipo.

Isso acontece porque um array NumPy normalmente trabalha com valores do mesmo tipo.

Como existe um valor decimal, o NumPy converte os inteiros para `float`.

---

## 23.3 Por que o NumPy utiliza um único tipo?

Uma lista Python pode armazenar diferentes tipos:

```python
dados = ["Maria", 19, 8.5, True]
```

Um array NumPy é otimizado para cálculos numéricos.

Quando os valores possuem o mesmo tipo, o computador consegue:

* organizar melhor a memória;
* realizar cálculos mais rapidamente;
* aplicar operações em todo o conjunto;
* utilizar menos recursos em determinadas situações.

Essa organização ajuda a explicar por que o NumPy é tão utilizado em Ciência de Dados.

---

# 24. Definindo o tipo durante a criação

Podemos informar o tipo desejado utilizando o parâmetro `dtype`.

```python
idades = np.array(
    [18, 19, 20, 21],
    dtype=float
)

print(idades)
print(idades.dtype)
```

Resultado:

```text
[18. 19. 20. 21.]
float64
```

Embora os valores tenham sido informados como inteiros, o array foi criado com tipo decimal.

---

## 24.1 Criando um array de inteiros

```python
notas = np.array(
    [7.5, 8.9, 9.2],
    dtype=int
)

print(notas)
```

Resultado:

```text
[7 8 9]
```

Atenção: ao converter números decimais para inteiros, a parte decimal é removida.

O NumPy não realiza arredondamento automaticamente nessa conversão.

Por exemplo:

```text
7.5 → 7
8.9 → 8
9.2 → 9
```

---

## 24.2 Arredondando antes da conversão

Para arredondar os valores, podemos utilizar:

```python
notas = np.array([7.5, 8.9, 9.2])

notas_arredondadas = np.round(notas)

print(notas_arredondadas)
```

Resultado:

```text
[8. 9. 9.]
```

Depois, podemos converter para inteiro:

```python
notas_inteiras = np.round(notas).astype(int)

print(notas_inteiras)
```

Resultado:

```text
[8 9 9]
```

---

# 25. Conversão com `astype()`

O método `astype()` permite converter o tipo de um array.

```python
notas = np.array([7.5, 8.0, 9.2])

notas_inteiras = notas.astype(int)

print(notas_inteiras)
print(notas_inteiras.dtype)
```

Resultado:

```text
[7 8 9]
int64
```

O array original não é alterado automaticamente.

```python
print(notas)
```

Resultado:

```text
[7.5 8.  9.2]
```

Foi criado um novo array com o tipo convertido.

---

# 26. Atenção com textos em arrays

Observe:

```python
dados = np.array([18, 19, "20", 21])

print(dados)
print(dados.dtype)
```

Como existe um texto no conjunto, o NumPy pode converter todos os valores para texto.

Resultado semelhante:

```text
['18' '19' '20' '21']
<U21
```

Nesse caso, operações matemáticas podem não funcionar como esperado.

Por exemplo:

```python
dados + 1
```

Esse código gera erro porque os valores estão armazenados como texto.

---

## 26.1 Corrigindo o tipo

Se os textos representam números válidos, podemos converter:

```python
dados = np.array(["18", "19", "20", "21"])

dados_numericos = dados.astype(int)

print(dados_numericos)
print(dados_numericos.dtype)
```

Resultado:

```text
[18 19 20 21]
int64
```

---

## 26.2 Erro de conversão

Observe:

```python
dados = np.array(["18", "dezenove", "20"])
```

Ao executar:

```python
dados.astype(int)
```

será gerado um erro porque a palavra `"dezenove"` não pode ser convertida diretamente para número inteiro.

Esse é um exemplo de problema de qualidade dos dados.

Antes de converter, precisamos identificar e corrigir valores inválidos.

Essa situação se conecta diretamente à Aula 02.

---

# 27. A propriedade `ndim`

A propriedade `ndim` informa a quantidade de dimensões de um array.

```python
idades = np.array([18, 19, 20, 21])

print(idades.ndim)
```

Resultado:

```text
1
```

Esse array possui uma dimensão.

Podemos imaginá-lo como uma linha de valores:

```text
[18, 19, 20, 21]
```

---

## 27.1 O que é uma dimensão?

Uma dimensão indica quantos índices são necessários para localizar um valor.

Em um array de uma dimensão:

```python
idades[0]
```

Precisamos de apenas um índice.

Em uma matriz de duas dimensões:

```python
dados[0, 1]
```

Precisamos de dois índices:

* índice da linha;
* índice da coluna.

---

# 28. Escalares, vetores, matrizes e tensores

Antes de avançarmos, vamos organizar os conceitos.

## 28.1 Escalar

Um escalar é um único valor.

```python
idade = 18
```

Exemplo:

```text
18
```

Ele não representa uma coleção de valores.

---

## 28.2 Vetor

Um vetor é uma sequência de valores.

```python
idades = np.array([18, 19, 20, 21])
```

Representação:

```text
[18, 19, 20, 21]
```

Esse array possui uma dimensão.

```python
print(idades.ndim)
```

Resultado:

```text
1
```

---

## 28.3 Matriz

Uma matriz organiza os dados em linhas e colunas.

```python
dados = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
])
```

Representação:

```text
[
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
]
```

Esse array possui duas dimensões.

```python
print(dados.ndim)
```

Resultado:

```text
2
```

---

## 28.4 Tensor

Um tensor é uma estrutura que pode possuir três ou mais dimensões.

Exemplo simplificado:

```python
tensor = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])
```

Verificando:

```python
print(tensor.ndim)
```

Resultado:

```text
3
```

No contexto matemático, escalares, vetores e matrizes também podem ser considerados tensores de diferentes ordens.

Entretanto, em uma introdução didática, podemos utilizar:

```text
0 dimensão → escalar
1 dimensão → vetor
2 dimensões → matriz
3 ou mais dimensões → tensor multidimensional
```

---

# 29. A propriedade `shape`

A propriedade `shape` informa o formato do array.

```python
idades = np.array([18, 19, 20, 21])

print(idades.shape)
```

Resultado:

```text
(4,)
```

Isso significa que o array possui quatro elementos em uma dimensão.

A vírgula indica que existe apenas um eixo.

---

## 29.1 Formato de uma matriz

```python
dados = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
])

print(dados.shape)
```

Resultado:

```text
(3, 3)
```

Isso significa:

```text
3 linhas
3 colunas
```

Em Machine Learning:

```text
3 amostras
3 features
```

---

# 30. Linhas como amostras e colunas como features

Considere:

```python
X = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85]
])
```

A estrutura possui:

```python
print(X.shape)
```

Resultado:

```text
(4, 3)
```

Interpretação:

```text
4 linhas → 4 alunos
3 colunas → 3 features
```

As features são:

```text
Coluna 0 → idade
Coluna 1 → nota
Coluna 2 → frequência
```

---

## 30.1 Analogia com uma ficha de alunos

Imagine quatro fichas.

Cada ficha contém:

```text
idade
nota
frequência
```

Ao organizarmos as fichas em uma tabela, cada ficha forma uma linha.

```text
Aluno 1 → [18, 8.5, 95]
Aluno 2 → [19, 6.0, 75]
Aluno 3 → [20, 9.0, 98]
Aluno 4 → [21, 7.5, 85]
```

A matriz reúne todas as fichas em uma única estrutura.

---

# 31. A propriedade `size`

A propriedade `size` informa a quantidade total de elementos.

```python
idades = np.array([18, 19, 20, 21])

print(idades.size)
```

Resultado:

```text
4
```

---

## 31.1 `size` em uma matriz

```python
dados = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
])

print(dados.size)
```

Resultado:

```text
9
```

A matriz possui:

```text
3 linhas × 3 colunas = 9 elementos
```

---

## 31.2 Comparando `shape` e `size`

```python
print(dados.shape)
print(dados.size)
```

Resultado:

```text
(3, 3)
9
```

`shape` mostra o formato.

`size` mostra a quantidade total de elementos.

---

# 32. Resumo das propriedades

Considere:

```python
dados = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
])
```

Execute:

```python
print("Tipo dos dados:", dados.dtype)
print("Formato:", dados.shape)
print("Dimensões:", dados.ndim)
print("Quantidade de elementos:", dados.size)
```

Possível resultado:

```text
Tipo dos dados: float64
Formato: (3, 3)
Dimensões: 2
Quantidade de elementos: 9
```

---

## 32.1 Por que o tipo é `float64`?

O array contém números inteiros e decimais:

```text
18
8.5
95
```

Como existe pelo menos um número decimal, o NumPy converte todos os valores para `float`.

Por isso, a exibição pode aparecer assim:

```text
[[18.   8.5 95. ]
 [19.   6.  75. ]
 [20.   9.  98. ]]
```

Os valores continuam representando as mesmas informações.

---

# 33. Prática guiada: descrevendo um array

Execute:

```python
import numpy as np

temperaturas = np.array([
    [20.5, 22.0, 24.5],
    [19.0, 21.5, 23.0]
])

print(temperaturas)
```

Agora verifique:

```python
print("dtype:", temperaturas.dtype)
print("shape:", temperaturas.shape)
print("ndim:", temperaturas.ndim)
print("size:", temperaturas.size)
```

Resultados esperados:

```text
dtype: float64
shape: (2, 3)
ndim: 2
size: 6
```

Interpretação:

```text
2 linhas
3 colunas
2 dimensões
6 valores no total
```

---

# 34. Criando arrays com funções do NumPy

Nem sempre precisamos escrever todos os valores manualmente.

O NumPy oferece funções para criar arrays automaticamente.

---

## 34.1 `np.zeros()`

Cria um array preenchido com zeros.

```python
valores = np.zeros(5)

print(valores)
```

Resultado:

```text
[0. 0. 0. 0. 0.]
```

---

## 34.2 Matriz de zeros

```python
matriz = np.zeros((3, 4))

print(matriz)
```

Resultado:

```text
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

Formato:

```python
print(matriz.shape)
```

Resultado:

```text
(3, 4)
```

---

## 34.3 Aplicação prática

Uma matriz de zeros pode ser utilizada para:

* inicializar valores;
* reservar uma estrutura para resultados;
* representar ausência inicial de dados;
* criar imagens completamente pretas;
* iniciar contadores;
* simular entradas.

---

# 35. `np.ones()`

Cria um array preenchido com o valor `1`.

```python
valores = np.ones(5)

print(valores)
```

Resultado:

```text
[1. 1. 1. 1. 1.]
```

---

## 35.1 Matriz de uns

```python
matriz = np.ones((2, 3))

print(matriz)
```

Resultado:

```text
[[1. 1. 1.]
 [1. 1. 1.]]
```

---

# 36. `np.full()`

Cria um array preenchido com um valor escolhido.

```python
valores = np.full(5, 7)

print(valores)
```

Resultado:

```text
[7 7 7 7 7]
```

---

## 36.1 Matriz preenchida

```python
notas_iniciais = np.full((3, 4), 10)

print(notas_iniciais)
```

Resultado:

```text
[[10 10 10 10]
 [10 10 10 10]
 [10 10 10 10]]
```

---

# 37. `np.arange()`

A função `np.arange()` cria uma sequência numérica.

```python
numeros = np.arange(5)

print(numeros)
```

Resultado:

```text
[0 1 2 3 4]
```

O valor final não é incluído.

---

## 37.1 Definindo início e fim

```python
numeros = np.arange(1, 6)

print(numeros)
```

Resultado:

```text
[1 2 3 4 5]
```

---

## 37.2 Definindo o intervalo

```python
numeros = np.arange(0, 11, 2)

print(numeros)
```

Resultado:

```text
[ 0  2  4  6  8 10]
```

Os parâmetros representam:

```text
início
fim
passo
```

---

## 37.3 Aplicação em Machine Learning

Podemos utilizar `np.arange()` para:

* criar índices;
* gerar valores para testes;
* representar épocas;
* criar sequências de parâmetros;
* gerar dados simulados.

---

# 38. `np.linspace()`

A função `np.linspace()` cria valores igualmente espaçados entre dois limites.

```python
valores = np.linspace(0, 1, 5)

print(valores)
```

Resultado:

```text
[0.   0.25 0.5  0.75 1.  ]
```

Nesse caso, foram criados cinco valores entre zero e um.

Ao contrário de `np.arange()`, o valor final normalmente é incluído.

---

## 38.1 Aplicação

```python
temperaturas = np.linspace(20, 30, 6)

print(temperaturas)
```

Resultado:

```text
[20. 22. 24. 26. 28. 30.]
```

---

# 39. Criando dados aleatórios

O NumPy pode gerar valores aleatórios.

Esses recursos são úteis para:

* criar exemplos;
* simular datasets;
* testar algoritmos;
* inicializar valores;
* dividir ou embaralhar dados.

---

## 39.1 Números aleatórios entre zero e um

```python
valores = np.random.rand(5)

print(valores)
```

Cada execução pode gerar valores diferentes.

Exemplo:

```text
[0.12 0.81 0.43 0.66 0.25]
```

---

## 39.2 Matriz aleatória

```python
matriz = np.random.rand(3, 2)

print(matriz)
```

Formato:

```text
3 linhas
2 colunas
```

---

## 39.3 Números inteiros aleatórios

```python
idades = np.random.randint(18, 60, size=10)

print(idades)
```

Esse código cria dez idades entre 18 e 59.

O limite final não é incluído.

---

## 39.4 Criando notas aleatórias

```python
notas = np.random.uniform(0, 10, size=10)

print(notas)
```

Essa função cria dez números decimais entre zero e dez.

---

# 40. Reprodutibilidade com semente aleatória

Ao gerar números aleatórios, os valores normalmente mudam em cada execução.

Para obter sempre os mesmos valores, podemos definir uma semente:

```python
np.random.seed(42)

valores = np.random.randint(0, 10, size=5)

print(valores)
```

Sempre que o código for executado com a mesma semente, o resultado será igual.

---

## 40.1 Conexão com a Aula 03

Na Aula 03, utilizamos:

```python
random_state=42
```

Exemplo:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

O objetivo é semelhante.

O valor fixa o processo aleatório para que a divisão seja reproduzível.

Assim, alunos e professor obtêm a mesma separação entre treino e teste.

---

# 41. Construindo uma matriz de alunos

Vamos criar um pequeno conjunto de dados.

```python
alunos = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85],
    [22, 5.0, 60]
])

print(alunos)
```

As colunas representam:

```text
idade
nota
frequência
```

---

## 41.1 Investigando a matriz

```python
print("Tipo:", alunos.dtype)
print("Formato:", alunos.shape)
print("Dimensões:", alunos.ndim)
print("Elementos:", alunos.size)
```

Resultado esperado:

```text
Tipo: float64
Formato: (5, 3)
Dimensões: 2
Elementos: 15
```

Interpretação:

```text
5 alunos
3 features
15 valores no total
```

---

# 42. Criando o target

As entradas são armazenadas em `X`.

```python
X = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85],
    [22, 5.0, 60]
])
```

Agora criamos o target:

```python
y = np.array([1, 0, 1, 1, 0])
```

Significado:

```text
1 → aprovado
0 → reprovado
```

---

## 42.1 Verificando os formatos

```python
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
```

Resultado:

```text
Formato de X: (5, 3)
Formato de y: (5,)
```

`X` possui duas dimensões porque contém linhas e colunas.

`y` possui uma dimensão porque contém uma resposta para cada aluno.

---

# 43. Quantidade de amostras em `X` e `y`

O número de linhas de `X` deve corresponder à quantidade de respostas de `y`.

```text
X → 5 alunos
y → 5 respostas
```

Por isso:

```python
print(X.shape[0])
print(y.shape[0])
```

Ambos devem apresentar:

```text
5
```

---

## 43.1 Exemplo incorreto

```python
X = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
])

y = np.array([1, 0])
```

Temos:

```text
3 alunos
2 respostas
```

Ao tentar treinar um modelo, será gerado um erro.

O modelo não sabe qual é a resposta do terceiro aluno.

---

# 44. Erro de quantidade inconsistente

Considere:

```python
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier()

modelo.fit(X, y)
```

Se `X` e `y` possuírem quantidades diferentes, pode aparecer uma mensagem semelhante a:

```text
ValueError: Found input variables with inconsistent numbers of samples
```

Tradução:

```text
Foram encontradas variáveis de entrada com quantidades inconsistentes de amostras.
```

---

## 44.1 Como investigar

Execute:

```python
print("Linhas de X:", X.shape[0])
print("Elementos de y:", y.shape[0])
```

Os valores precisam ser iguais.

---

# 45. Treinando a Árvore de Decisão com NumPy

Vamos utilizar arrays NumPy diretamente.

```python
import numpy as np

from sklearn.tree import DecisionTreeClassifier
```

Dados de entrada:

```python
X = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85],
    [22, 5.0, 60],
    [23, 6.5, 78],
    [24, 8.0, 90],
    [25, 4.5, 55]
])
```

Target:

```python
y = np.array([
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    0
])
```

Criando o modelo:

```python
modelo = DecisionTreeClassifier(
    random_state=42
)
```

Treinando:

```python
modelo.fit(X, y)
```

---

## 45.1 Fazendo uma previsão

```python
novo_aluno = np.array([
    [20, 8.0, 92]
])

resultado = modelo.predict(novo_aluno)

print(resultado)
```

Resultado possível:

```text
[1]
```

Interpretação:

```text
1 → aprovado
```

---

# 46. Por que o novo aluno precisa de duas dimensões?

Observe:

```python
novo_aluno = np.array([
    [20, 8.0, 92]
])
```

O formato será:

```python
print(novo_aluno.shape)
```

Resultado:

```text
(1, 3)
```

Interpretação:

```text
1 aluno
3 features
```

---

## 46.1 Exemplo com uma dimensão

```python
novo_aluno = np.array([20, 8.0, 92])

print(novo_aluno.shape)
```

Resultado:

```text
(3,)
```

Essa estrutura representa apenas uma sequência de três valores.

O modelo espera uma tabela com:

```text
linhas → amostras
colunas → features
```

Por isso, mesmo para um único aluno, normalmente utilizamos:

```python
[[20, 8.0, 92]]
```

---

# 47. Erro esperado com array unidimensional

Se utilizarmos:

```python
novo_aluno = np.array([20, 8.0, 92])

modelo.predict(novo_aluno)
```

pode aparecer uma mensagem semelhante:

```text
Expected 2D array, got 1D array instead
```

Tradução:

```text
Era esperado um array de duas dimensões, mas foi recebido um array de uma dimensão.
```

---

## 47.1 Solução 1: colchetes adicionais

```python
novo_aluno = np.array([
    [20, 8.0, 92]
])
```

---

## 47.2 Solução 2: `reshape()`

```python
novo_aluno = np.array([20, 8.0, 92])

novo_aluno = novo_aluno.reshape(1, -1)

print(novo_aluno.shape)
```

Resultado:

```text
(1, 3)
```

O `reshape()` será aprofundado posteriormente.

---

# 48. Ordem das features

Na matriz de treinamento, definimos:

```text
Coluna 0 → idade
Coluna 1 → nota
Coluna 2 → frequência
```

Portanto, o novo aluno precisa seguir a mesma ordem:

```python
novo_aluno = np.array([
    [20, 8.0, 92]
])
```

Isso significa:

```text
idade = 20
nota = 8.0
frequência = 92
```

---

## 48.1 Ordem incorreta

Considere:

```python
novo_aluno = np.array([
    [92, 20, 8.0]
])
```

O modelo interpretará:

```text
idade = 92
nota = 20
frequência = 8
```

O código pode executar sem erro, mas os dados estarão errados.

Esse é um problema perigoso porque o programa funciona, mas a previsão perde o sentido.

---

# 49. Nomes das colunas e arrays NumPy

Quando utilizamos um DataFrame do Pandas:

```python
X = df[["idade", "nota", "frequencia"]]
```

os nomes das colunas permanecem associados aos dados.

Quando convertemos para NumPy:

```python
X_numpy = X.to_numpy()
```

obtemos apenas os valores.

Exemplo:

```text
[[18.   8.5 95. ]
 [19.   6.  75. ]
 [20.   9.  98. ]]
```

Os nomes das colunas não fazem parte do array.

Por isso, precisamos registrar e respeitar a ordem das features.

---

# 50. Verificando nomes antes da conversão

```python
print(X.columns)
```

Resultado:

```text
Index(['idade', 'nota', 'frequencia'], dtype='object')
```

Podemos salvar os nomes:

```python
nomes_features = X.columns.to_list()

print(nomes_features)
```

Resultado:

```text
['idade', 'nota', 'frequencia']
```

Depois, podemos converter:

```python
X_numpy = X.to_numpy()
```

---

# 51. DataFrame e ndarray

Considere:

```python
import pandas as pd

dados = {
    "idade": [18, 19, 20],
    "nota": [8.5, 6.0, 9.0],
    "frequencia": [95, 75, 98]
}

df = pd.DataFrame(dados)
```

Verificando:

```python
print(type(df))
```

Resultado:

```text
<class 'pandas.core.frame.DataFrame'>
```

Convertendo:

```python
array_dados = df.to_numpy()
```

Verificando:

```python
print(type(array_dados))
```

Resultado:

```text
<class 'numpy.ndarray'>
```

---

## 51.1 Comparação

| Característica                  | DataFrame Pandas | Array NumPy                       |
| ------------------------------- | ---------------- | --------------------------------- |
| Possui nomes de colunas         | Sim              | Não                               |
| Possui índices de linhas        | Sim              | Não da mesma forma                |
| Pode misturar tipos por coluna  | Sim              | Normalmente utiliza um único tipo |
| É adequado para análise tabular | Sim              | Sim, principalmente para cálculos |
| É muito usado no Scikit-Learn   | Sim              | Sim                               |
| Possui operações vetorizadas    | Sim              | Sim                               |
| É base para cálculos numéricos  | Indiretamente    | Sim                               |

---

# 52. O que o Scikit-Learn recebe?

O Scikit-Learn aceita várias estruturas compatíveis.

Por exemplo:

```python
modelo.fit(X_dataframe, y_series)
```

Também aceita:

```python
modelo.fit(X_numpy, y_numpy)
```

O importante é que os dados tenham:

* quantidade correta de amostras;
* quantidade correta de features;
* tipos compatíveis;
* ausência de valores inválidos;
* formato esperado pelo algoritmo.

---

# 53. Prática completa com Pandas e NumPy

```python
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
```

Criando o DataFrame:

```python
dados = {
    "idade": [18, 19, 20, 21, 22, 23],
    "nota": [8.5, 6.0, 9.0, 7.5, 5.0, 6.5],
    "frequencia": [95, 75, 98, 85, 60, 78],
    "aprovado": [1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(dados)

display(df)
```

Selecionando as features:

```python
X_dataframe = df[
    ["idade", "nota", "frequencia"]
]
```

Selecionando o target:

```python
y_series = df["aprovado"]
```

Verificando os tipos:

```python
print(type(X_dataframe))
print(type(y_series))
```

Convertendo:

```python
X_numpy = X_dataframe.to_numpy()
y_numpy = y_series.to_numpy()
```

Verificando novamente:

```python
print(type(X_numpy))
print(type(y_numpy))
```

---

## 53.1 Verificando os formatos

```python
print("X DataFrame:", X_dataframe.shape)
print("X NumPy:", X_numpy.shape)

print("y Series:", y_series.shape)
print("y NumPy:", y_numpy.shape)
```

Os formatos devem permanecer equivalentes.

---

## 53.2 Treinando com NumPy

```python
modelo = DecisionTreeClassifier(
    random_state=42
)

modelo.fit(X_numpy, y_numpy)
```

Previsão:

```python
novo_aluno = np.array([
    [20, 8.2, 90]
])

resultado = modelo.predict(novo_aluno)

print(resultado)
```

---

# 54. Atenção aos avisos de nomes de features

Quando o modelo é treinado com DataFrame:

```python
modelo.fit(X_dataframe, y_series)
```

o Scikit-Learn pode registrar os nomes das colunas.

Se depois realizarmos a previsão com um array NumPy:

```python
novo_aluno = np.array([
    [20, 8.2, 90]
])

modelo.predict(novo_aluno)
```

pode aparecer um aviso semelhante:

```text
X does not have valid feature names
```

Isso acontece porque o modelo foi treinado com nomes de colunas, mas recebeu um array sem nomes.

---

## 54.1 Solução recomendada

Se o modelo foi treinado com DataFrame, faça a previsão utilizando outro DataFrame com as mesmas colunas.

```python
novo_aluno = pd.DataFrame({
    "idade": [20],
    "nota": [8.2],
    "frequencia": [90]
})

resultado = modelo.predict(novo_aluno)

print(resultado)
```

---

## 54.2 Outra possibilidade

Treinar e prever sempre utilizando arrays NumPy:

```python
modelo.fit(X_numpy, y_numpy)

novo_aluno = np.array([
    [20, 8.2, 90]
])

modelo.predict(novo_aluno)
```

O importante é manter consistência.

---

# 55. Boas práticas de consistência

Escolha uma forma de representar os dados e mantenha-a durante o fluxo.

## Opção A — Pandas

```text
Treinamento → DataFrame
Previsão → DataFrame
```

## Opção B — NumPy

```text
Treinamento → ndarray
Previsão → ndarray
```

As duas formas podem funcionar.

Para iniciantes, o DataFrame é útil porque preserva os nomes das colunas.

Para cálculos e entendimento das estruturas internas, o NumPy é fundamental.

---

# 56. Atividade prática 1 — Identificando propriedades

Crie o array:

```python
dados = np.array([
    [17, 7.5, 90],
    [18, 8.0, 95],
    [19, 6.5, 70],
    [20, 9.0, 98]
])
```

Responda utilizando código:

1. Qual é o tipo dos dados?
2. Qual é o formato do array?
3. Quantas dimensões ele possui?
4. Quantos elementos existem no total?
5. Quantos alunos estão representados?
6. Quantas features existem?

Comandos esperados:

```python
print(dados.dtype)
print(dados.shape)
print(dados.ndim)
print(dados.size)
```

---

# 57. Atividade prática 2 — Criando `X` e `y`

Crie uma matriz com os seguintes alunos:

| idade | nota | frequência |
| ----: | ---: | ---------: |
|    18 |  8,0 |         90 |
|    19 |  5,5 |         65 |
|    20 |  9,0 |         95 |
|    21 |  6,5 |         75 |
|    22 |  7,5 |         85 |

Crie o target:

```text
Aprovado
Reprovado
Aprovado
Reprovado
Aprovado
```

Utilize:

```text
1 → aprovado
0 → reprovado
```

Depois:

1. verifique `X.shape`;
2. verifique `y.shape`;
3. confirme se as quantidades de amostras são iguais;
4. treine uma Árvore de Decisão;
5. faça uma previsão para um novo aluno.

---

# 58. Atividade prática 3 — Encontrando o erro

Analise:

```python
X = np.array([
    [18, 8.0, 90],
    [19, 5.5, 65],
    [20, 9.0, 95]
])

y = np.array([1, 0])
```

Responda:

1. Qual é o problema?
2. Qual é o formato de `X`?
3. Qual é o formato de `y`?
4. Como corrigir?
5. Qual erro pode aparecer no treinamento?

---

# 59. Atividade prática 4 — Novo aluno

Analise:

```python
novo_aluno = np.array([20, 8.5, 92])
```

Responda:

1. Qual é o `shape`?
2. Quantas dimensões possui?
3. Por que o modelo pode recusar essa estrutura?
4. Como transformar em uma matriz com uma linha?
5. Como fazer isso utilizando colchetes?
6. Como fazer isso utilizando `reshape()`?

---

# 60. Desafio rápido — Dataset simulado

Crie um dataset com 20 alunos utilizando NumPy.

As regras são:

* idade entre 16 e 50;
* nota entre 0 e 10;
* frequência entre 50 e 100;
* target igual a `1` quando:

  * nota for maior ou igual a 7;
  * frequência for maior ou igual a 75;
* target igual a `0` nos demais casos.

Comece definindo a semente:

```python
np.random.seed(42)
```

Crie as idades:

```python
idades = np.random.randint(
    16,
    51,
    size=20
)
```

Crie as notas:

```python
notas = np.random.uniform(
    0,
    10,
    size=20
)
```

Crie as frequências:

```python
frequencias = np.random.randint(
    50,
    101,
    size=20
)
```

Neste momento, ainda não estudamos profundamente a junção de colunas.

Uma possibilidade é:

```python
X = np.column_stack(
    (
        idades,
        notas,
        frequencias
    )
)
```

Crie o target:

```python
y = np.where(
    (notas >= 7) & (frequencias >= 75),
    1,
    0
)
```

Verifique:

```python
print(X)
print(y)

print(X.shape)
print(y.shape)
```

---

# 61. Reflexão sobre o desafio

Responda:

1. Quantos alunos foram criados?
2. Quantas features existem?
3. Qual é o tipo de `X`?
4. Qual é o tipo de `y`?
5. Qual é o formato de cada estrutura?
6. Quantos alunos foram classificados como aprovados?
7. Essa regra representa um modelo de Machine Learning?
8. Ou foi uma regra programada diretamente?
9. Qual é a diferença entre essa regra e uma Árvore de Decisão treinada?

---

# 62. Regra programada versus modelo treinado

No desafio, escrevemos diretamente a regra:

```python
(notas >= 7) & (frequencias >= 75)
```

Isso significa que o computador não descobriu a regra.

Nós definimos a condição.

Trata-se de uma regra programada.

Em Machine Learning, fornecemos exemplos:

```text
features + respostas conhecidas
```

e o modelo tenta encontrar padrões.

Essa diferença é essencial.

---

# 63. Conexão com Redes Neurais

Uma Rede Neural também recebe estruturas organizadas numericamente.

Para dados tabulares, podemos ter:

```text
X.shape = (20, 3)
```

Interpretação:

```text
20 alunos
3 features
```

Uma Rede Neural poderia receber essa mesma matriz.

O algoritmo seria diferente, mas a organização dos dados continuaria semelhante.

---

## 63.1 Entrada de uma Rede Neural

Considere:

```text
idade
nota
frequência
```

Essas três features podem formar três entradas.

Representação simplificada:

```text
idade --------\
               \
nota -----------> Rede Neural → resultado
               /
frequência ----/
```

Cada linha de `X` representa um exemplo apresentado à rede.

---

# 64. O significado de lote

Em Deep Learning, é comum trabalhar com grupos de exemplos chamados de lotes ou `batches`.

Considere:

```text
X.shape = (32, 3)
```

Pode significar:

```text
32 alunos em um lote
3 features para cada aluno
```

A Rede Neural processa vários exemplos durante o treinamento.

O conceito será aprofundado na disciplina de Redes Neurais.

Nesta aula, o importante é compreender que a primeira dimensão frequentemente representa a quantidade de exemplos.

---

# 65. Checklist de compreensão da Parte 2

Marque os itens que você já consegue explicar:

* [ ] Sei o que significa `dtype`.
* [ ] Sei o que significa `shape`.
* [ ] Sei o que significa `ndim`.
* [ ] Sei o que significa `size`.
* [ ] Consigo diferenciar vetor e matriz.
* [ ] Consigo interpretar um formato como `(5, 3)`.
* [ ] Sei que as linhas representam amostras.
* [ ] Sei que as colunas representam features.
* [ ] Sei por que `X` e `y` devem ter a mesma quantidade de amostras.
* [ ] Sei por que um novo aluno deve ser enviado como array 2D.
* [ ] Sei converter um DataFrame para NumPy.
* [ ] Sei que arrays NumPy não preservam os nomes das colunas.
* [ ] Entendo por que a ordem das features precisa ser mantida.
* [ ] Sei que Machine Learning e Redes Neurais utilizam estruturas numéricas.

---

# 66. Resumo da Parte 2

Nesta parte, estudamos:

* tipos de dados com `dtype`;
* conversões com `astype()`;
* escalares, vetores, matrizes e tensores;
* dimensões com `ndim`;
* formatos com `shape`;
* quantidade de valores com `size`;
* criação de arrays com:

  * `np.zeros()`;
  * `np.ones()`;
  * `np.full()`;
  * `np.arange()`;
  * `np.linspace()`;
* criação de dados aleatórios;
* reprodutibilidade com sementes;
* construção de `X` e `y`;
* compatibilidade entre a quantidade de amostras;
* treinamento da Árvore de Decisão com arrays;
* formato de um novo registro;
* conversão entre Pandas e NumPy;
* consistência entre treinamento e previsão;
* preparação conceitual para Redes Neurais.

---

# 67. O que veremos na Parte 3

Na próxima parte, aprenderemos a acessar e modificar os dados dentro dos arrays.

Serão estudados:

* indexação;
* índices positivos;
* índices negativos;
* acesso a linhas;
* acesso a colunas;
* acesso a células;
* slicing;
* seleção de intervalos;
* filtros;
* máscaras booleanas;
* comparação entre valores;
* combinação de condições.

Esses recursos permitirão investigar e preparar os dados antes de entregá-los a um modelo.

© @karizeviecelli - 2026

----
# Aula 04 — Como os Dados Chegam ao Machine Learning e ao Deep Learning

## Parte 3 — Indexação, slicing e filtros com NumPy

© @karizeviecelli - 2026

---

# 68. Acessando dados dentro de um array

Na Parte 2, aprendemos que um array NumPy pode representar:

* uma sequência de valores;
* uma tabela;
* uma matriz de features;
* um conjunto de amostras;
* estruturas com várias dimensões.

Agora aprenderemos a acessar partes específicas desses dados.

Esse processo é importante porque, durante a análise de um dataset, frequentemente precisamos:

* acessar um registro;
* selecionar uma coluna;
* analisar uma feature;
* selecionar um intervalo;
* encontrar valores acima ou abaixo de um limite;
* filtrar dados;
* corrigir valores;
* separar grupos.

Para isso, utilizaremos:

* indexação;
* slicing;
* máscaras booleanas.

---

# 69. Revisando os índices

Em Python e NumPy, os índices começam em zero.

Considere:

```python
notas = np.array([5.5, 7.0, 8.5, 9.0])
```

A organização dos índices será:

```text
Valor:   5.5   7.0   8.5   9.0
Índice:    0     1     2     3
```

Para acessar o primeiro valor:

```python
print(notas[0])
```

Resultado:

```text
5.5
```

Para acessar o segundo valor:

```python
print(notas[1])
```

Resultado:

```text
7.0
```

Para acessar o último valor utilizando índice positivo:

```python
print(notas[3])
```

Resultado:

```text
9.0
```

---

# 70. Índices negativos

O Python também permite acessar elementos contando a partir do final.

Considere:

```python
notas = np.array([5.5, 7.0, 8.5, 9.0])
```

Os índices negativos podem ser representados assim:

```text
Valor:             5.5   7.0   8.5   9.0
Índice positivo:     0     1     2     3
Índice negativo:    -4    -3    -2    -1
```

Para acessar o último elemento:

```python
print(notas[-1])
```

Resultado:

```text
9.0
```

Para acessar o penúltimo:

```python
print(notas[-2])
```

Resultado:

```text
8.5
```

---

## 70.1 Quando usar índices negativos?

Índices negativos são úteis quando:

* não sabemos o tamanho exato do array;
* queremos acessar o último registro;
* queremos acessar os últimos valores;
* estamos analisando o final de uma sequência.

Exemplo:

```python
temperaturas = np.array([20, 21, 23, 25, 24, 22])

print("Última temperatura:", temperaturas[-1])
```

---

# 71. Alterando um valor

Podemos substituir um valor utilizando seu índice.

```python
notas = np.array([5.5, 7.0, 8.5, 9.0])

notas[0] = 6.0

print(notas)
```

Resultado:

```text
[6.  7.  8.5 9. ]
```

O primeiro valor foi alterado.

---

## 71.1 Atenção ao tipo do array

Considere:

```python
valores = np.array([1, 2, 3, 4])
```

Esse array contém inteiros.

Agora tente:

```python
valores[0] = 2.8

print(valores)
```

Resultado possível:

```text
[2 2 3 4]
```

A parte decimal foi removida porque o array foi criado como inteiro.

Para preservar números decimais:

```python
valores = np.array(
    [1, 2, 3, 4],
    dtype=float
)

valores[0] = 2.8

print(valores)
```

Resultado:

```text
[2.8 2.  3.  4. ]
```

---

# 72. Erro de índice inexistente

Considere:

```python
notas = np.array([5.5, 7.0, 8.5, 9.0])
```

O array possui quatro valores.

Os índices válidos são:

```text
0
1
2
3
```

Se tentarmos:

```python
print(notas[4])
```

será gerado um erro semelhante a:

```text
IndexError: index 4 is out of bounds for axis 0 with size 4
```

Isso significa que o índice solicitado não existe.

---

## 72.1 Como investigar

Utilize:

```python
print(notas.size)
```

ou:

```python
print(notas.shape)
```

Resultado:

```text
4
```

ou:

```text
(4,)
```

---

# 73. Slicing em arrays de uma dimensão

Slicing significa selecionar uma parte do array.

A estrutura básica é:

```python
array[inicio:fim]
```

O índice inicial é incluído.

O índice final não é incluído.

Considere:

```python
notas = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
```

Para selecionar os valores dos índices 1 até 3:

```python
print(notas[1:4])
```

Resultado:

```text
[6. 7. 8.]
```

O índice 4 não foi incluído.

---

# 74. Selecionando do início até uma posição

```python
print(notas[:3])
```

Resultado:

```text
[5. 6. 7.]
```

Isso significa:

```text
do início até o índice 3
```

O índice 3 não é incluído.

---

# 75. Selecionando de uma posição até o final

```python
print(notas[2:])
```

Resultado:

```text
[7. 8. 9.]
```

Isso significa:

```text
do índice 2 até o final
```

---

# 76. Selecionando todos os elementos

```python
print(notas[:])
```

Resultado:

```text
[5. 6. 7. 8. 9.]
```

Esse comando seleciona todo o array.

---

# 77. Utilizando passos no slicing

A estrutura completa é:

```python
array[inicio:fim:passo]
```

Considere:

```python
numeros = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

Selecionando de dois em dois:

```python
print(numeros[::2])
```

Resultado:

```text
[0 2 4 6 8]
```

---

## 77.1 Selecionando valores ímpares pela posição

```python
print(numeros[1::2])
```

Resultado:

```text
[1 3 5 7 9]
```

---

## 77.2 Invertendo o array

```python
print(numeros[::-1])
```

Resultado:

```text
[9 8 7 6 5 4 3 2 1 0]
```

O passo `-1` percorre o array ao contrário.

---

# 78. Slicing cria uma visão do array

Em muitos casos, o slicing do NumPy cria uma visualização dos mesmos dados, chamada de `view`.

Observe:

```python
numeros = np.array([10, 20, 30, 40, 50])

parte = numeros[1:4]

print(parte)
```

Resultado:

```text
[20 30 40]
```

Agora altere:

```python
parte[0] = 999

print(parte)
print(numeros)
```

Resultado possível:

```text
[999  30  40]
[ 10 999  30  40  50]
```

O array original também foi alterado.

Isso acontece porque `parte` pode compartilhar os mesmos dados na memória.

---

## 78.1 Criando uma cópia independente

Para evitar alterações no array original:

```python
numeros = np.array([10, 20, 30, 40, 50])

parte = numeros[1:4].copy()

parte[0] = 999

print(parte)
print(numeros)
```

Resultado:

```text
[999  30  40]
[10 20 30 40 50]
```

Agora os arrays são independentes.

---

# 79. Indexação em matrizes

Considere a matriz:

```python
alunos = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85]
])
```

As colunas representam:

```text
Coluna 0 → idade
Coluna 1 → nota
Coluna 2 → frequência
```

As linhas representam os alunos.

---

# 80. Acessando uma linha

Para acessar o primeiro aluno:

```python
print(alunos[0])
```

Resultado:

```text
[18.   8.5 95. ]
```

Para acessar o terceiro aluno:

```python
print(alunos[2])
```

Resultado:

```text
[20.  9. 98.]
```

---

## 80.1 Interpretação

Ao executar:

```python
alunos[2]
```

estamos pedindo:

```text
linha de índice 2
```

Como os índices começam em zero, essa é a terceira linha.

---

# 81. Acessando uma célula específica

Para acessar uma célula, informamos:

```python
matriz[linha, coluna]
```

Exemplo:

```python
print(alunos[0, 1])
```

Resultado:

```text
8.5
```

Interpretação:

```text
linha 0 → primeiro aluno
coluna 1 → nota
```

---

## 81.1 Outro exemplo

```python
print(alunos[2, 2])
```

Resultado:

```text
98.0
```

Interpretação:

```text
linha 2 → terceiro aluno
coluna 2 → frequência
```

---

# 82. Outra forma de acessar uma célula

Também podemos escrever:

```python
print(alunos[0][1])
```

Resultado:

```text
8.5
```

Entretanto, em NumPy, a forma mais comum e recomendada é:

```python
alunos[0, 1]
```

Ela é mais clara e direta.

---

# 83. Acessando uma coluna inteira

Para selecionar uma coluna, utilizamos:

```python
matriz[:, coluna]
```

O símbolo `:` significa:

```text
todas as linhas
```

Para acessar todas as idades:

```python
idades = alunos[:, 0]

print(idades)
```

Resultado:

```text
[18. 19. 20. 21.]
```

---

## 83.1 Selecionando todas as notas

```python
notas = alunos[:, 1]

print(notas)
```

Resultado:

```text
[8.5 6.  9.  7.5]
```

---

## 83.2 Selecionando todas as frequências

```python
frequencias = alunos[:, 2]

print(frequencias)
```

Resultado:

```text
[95. 75. 98. 85.]
```

---

# 84. Interpretando `:`

Considere:

```python
alunos[:, 1]
```

Podemos ler como:

```text
todas as linhas
da coluna 1
```

Portanto:

```text
todas as notas
```

---

# 85. Selecionando várias linhas

Para selecionar as duas primeiras linhas:

```python
print(alunos[0:2])
```

Resultado:

```text
[[18.   8.5 95. ]
 [19.   6.  75. ]]
```

O índice 2 não foi incluído.

---

# 86. Selecionando linhas e colunas ao mesmo tempo

Considere:

```python
print(alunos[0:3, 1:3])
```

Resultado:

```text
[[ 8.5 95. ]
 [ 6.  75. ]
 [ 9.  98. ]]
```

Interpretação:

```text
linhas 0 até 2
colunas 1 até 2
```

Nesse caso, selecionamos:

```text
nota
frequência
```

dos três primeiros alunos.

---

# 87. Selecionando uma faixa de colunas

```python
print(alunos[:, 1:3])
```

Resultado:

```text
[[ 8.5 95. ]
 [ 6.  75. ]
 [ 9.  98. ]
 [ 7.5 85. ]]
```

Interpretação:

```text
todas as linhas
colunas 1 e 2
```

---

# 88. Mantendo duas dimensões ao selecionar uma coluna

Observe:

```python
notas = alunos[:, 1]

print(notas.shape)
```

Resultado:

```text
(4,)
```

O resultado possui uma dimensão.

Se precisarmos manter a estrutura como matriz:

```python
notas_matriz = alunos[:, 1:2]

print(notas_matriz)
print(notas_matriz.shape)
```

Resultado:

```text
[[8.5]
 [6. ]
 [9. ]
 [7.5]]
```

Formato:

```text
(4, 1)
```

---

## 88.1 Diferença importante

```python
alunos[:, 1]
```

Resultado:

```text
(4,)
```

Representa um vetor.

```python
alunos[:, 1:2]
```

Resultado:

```text
(4, 1)
```

Representa uma matriz com quatro linhas e uma coluna.

Essa diferença pode ser importante para determinados algoritmos.

---

# 89. Seleção de linhas específicas

Podemos selecionar linhas por seus índices.

```python
print(alunos[[0, 2]])
```

Resultado:

```text
[[18.   8.5 95. ]
 [20.   9.  98. ]]
```

Foram selecionados:

```text
primeiro aluno
terceiro aluno
```

---

# 90. Seleção de colunas específicas

Para selecionar colunas não consecutivas:

```python
print(alunos[:, [0, 2]])
```

Resultado:

```text
[[18. 95.]
 [19. 75.]
 [20. 98.]
 [21. 85.]]
```

Foram selecionadas:

```text
idade
frequência
```

A coluna de notas não foi incluída.

---

# 91. Alterando uma célula da matriz

```python
alunos[1, 1] = 6.5

print(alunos)
```

A nota do segundo aluno foi alterada.

---

# 92. Alterando uma linha inteira

```python
alunos[0] = [18, 9.0, 97]

print(alunos)
```

A primeira linha foi substituída.

A quantidade de valores precisa ser compatível com o número de colunas.

---

# 93. Alterando uma coluna inteira

```python
alunos[:, 2] = [96, 76, 99, 86]

print(alunos)
```

Todas as frequências foram atualizadas.

A quantidade de valores precisa ser compatível com o número de linhas.

---

# 94. Aplicando um valor a uma coluna inteira

```python
dados = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98]
])

dados[:, 1] = 10

print(dados)
```

Resultado:

```text
[[18. 10. 95.]
 [19. 10. 75.]
 [20. 10. 98.]]
```

Todas as notas foram substituídas por 10.

---

# 95. Máscaras booleanas

Uma máscara booleana é um array contendo valores:

```text
True
False
```

Ela é criada a partir de uma comparação.

Considere:

```python
notas = np.array([5.0, 6.5, 7.0, 8.5, 9.5])
```

Execute:

```python
resultado = notas >= 7

print(resultado)
```

Resultado:

```text
[False False  True  True  True]
```

Cada posição indica se a condição foi atendida.

---

# 96. Utilizando a máscara para filtrar

```python
notas_aprovadas = notas[notas >= 7]

print(notas_aprovadas)
```

Resultado:

```text
[7.  8.5 9.5]
```

O NumPy retornou apenas os valores em que a condição foi verdadeira.

---

# 97. Filtrando valores menores

```python
notas_baixas = notas[notas < 7]

print(notas_baixas)
```

Resultado:

```text
[5.  6.5]
```

---

# 98. Operadores de comparação

Podemos utilizar:

| Operador | Significado    |
| -------- | -------------- |
| `>`      | maior que      |
| `<`      | menor que      |
| `>=`     | maior ou igual |
| `<=`     | menor ou igual |
| `==`     | igual          |
| `!=`     | diferente      |

Exemplos:

```python
notas > 8
```

```python
notas <= 6
```

```python
notas == 7
```

```python
notas != 5
```

---

# 99. Contando valores que atendem à condição

Considere:

```python
notas = np.array([5.0, 6.5, 7.0, 8.5, 9.5])
```

Podemos contar quantas notas são maiores ou iguais a 7:

```python
quantidade = np.sum(notas >= 7)

print(quantidade)
```

Resultado:

```text
3
```

Isso funciona porque:

```text
True → 1
False → 0
```

---

## 99.1 Outra forma com `count_nonzero()`

```python
quantidade = np.count_nonzero(notas >= 7)

print(quantidade)
```

Resultado:

```text
3
```

---

# 100. Verificando se existe algum valor

A função `np.any()` verifica se pelo menos um valor atende à condição.

```python
tem_nota_dez = np.any(notas == 10)

print(tem_nota_dez)
```

Resultado:

```text
False
```

---

# 101. Verificando se todos atendem à condição

A função `np.all()` verifica se todos os valores atendem à condição.

```python
todos_acima_de_quatro = np.all(notas >= 4)

print(todos_acima_de_quatro)
```

Resultado:

```text
True
```

---

# 102. Combinando condições

Para combinar condições com NumPy, utilizamos:

```text
& → e
| → ou
~ → negação
```

As condições devem ser colocadas entre parênteses.

Considere:

```python
notas = np.array([5.0, 6.5, 7.0, 8.5, 9.5])
```

Selecionando notas entre 7 e 9:

```python
filtro = (notas >= 7) & (notas <= 9)

print(notas[filtro])
```

Resultado:

```text
[7.  8.5]
```

---

# 103. Utilizando a condição `ou`

Selecionando notas menores que 6 ou maiores que 9:

```python
filtro = (notas < 6) | (notas > 9)

print(notas[filtro])
```

Resultado:

```text
[5.  9.5]
```

---

# 104. Negando uma condição

```python
filtro = ~(notas >= 7)

print(notas[filtro])
```

Resultado:

```text
[5.  6.5]
```

A expressão significa:

```text
não maior ou igual a 7
```

---

# 105. Por que não utilizar `and` e `or`?

Em condições com arrays NumPy, normalmente não utilizamos:

```python
and
```

ou:

```python
or
```

Exemplo incorreto:

```python
(notas >= 7) and (notas <= 9)
```

Esse código pode gerar erro.

Utilize:

```python
(notas >= 7) & (notas <= 9)
```

Para `ou`, utilize:

```python
(notas < 6) | (notas > 9)
```

---

# 106. Filtrando linhas de uma matriz

Considere:

```python
alunos = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85],
    [22, 5.0, 60]
])
```

Queremos selecionar alunos com nota maior ou igual a 7.

A coluna de notas é:

```python
alunos[:, 1]
```

Criando a condição:

```python
filtro = alunos[:, 1] >= 7

print(filtro)
```

Resultado:

```text
[ True False  True  True False]
```

Aplicando à matriz:

```python
alunos_selecionados = alunos[filtro]

print(alunos_selecionados)
```

Resultado:

```text
[[18.   8.5 95. ]
 [20.   9.  98. ]
 [21.   7.5 85. ]]
```

---

# 107. Filtrando por frequência

Selecionando alunos com frequência menor que 75:

```python
filtro = alunos[:, 2] < 75

print(alunos[filtro])
```

Resultado:

```text
[[22.  5. 60.]]
```

---

# 108. Combinando nota e frequência

Selecionando alunos com:

```text
nota maior ou igual a 7
e
frequência maior ou igual a 75
```

```python
filtro = (
    (alunos[:, 1] >= 7)
    &
    (alunos[:, 2] >= 75)
)

aprovados = alunos[filtro]

print(aprovados)
```

Resultado:

```text
[[18.   8.5 95. ]
 [20.   9.  98. ]
 [21.   7.5 85. ]]
```

---

# 109. Selecionando alunos em risco

Considere como aluno em risco aquele que possui:

```text
nota menor que 7
ou
frequência menor que 75
```

```python
filtro_risco = (
    (alunos[:, 1] < 7)
    |
    (alunos[:, 2] < 75)
)

alunos_risco = alunos[filtro_risco]

print(alunos_risco)
```

---

# 110. Criando um target com máscara booleana

Podemos criar um target utilizando uma condição.

```python
aprovado = (
    (alunos[:, 1] >= 7)
    &
    (alunos[:, 2] >= 75)
)
```

Resultado booleano:

```python
print(aprovado)
```

Possível resultado:

```text
[ True False  True  True False]
```

Convertendo para números:

```python
y = aprovado.astype(int)

print(y)
```

Resultado:

```text
[1 0 1 1 0]
```

---

## 110.1 Utilizando `np.where()`

Outra forma:

```python
y = np.where(
    (alunos[:, 1] >= 7)
    &
    (alunos[:, 2] >= 75),
    1,
    0
)

print(y)
```

Resultado:

```text
[1 0 1 1 0]
```

---

# 111. Entendendo `np.where()`

A estrutura é:

```python
np.where(
    condição,
    valor_se_verdadeiro,
    valor_se_falso
)
```

Exemplo:

```python
situacao = np.where(
    notas >= 7,
    "Aprovado",
    "Reprovado"
)

print(situacao)
```

Resultado:

```text
['Reprovado' 'Reprovado' 'Aprovado' 'Aprovado' 'Aprovado']
```

---

# 112. Substituindo valores com condição

Considere:

```python
notas = np.array([4.0, 6.5, 7.0, 8.5, 11.0])
```

A nota 11 é inválida.

Podemos limitar os valores:

```python
notas_corrigidas = np.clip(
    notas,
    0,
    10
)

print(notas_corrigidas)
```

Resultado:

```text
[ 4.   6.5  7.   8.5 10. ]
```

---

# 113. Corrigindo valores diretamente

Também podemos utilizar uma máscara:

```python
notas[notas > 10] = 10

print(notas)
```

Resultado:

```text
[ 4.   6.5  7.   8.5 10. ]
```

---

## 113.1 Corrigindo valores negativos

```python
notas = np.array([-1.0, 5.0, 7.5, 8.0])

notas[notas < 0] = 0

print(notas)
```

Resultado:

```text
[0.  5.  7.5 8. ]
```

---

# 114. Identificando valores ausentes

O NumPy representa valores numéricos ausentes com:

```text
NaN
```

A sigla significa:

```text
Not a Number
```

Exemplo:

```python
notas = np.array([
    7.5,
    8.0,
    np.nan,
    9.0
])

print(notas)
```

Resultado:

```text
[7.5 8.  nan 9. ]
```

---

# 115. Verificando valores `NaN`

```python
print(np.isnan(notas))
```

Resultado:

```text
[False False  True False]
```

---

## 115.1 Contando valores ausentes

```python
quantidade_nulos = np.sum(
    np.isnan(notas)
)

print(quantidade_nulos)
```

Resultado:

```text
1
```

---

# 116. Filtrando valores válidos

```python
notas_validas = notas[
    ~np.isnan(notas)
]

print(notas_validas)
```

Resultado:

```text
[7.5 8.  9. ]
```

O símbolo `~` negou a condição.

Portanto:

```text
não é NaN
```

---

# 117. Calculando média com `NaN`

Considere:

```python
notas = np.array([
    7.5,
    8.0,
    np.nan,
    9.0
])
```

Ao executar:

```python
print(np.mean(notas))
```

Resultado:

```text
nan
```

Isso acontece porque existe um valor ausente.

---

## 117.1 Utilizando `np.nanmean()`

```python
media = np.nanmean(notas)

print(media)
```

Resultado:

```text
8.166666666666666
```

A função ignorou os valores `NaN`.

Outras funções semelhantes:

```python
np.nansum()
```

```python
np.nanmin()
```

```python
np.nanmax()
```

```python
np.nanmedian()
```

---

# 118. Substituindo valores ausentes pela média

```python
notas = np.array([
    7.5,
    8.0,
    np.nan,
    9.0
])

media = np.nanmean(notas)

notas[np.isnan(notas)] = media

print(notas)
```

Resultado aproximado:

```text
[7.5        8.         8.16666667 9.        ]
```

---

## 118.1 Conexão com a Aula 02

Na Aula 02, fizemos tratamento de valores ausentes utilizando Pandas.

Exemplo:

```python
df["nota"] = df["nota"].fillna(
    df["nota"].mean()
)
```

Com NumPy, podemos realizar um processo semelhante utilizando:

```python
np.isnan()
```

e:

```python
np.nanmean()
```

---

# 119. Filtragem e preparação de dados

Antes de treinar um modelo, podemos utilizar máscaras para verificar:

* valores negativos;
* notas acima de 10;
* frequências acima de 100;
* idades inválidas;
* valores ausentes;
* registros fora do padrão.

Exemplo:

```python
frequencias = np.array([
    95,
    80,
    110,
    70,
    -5
])
```

Encontrando valores inválidos:

```python
invalidos = (
    (frequencias < 0)
    |
    (frequencias > 100)
)

print(frequencias[invalidos])
```

Resultado:

```text
[110  -5]
```

---

# 120. Corrigindo frequências

```python
frequencias_corrigidas = np.clip(
    frequencias,
    0,
    100
)

print(frequencias_corrigidas)
```

Resultado:

```text
[ 95  80 100  70   0]
```

---

# 121. Separando grupos

Considere:

```python
notas = np.array([
    4.5,
    6.0,
    7.0,
    8.5,
    9.5
])
```

Podemos criar grupos.

Notas baixas:

```python
grupo_baixo = notas[notas < 6]
```

Notas intermediárias:

```python
grupo_intermediario = notas[
    (notas >= 6)
    &
    (notas < 8)
]
```

Notas altas:

```python
grupo_alto = notas[notas >= 8]
```

---

# 122. Classificando faixas com `np.where()`

Podemos criar duas categorias:

```python
categoria = np.where(
    notas >= 7,
    "Satisfatório",
    "Abaixo do esperado"
)

print(categoria)
```

Para várias categorias, podemos utilizar `np.select()`.

---

# 123. Criando várias categorias com `np.select()`

```python
condicoes = [
    notas < 6,
    (notas >= 6) & (notas < 8),
    notas >= 8
]

categorias = [
    "Baixo",
    "Intermediário",
    "Alto"
]

resultado = np.select(
    condicoes,
    categorias,
    default="Não classificado"
)

print(resultado)
```

Resultado:

```text
['Baixo' 'Intermediário' 'Intermediário' 'Alto' 'Alto']
```

---

# 124. Indexação e Machine Learning

Durante a construção de um modelo, podemos usar indexação para:

* selecionar features;
* remover colunas;
* analisar uma variável;
* criar subconjuntos;
* separar dados;
* investigar erros;
* selecionar amostras específicas.

Considere:

```python
X = alunos[:, [0, 1, 2]]
```

Esse código seleciona:

```text
idade
nota
frequência
```

Se quisermos utilizar apenas nota e frequência:

```python
X_reduzido = alunos[:, [1, 2]]
```

---

# 125. Comparando diferentes conjuntos de features

Modelo com três features:

```python
X_completo = alunos[:, [0, 1, 2]]
```

Modelo com duas features:

```python
X_reduzido = alunos[:, [1, 2]]
```

Podemos treinar dois modelos e comparar os resultados.

Isso permite investigar:

```text
A idade realmente ajuda na previsão?
```

Essa análise está relacionada à seleção de features.

---

# 126. Treinando com dados filtrados

Considere:

```python
X = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85],
    [22, 5.0, 60],
    [65, 8.0, 90]
])

y = np.array([1, 0, 1, 1, 0, 1])
```

Suponha que o projeto considere idades entre 16 e 60.

Criamos o filtro:

```python
filtro_idade = (
    (X[:, 0] >= 16)
    &
    (X[:, 0] <= 60)
)
```

Aplicamos o mesmo filtro em `X` e `y`:

```python
X_filtrado = X[filtro_idade]
y_filtrado = y[filtro_idade]
```

Verificamos:

```python
print(X_filtrado)
print(y_filtrado)
```

---

## 126.1 Por que aplicar o mesmo filtro?

Cada linha de `X` possui uma resposta correspondente em `y`.

Se removermos uma linha apenas de `X`, os dados ficarão desalinhados.

Por isso:

```python
X_filtrado = X[filtro]
y_filtrado = y[filtro]
```

A mesma máscara deve ser aplicada às duas estruturas.

---

# 127. Erro de desalinhamento

Exemplo incorreto:

```python
X_filtrado = X[filtro_idade]

modelo.fit(
    X_filtrado,
    y
)
```

Agora:

```text
X_filtrado possui menos linhas
y ainda possui todas as respostas
```

Isso pode gerar erro de quantidade inconsistente.

---

# 128. Máscaras e datasets reais

Em datasets reais, podemos construir filtros como:

```python
idade_valida = (
    (df["idade"] >= 16)
    &
    (df["idade"] <= 100)
)
```

No Pandas, a lógica é semelhante.

Exemplo:

```python
df_filtrado = df[
    (df["idade"] >= 16)
    &
    (df["idade"] <= 100)
]
```

O conceito de máscara booleana aprendido com NumPy também aparece no Pandas.

---

# 129. Pandas e NumPy utilizam lógica semelhante

## NumPy

```python
notas[notas >= 7]
```

## Pandas

```python
df[df["nota"] >= 7]
```

Nos dois casos:

1. criamos uma condição;
2. obtemos valores `True` e `False`;
3. utilizamos a condição para selecionar dados.

---

# 130. Atividade prática 1 — Índices

Considere:

```python
valores = np.array([
    10,
    20,
    30,
    40,
    50,
    60
])
```

Resolva:

1. Acesse o primeiro valor.
2. Acesse o quarto valor.
3. Acesse o último valor.
4. Acesse o penúltimo valor.
5. Altere o segundo valor para `25`.
6. Inverta o array.
7. Selecione os três primeiros valores.
8. Selecione do terceiro valor até o final.
9. Selecione os valores de dois em dois.

---

# 131. Atividade prática 2 — Matriz de alunos

Considere:

```python
alunos = np.array([
    [17, 7.5, 90],
    [18, 8.0, 95],
    [19, 6.5, 70],
    [20, 9.0, 98],
    [21, 5.5, 65]
])
```

Resolva:

1. Acesse o primeiro aluno.
2. Acesse o último aluno.
3. Acesse a nota do segundo aluno.
4. Acesse a frequência do quarto aluno.
5. Selecione todas as idades.
6. Selecione todas as notas.
7. Selecione todas as frequências.
8. Selecione os três primeiros alunos.
9. Selecione nota e frequência de todos os alunos.
10. Selecione idade e frequência, ignorando a nota.

---

# 132. Atividade prática 3 — Filtros

Utilizando a matriz anterior:

1. Selecione alunos com nota maior ou igual a 7.
2. Selecione alunos com frequência menor que 75.
3. Selecione alunos com idade maior que 18.
4. Selecione alunos com nota maior ou igual a 7 e frequência maior ou igual a 75.
5. Selecione alunos com nota menor que 6 ou frequência menor que 70.
6. Conte quantos alunos possuem nota maior ou igual a 7.
7. Verifique se existe algum aluno com frequência igual a 100.
8. Verifique se todos os alunos possuem idade maior ou igual a 16.

---

# 133. Atividade prática 4 — Valores inválidos

Considere:

```python
notas = np.array([
    7.5,
    -1.0,
    8.5,
    11.0,
    np.nan,
    6.0
])
```

Resolva:

1. Identifique notas menores que zero.
2. Identifique notas maiores que dez.
3. Conte os valores ausentes.
4. Selecione somente notas válidas.
5. Corrija valores menores que zero para zero.
6. Corrija valores maiores que dez para dez.
7. Substitua o valor ausente pela média das notas válidas.
8. Calcule a média final.

---

# 134. Desafio — Análise de risco escolar

Considere o dataset:

```python
dados = np.array([
    [16, 8.0, 95, 10],
    [17, 5.5, 68, 4],
    [18, 7.0, 80, 8],
    [19, 4.5, 60, 2],
    [20, 9.0, 98, 12],
    [21, 6.0, 72, 5],
    [22, 8.5, 90, 11],
    [23, 5.0, 65, 3]
])
```

As colunas representam:

```text
0 → idade
1 → nota
2 → frequência
3 → atividades entregues
```

Um aluno será considerado em risco quando atender a pelo menos uma das condições:

```text
nota menor que 6
frequência menor que 75
atividades entregues menor que 5
```

---

## 134.1 Tarefas

1. Exiba o formato do dataset.
2. Informe quantas amostras existem.
3. Informe quantas features existem.
4. Extraia cada coluna em um array separado.
5. Crie uma máscara para alunos com nota menor que 6.
6. Crie uma máscara para frequência menor que 75.
7. Crie uma máscara para menos de 5 atividades.
8. Crie a máscara final de risco.
9. Exiba somente os alunos em risco.
10. Conte quantos alunos estão em risco.
11. Calcule a porcentagem de alunos em risco.
12. Crie um target:

    * `1` para aluno em risco;
    * `0` para aluno fora de risco.
13. Exiba `X`.
14. Exiba `y`.
15. Verifique se `X` e `y` possuem a mesma quantidade de amostras.

---

## 134.2 Estrutura inicial

```python
idades = dados[:, 0]
notas = dados[:, 1]
frequencias = dados[:, 2]
atividades = dados[:, 3]
```

Máscara de risco:

```python
risco = (
    (notas < 6)
    |
    (frequencias < 75)
    |
    (atividades < 5)
)
```

Target:

```python
y = risco.astype(int)
```

---

# 135. Reflexão sobre o desafio

Responda:

1. A máscara criada é um modelo de Machine Learning?
2. Quem definiu as regras de risco?
3. Como um modelo de Machine Learning poderia aprender essas regras?
4. Quais seriam as features?
5. Qual seria o target?
6. Uma Rede Neural poderia receber os mesmos dados?
7. Qual seria o formato esperado da entrada?
8. O que mudaria entre uma Árvore de Decisão e uma Rede Neural?
9. O que permaneceria igual?

---

# 136. Ponte para Deep Learning

Os filtros estudados nesta parte continuam importantes em Deep Learning.

Antes de enviar dados para uma Rede Neural, precisamos verificar:

* se há valores ausentes;
* se os dados estão no formato correto;
* se as dimensões são compatíveis;
* se as amostras estão alinhadas com os targets;
* se existem valores inválidos;
* se as features estão na ordem correta.

Uma Rede Neural não corrige automaticamente problemas nos dados.

Ela aprende a partir do que recebe.

Se os dados estiverem incorretos, o aprendizado também será prejudicado.

---

# 137. Exemplo de formato para dados tabulares

Considere:

```python
X.shape
```

Resultado:

```text
(100, 4)
```

Interpretação:

```text
100 alunos
4 features
```

Esse formato pode ser utilizado por:

* Árvore de Decisão;
* KNN;
* Regressão;
* Random Forest;
* Rede Neural densa.

O algoritmo muda, mas a organização inicial dos dados pode permanecer semelhante.

---

# 138. Exemplo de lote em uma Rede Neural

Considere:

```text
(32, 4)
```

Isso pode representar:

```text
32 alunos processados em um lote
4 features por aluno
```

A primeira dimensão indica a quantidade de amostras.

A segunda dimensão indica a quantidade de características de cada amostra.

---

# 139. Erros comuns nesta etapa

## Erro 1 — Esquecer que o índice começa em zero

```python
dados[1]
```

Esse comando acessa o segundo registro, e não o primeiro.

---

## Erro 2 — Incluir o limite final no slicing

```python
dados[0:3]
```

Esse comando seleciona os índices:

```text
0
1
2
```

O índice 3 não é incluído.

---

## Erro 3 — Confundir linha e coluna

```python
dados[1, 2]
```

A ordem é:

```text
linha
coluna
```

---

## Erro 4 — Esquecer os parênteses nas condições

Incorreto:

```python
notas >= 7 & frequencias >= 75
```

Correto:

```python
(notas >= 7) & (frequencias >= 75)
```

---

## Erro 5 — Utilizar `and` com arrays

Incorreto:

```python
(notas >= 7) and (frequencias >= 75)
```

Correto:

```python
(notas >= 7) & (frequencias >= 75)
```

---

## Erro 6 — Filtrar `X`, mas não filtrar `y`

Incorreto:

```python
X = X[filtro]
```

e manter `y` sem alteração.

Correto:

```python
X = X[filtro]
y = y[filtro]
```

---

## Erro 7 — Alterar uma view sem perceber

```python
parte = array[0:3]
```

Modificar `parte` pode alterar o array original.

Para evitar:

```python
parte = array[0:3].copy()
```

---

# 140. Checklist de compreensão da Parte 3

* [ ] Sei acessar um valor por índice.
* [ ] Sei utilizar índices negativos.
* [ ] Sei selecionar um intervalo com slicing.
* [ ] Sei utilizar início, fim e passo.
* [ ] Sei inverter um array.
* [ ] Sei acessar uma linha da matriz.
* [ ] Sei acessar uma coluna da matriz.
* [ ] Sei acessar uma célula específica.
* [ ] Sei selecionar várias linhas.
* [ ] Sei selecionar várias colunas.
* [ ] Sei criar uma máscara booleana.
* [ ] Sei filtrar dados por condição.
* [ ] Sei combinar condições com `&`.
* [ ] Sei combinar condições com `|`.
* [ ] Sei utilizar `np.any()`.
* [ ] Sei utilizar `np.all()`.
* [ ] Sei utilizar `np.where()`.
* [ ] Sei identificar valores `NaN`.
* [ ] Sei filtrar valores ausentes.
* [ ] Sei aplicar a mesma máscara em `X` e `y`.
* [ ] Entendo como os filtros ajudam a preparar dados para modelos.

---

# 141. Resumo da Parte 3

Nesta parte, aprendemos:

* índices positivos;
* índices negativos;
* alteração de valores;
* slicing;
* seleção por intervalos;
* uso de passos;
* inversão de arrays;
* diferença entre view e cópia;
* acesso a linhas;
* acesso a colunas;
* acesso a células;
* seleção de múltiplas linhas e colunas;
* máscaras booleanas;
* operadores de comparação;
* combinação de condições;
* filtros em matrizes;
* criação de targets;
* uso de `np.where()`;
* identificação e tratamento de valores inválidos;
* identificação de valores ausentes;
* aplicação de filtros em `X` e `y`;
* relação entre filtragem, Machine Learning e Deep Learning.

---

# 142. O que veremos na Parte 4

Na próxima parte, aprenderemos a realizar cálculos sobre conjuntos inteiros de dados.

Serão estudados:

* operações vetorizadas;
* soma;
* subtração;
* multiplicação;
* divisão;
* média;
* mediana;
* mínimo;
* máximo;
* soma total;
* desvio padrão;
* operações por linha;
* operações por coluna;
* conceito de eixo;
* broadcasting;
* comparação entre laços e vetorização;
* preparação matemática dos dados para Machine Learning.

© @karizeviecelli - 2026

--
# Aula 04 — Como os Dados Chegam ao Machine Learning e ao Deep Learning

## Parte 4 — Operações vetorizadas, estatística, eixos e broadcasting

© @karizeviecelli - 2026

---

# 143. Operações com arrays NumPy

Nas partes anteriores, aprendemos a:

* criar arrays;
* verificar suas dimensões;
* acessar linhas e colunas;
* utilizar índices;
* aplicar filtros;
* selecionar registros;
* corrigir valores.

Agora vamos aprender a realizar cálculos sobre os dados.

Esses cálculos são fundamentais em Machine Learning porque os algoritmos trabalham com números.

Durante a preparação dos dados, podemos precisar:

* aumentar ou diminuir valores;
* calcular médias;
* encontrar valores mínimos e máximos;
* medir a variação dos dados;
* comparar colunas;
* ajustar escalas;
* transformar matrizes;
* aplicar cálculos em todas as amostras.

O NumPy permite realizar essas operações de forma direta e eficiente.

---

# 144. Operações vetorizadas

Uma operação vetorizada é aplicada a vários valores ao mesmo tempo.

Considere:

```python
import numpy as np

notas = np.array([5.0, 6.5, 7.0, 8.5, 9.0])
```

Para aumentar todas as notas em um ponto:

```python
notas_atualizadas = notas + 1

print(notas_atualizadas)
```

Resultado:

```text
[ 6.   7.5  8.   9.5 10. ]
```

O NumPy aplicou a operação a todos os elementos do array.

---

# 145. Comparando lista Python e NumPy

Com uma lista Python:

```python
notas_lista = [5.0, 6.5, 7.0, 8.5, 9.0]
```

Para aumentar todas as notas, poderíamos utilizar:

```python
novas_notas = []

for nota in notas_lista:
    novas_notas.append(nota + 1)

print(novas_notas)
```

Com NumPy:

```python
notas_array = np.array([5.0, 6.5, 7.0, 8.5, 9.0])

novas_notas = notas_array + 1

print(novas_notas)
```

A operação vetorizada torna o código mais curto e mais próximo da linguagem matemática.

---

# 146. Soma

```python
valores = np.array([10, 20, 30, 40])

resultado = valores + 5

print(resultado)
```

Resultado:

```text
[15 25 35 45]
```

O valor `5` foi somado a cada elemento.

---

# 147. Subtração

```python
resultado = valores - 2

print(resultado)
```

Resultado:

```text
[ 8 18 28 38]
```

---

# 148. Multiplicação

```python
resultado = valores * 2

print(resultado)
```

Resultado:

```text
[20 40 60 80]
```

---

# 149. Divisão

```python
resultado = valores / 10

print(resultado)
```

Resultado:

```text
[1. 2. 3. 4.]
```

Mesmo que os valores originais sejam inteiros, a divisão normalmente produz números decimais.

---

# 150. Potência

```python
resultado = valores ** 2

print(resultado)
```

Resultado:

```text
[ 100  400  900 1600]
```

O operador `**` representa potência.

---

# 151. Raiz quadrada

```python
resultado = np.sqrt(valores)

print(resultado)
```

Resultado aproximado:

```text
[3.16227766 4.47213595 5.47722558 6.32455532]
```

---

# 152. Operações entre dois arrays

Também podemos realizar operações entre arrays.

```python
array_a = np.array([10, 20, 30])

array_b = np.array([1, 2, 3])
```

Soma:

```python
print(array_a + array_b)
```

Resultado:

```text
[11 22 33]
```

Subtração:

```python
print(array_a - array_b)
```

Resultado:

```text
[ 9 18 27]
```

Multiplicação:

```python
print(array_a * array_b)
```

Resultado:

```text
[10 40 90]
```

Divisão:

```python
print(array_a / array_b)
```

Resultado:

```text
[10. 10. 10.]
```

---

# 153. Operações elemento a elemento

As operações anteriores acontecem posição por posição.

```text
10 + 1
20 + 2
30 + 3
```

Por isso, dizemos que são operações elemento a elemento.

Na multiplicação:

```text
10 × 1
20 × 2
30 × 3
```

Não se trata de multiplicação matricial.

É uma multiplicação entre elementos correspondentes.

---

# 154. Arrays com formatos incompatíveis

Considere:

```python
array_a = np.array([10, 20, 30])

array_b = np.array([1, 2])
```

Ao executar:

```python
array_a + array_b
```

pode aparecer um erro semelhante a:

```text
ValueError: operands could not be broadcast together
```

Isso acontece porque os formatos não são compatíveis.

```text
array_a.shape = (3,)
array_b.shape = (2,)
```

O NumPy não consegue decidir como combinar os elementos.

---

# 155. Operações em matrizes

Considere:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98]
])
```

Podemos somar um valor a toda a matriz:

```python
resultado = dados + 1

print(resultado)
```

Resultado:

```text
[[19.   9.  91. ]
 [20.   7.5 76. ]
 [21.  10.  99. ]]
```

Entretanto, nesse caso, somar `1` a todas as colunas pode não fazer sentido.

A idade, a nota e a frequência representam grandezas diferentes.

Por isso, devemos pensar no significado dos dados antes de aplicar operações.

---

# 156. Aplicando operação em uma coluna

Considere:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98]
])
```

Para aumentar apenas as notas:

```python
dados[:, 1] = dados[:, 1] + 1

print(dados)
```

Resultado:

```text
[[18.   9.  90. ]
 [19.   7.5 75. ]
 [20.  10.  98. ]]
```

---

## 156.1 Forma reduzida

Também podemos escrever:

```python
dados[:, 1] += 1
```

Esse comando altera diretamente a coluna.

---

# 157. Limitando os valores

Após aumentar as notas, alguma nota pode ficar acima de 10.

Podemos utilizar:

```python
dados[:, 1] = np.clip(
    dados[:, 1],
    0,
    10
)
```

Agora todas as notas permanecerão entre 0 e 10.

---

# 158. Funções estatísticas

O NumPy possui várias funções para resumir os dados.

As mais utilizadas são:

```text
np.sum()
np.mean()
np.median()
np.min()
np.max()
np.std()
np.var()
```

Essas funções ajudam a compreender o comportamento de um conjunto de dados.

---

# 159. Soma total com `np.sum()`

```python
notas = np.array([5.0, 6.5, 7.0, 8.5, 9.0])

soma = np.sum(notas)

print(soma)
```

Resultado:

```text
36.0
```

---

# 160. Média com `np.mean()`

```python
media = np.mean(notas)

print(media)
```

Resultado:

```text
7.2
```

A média é calculada pela soma dos valores dividida pela quantidade de elementos.

```text
36 ÷ 5 = 7,2
```

---

# 161. Mediana com `np.median()`

```python
mediana = np.median(notas)

print(mediana)
```

Resultado:

```text
7.0
```

A mediana é o valor central quando os dados estão ordenados.

Considere:

```text
5.0
6.5
7.0
8.5
9.0
```

O valor central é 7.

---

# 162. Média e mediana não são a mesma coisa

Considere:

```python
salarios = np.array([
    2000,
    2200,
    2300,
    2400,
    15000
])
```

Média:

```python
print(np.mean(salarios))
```

Resultado:

```text
4780.0
```

Mediana:

```python
print(np.median(salarios))
```

Resultado:

```text
2300.0
```

O valor 15000 aumenta bastante a média.

A mediana representa melhor o centro dos dados nesse caso.

---

# 163. Valores mínimo e máximo

```python
print(np.min(notas))
```

Resultado:

```text
5.0
```

```python
print(np.max(notas))
```

Resultado:

```text
9.0
```

Essas funções ajudam a identificar os limites do conjunto.

---

# 164. Amplitude

A amplitude é a diferença entre o valor máximo e o mínimo.

```python
amplitude = np.max(notas) - np.min(notas)

print(amplitude)
```

Resultado:

```text
4.0
```

A amplitude fornece uma medida simples da variação dos dados.

---

# 165. Desvio padrão

O desvio padrão mede o quanto os valores se afastam da média.

```python
desvio = np.std(notas)

print(desvio)
```

Um desvio padrão pequeno indica que os valores estão próximos da média.

Um desvio padrão maior indica maior dispersão.

---

## 165.1 Analogia

Considere duas turmas.

Turma A:

```text
7.0
7.1
6.9
7.0
7.0
```

Turma B:

```text
2.0
5.0
7.0
9.0
12.0
```

As duas turmas podem ter médias semelhantes.

Entretanto, os valores da Turma B estão muito mais espalhados.

O desvio padrão ajuda a identificar essa diferença.

---

# 166. Variância

A variância também mede a dispersão dos dados.

```python
variancia = np.var(notas)

print(variancia)
```

O desvio padrão é a raiz quadrada da variância.

```python
print(np.sqrt(np.var(notas)))
```

Esse resultado deve ser equivalente a:

```python
print(np.std(notas))
```

---

# 167. Resumo estatístico

```python
notas = np.array([5.0, 6.5, 7.0, 8.5, 9.0])

print("Quantidade:", notas.size)
print("Soma:", np.sum(notas))
print("Média:", np.mean(notas))
print("Mediana:", np.median(notas))
print("Mínimo:", np.min(notas))
print("Máximo:", np.max(notas))
print("Desvio padrão:", np.std(notas))
print("Variância:", np.var(notas))
```

---

# 168. O conceito de eixo

Em uma matriz, podemos calcular valores:

* para a matriz inteira;
* por coluna;
* por linha.

Para isso, utilizamos o parâmetro `axis`.

Considere:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98],
    [21, 7.5, 85]
])
```

As colunas representam:

```text
idade
nota
frequência
```

---

# 169. Operação sem eixo

```python
media_total = np.mean(dados)

print(media_total)
```

O NumPy calcula a média de todos os números da matriz.

Entretanto, esse resultado não possui grande significado, porque mistura:

* idade;
* nota;
* frequência.

Essas variáveis possuem escalas diferentes.

---

# 170. `axis=0`

Ao utilizar:

```python
np.mean(dados, axis=0)
```

o NumPy calcula a média de cada coluna.

```python
medias_colunas = np.mean(
    dados,
    axis=0
)

print(medias_colunas)
```

Resultado aproximado:

```text
[19.5   7.75 87.  ]
```

Interpretação:

```text
média das idades
média das notas
média das frequências
```

---

# 171. Como entender `axis=0`

Podemos pensar que o eixo 0 percorre as linhas para calcular um resultado por coluna.

```text
           idade   nota   frequência
Aluno 1      18     8.0       90
Aluno 2      19     6.5       75
Aluno 3      20     9.0       98
Aluno 4      21     7.5       85
             ↓       ↓         ↓
          média   média      média
```

---

# 172. `axis=1`

Ao utilizar:

```python
np.mean(dados, axis=1)
```

o NumPy calcula uma média para cada linha.

```python
medias_linhas = np.mean(
    dados,
    axis=1
)

print(medias_linhas)
```

Entretanto, nesse exemplo, essa média mistura idade, nota e frequência.

Matematicamente o cálculo funciona, mas pode não ter significado prático.

---

# 173. Como entender `axis=1`

O eixo 1 percorre as colunas para calcular um resultado por linha.

```text
Aluno 1 → [18, 8.0, 90] → média
Aluno 2 → [19, 6.5, 75] → média
Aluno 3 → [20, 9.0, 98] → média
Aluno 4 → [21, 7.5, 85] → média
```

---

# 174. Regra prática para eixos

Em uma matriz bidimensional:

```text
axis=0 → operação por coluna
axis=1 → operação por linha
```

Essa regra ajuda a lembrar o comportamento.

---

# 175. Soma por coluna

```python
somas_colunas = np.sum(
    dados,
    axis=0
)

print(somas_colunas)
```

O resultado contém:

```text
soma das idades
soma das notas
soma das frequências
```

---

# 176. Mínimo por coluna

```python
minimos = np.min(
    dados,
    axis=0
)

print(minimos)
```

Resultado:

```text
menor idade
menor nota
menor frequência
```

---

# 177. Máximo por coluna

```python
maximos = np.max(
    dados,
    axis=0
)

print(maximos)
```

Resultado:

```text
maior idade
maior nota
maior frequência
```

---

# 178. Desvio padrão por coluna

```python
desvios = np.std(
    dados,
    axis=0
)

print(desvios)
```

Isso permite verificar a dispersão de cada feature.

---

# 179. Analisando apenas notas e frequência

Podemos selecionar colunas específicas:

```python
notas_frequencias = dados[:, 1:3]
```

Agora:

```python
print(
    np.mean(
        notas_frequencias,
        axis=0
    )
)
```

O resultado conterá:

```text
média das notas
média das frequências
```

---

# 180. `argmin()` e `argmax()`

As funções `argmin()` e `argmax()` retornam o índice do menor ou maior valor.

```python
notas = np.array([
    5.0,
    6.5,
    9.0,
    8.0
])
```

Índice da menor nota:

```python
indice_menor = np.argmin(notas)

print(indice_menor)
```

Resultado:

```text
0
```

Índice da maior nota:

```python
indice_maior = np.argmax(notas)

print(indice_maior)
```

Resultado:

```text
2
```

---

# 181. Encontrando o aluno com maior nota

Considere:

```python
alunos = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98],
    [21, 7.5, 85]
])
```

Extraímos as notas:

```python
notas = alunos[:, 1]
```

Encontramos o índice:

```python
indice_maior_nota = np.argmax(notas)
```

Selecionamos o aluno:

```python
aluno_maior_nota = alunos[
    indice_maior_nota
]

print(aluno_maior_nota)
```

Resultado:

```text
[20.  9. 98.]
```

---

# 182. Ordenando valores

```python
notas = np.array([
    7.5,
    5.0,
    9.0,
    6.5
])

notas_ordenadas = np.sort(notas)

print(notas_ordenadas)
```

Resultado:

```text
[5.  6.5 7.5 9. ]
```

O array original não é alterado automaticamente.

---

# 183. Ordenação decrescente

```python
notas_decrescentes = np.sort(notas)[::-1]

print(notas_decrescentes)
```

Resultado:

```text
[9.  7.5 6.5 5. ]
```

---

# 184. Ordenando uma matriz por uma coluna

Considere:

```python
alunos = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98],
    [21, 7.5, 85]
])
```

Para ordenar pelas notas:

```python
indices = np.argsort(
    alunos[:, 1]
)

alunos_ordenados = alunos[indices]

print(alunos_ordenados)
```

---

## 184.1 Ordem decrescente

```python
indices = np.argsort(
    alunos[:, 1]
)[::-1]

alunos_ordenados = alunos[indices]

print(alunos_ordenados)
```

---

# 185. Percentis

Percentis dividem os dados em partes.

```python
notas = np.array([
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0
])
```

Primeiro quartil:

```python
q1 = np.percentile(
    notas,
    25
)
```

Mediana:

```python
q2 = np.percentile(
    notas,
    50
)
```

Terceiro quartil:

```python
q3 = np.percentile(
    notas,
    75
)
```

---

# 186. Por que percentis são úteis?

Percentis ajudam a:

* compreender a distribuição;
* identificar valores extremos;
* comparar grupos;
* estabelecer faixas;
* detectar possíveis outliers.

Exemplo:

```python
print("25%:", q1)
print("50%:", q2)
print("75%:", q3)
```

---

# 187. O que é broadcasting?

Broadcasting é um mecanismo do NumPy que permite combinar arrays de formatos diferentes em determinadas situações.

Considere:

```python
notas = np.array([
    5.0,
    6.5,
    7.0,
    8.5
])
```

Ao executar:

```python
notas + 1
```

o NumPy trata o valor `1` como se ele fosse aplicado a todas as posições.

Podemos visualizar:

```text
[5.0, 6.5, 7.0, 8.5]
+
[1.0, 1.0, 1.0, 1.0]
```

O NumPy não precisa criar explicitamente o segundo array.

---

# 188. Broadcasting em uma matriz

Considere:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98]
])
```

Agora:

```python
ajustes = np.array([
    0,
    1,
    0
])
```

Executando:

```python
resultado = dados + ajustes

print(resultado)
```

Resultado:

```text
[[18.   9.  90. ]
 [19.   7.5 75. ]
 [20.  10.  98. ]]
```

O array `ajustes` foi aplicado a cada linha.

---

# 189. Interpretando o broadcasting

Temos:

```text
dados.shape = (3, 3)
ajustes.shape = (3,)
```

O NumPy associa:

```text
0 → idade
1 → nota
0 → frequência
```

Em cada linha, apenas a nota é aumentada.

---

# 190. Exemplo de ajuste por coluna

```python
ajustes = np.array([
    1,
    0.5,
    2
])
```

Aplicando:

```python
resultado = dados + ajustes
```

Interpretação:

```text
idade + 1
nota + 0.5
frequência + 2
```

para todos os alunos.

---

# 191. Subtraindo a média de cada coluna

Considere:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98]
])
```

Calculamos as médias:

```python
medias = np.mean(
    dados,
    axis=0
)

print(medias)
```

Agora subtraímos:

```python
dados_centralizados = dados - medias

print(dados_centralizados)
```

O broadcasting aplica a média correspondente a cada coluna.

---

# 192. O que significa centralizar os dados?

Centralizar significa subtrair a média.

Após a centralização, cada coluna terá média próxima de zero.

```python
print(
    np.mean(
        dados_centralizados,
        axis=0
    )
)
```

O resultado será próximo de:

```text
[0. 0. 0.]
```

Pequenas diferenças podem aparecer devido à precisão decimal.

---

# 193. Padronização manual

Uma forma comum de padronização é:

```text
valor padronizado =
(valor - média) ÷ desvio padrão
```

Com NumPy:

```python
medias = np.mean(
    dados,
    axis=0
)

desvios = np.std(
    dados,
    axis=0
)

dados_padronizados = (
    dados - medias
) / desvios

print(dados_padronizados)
```

---

# 194. O que acontece após a padronização?

As colunas passam a ter aproximadamente:

```text
média = 0
desvio padrão = 1
```

Verificando:

```python
print(
    np.mean(
        dados_padronizados,
        axis=0
    )
)
```

```python
print(
    np.std(
        dados_padronizados,
        axis=0
    )
)
```

---

# 195. Atenção à divisão por zero

Se uma coluna possuir sempre o mesmo valor, seu desvio padrão será zero.

Exemplo:

```python
dados = np.array([
    [18, 10],
    [19, 10],
    [20, 10]
])
```

A segunda coluna não varia.

```python
print(
    np.std(
        dados,
        axis=0
    )
)
```

Resultado:

```text
[0.81649658 0.        ]
```

Dividir por zero pode gerar:

```text
nan
inf
```

---

## 195.1 Solução didática

Podemos substituir desvios iguais a zero:

```python
desvios = np.std(
    dados,
    axis=0
)

desvios[desvios == 0] = 1
```

Depois:

```python
dados_padronizados = (
    dados - np.mean(
        dados,
        axis=0
    )
) / desvios
```

Na prática, utilizaremos ferramentas prontas do Scikit-Learn, como `StandardScaler`.

---

# 196. Normalização entre zero e um

Outra transformação comum é colocar os dados entre 0 e 1.

A fórmula é:

```text
valor normalizado =
(valor - mínimo) ÷ (máximo - mínimo)
```

Com NumPy:

```python
minimos = np.min(
    dados,
    axis=0
)

maximos = np.max(
    dados,
    axis=0
)

dados_normalizados = (
    dados - minimos
) / (
    maximos - minimos
)

print(dados_normalizados)
```

---

# 197. Por que ajustar a escala dos dados?

Considere duas features:

```text
nota → entre 0 e 10
frequência → entre 0 e 100
```

A frequência possui valores numericamente maiores.

Alguns algoritmos utilizam distância.

Nesses casos, a frequência pode influenciar mais o cálculo apenas por causa da escala.

Isso será especialmente importante quando estudarmos KNN.

---

# 198. Exemplo de distância

Aluno A:

```text
nota = 8
frequência = 80
```

Aluno B:

```text
nota = 9
frequência = 90
```

Diferenças:

```text
nota → diferença de 1
frequência → diferença de 10
```

Sem ajuste de escala, a frequência terá maior peso numérico.

Por isso, algoritmos baseados em distância costumam exigir normalização ou padronização.

---

# 199. Árvore de Decisão e escala

A Árvore de Decisão geralmente não depende da distância entre os dados.

Ela cria regras como:

```text
nota <= 6.5
frequência > 75
```

Por isso, a escala costuma afetar menos esse algoritmo.

Já algoritmos como KNN são bastante sensíveis à escala.

---

# 200. Broadcasting com vetor coluna

Considere:

```python
valores = np.array([
    [1],
    [2],
    [3]
])
```

Formato:

```python
print(valores.shape)
```

Resultado:

```text
(3, 1)
```

Agora:

```python
colunas = np.array([
    10,
    20,
    30
])
```

Formato:

```text
(3,)
```

Ao somar:

```python
resultado = valores + colunas

print(resultado)
```

Resultado:

```text
[[11 21 31]
 [12 22 32]
 [13 23 33]]
```

---

# 201. Como o broadcasting funcionou?

O NumPy combinou:

```text
(3, 1)
com
(3,)
```

O vetor coluna foi expandido horizontalmente.

O vetor linha foi expandido verticalmente.

O resultado possui formato:

```text
(3, 3)
```

---

# 202. Regras simplificadas de broadcasting

Ao comparar os formatos da direita para a esquerda, as dimensões são compatíveis quando:

* são iguais;
* uma delas é igual a 1;
* uma dimensão não existe e pode ser adicionada.

Exemplos compatíveis:

```text
(3, 3)
(3,)
```

```text
(4, 1)
(3,)
```

Exemplo incompatível:

```text
(3,)
(2,)
```

---

# 203. Operações lógicas vetorizadas

Além das operações matemáticas, também podemos aplicar lógica.

```python
notas = np.array([
    5.0,
    6.5,
    7.0,
    8.5,
    9.0
])
```

Criando uma condição:

```python
aprovados = notas >= 7

print(aprovados)
```

Resultado:

```text
[False False  True  True  True]
```

Isso também é uma operação vetorizada.

---

# 204. Funções universais do NumPy

O NumPy possui funções chamadas de `ufuncs`.

São funções que operam elemento a elemento.

Exemplos:

```python
np.sqrt()
```

```python
np.abs()
```

```python
np.exp()
```

```python
np.log()
```

```python
np.round()
```

```python
np.maximum()
```

```python
np.minimum()
```

---

# 205. Valor absoluto

```python
valores = np.array([
    -10,
    -5,
    0,
    5,
    10
])

print(np.abs(valores))
```

Resultado:

```text
[10  5  0  5 10]
```

---

# 206. Arredondamento

```python
valores = np.array([
    7.126,
    8.789,
    9.444
])

print(
    np.round(
        valores,
        2
    )
)
```

Resultado:

```text
[7.13 8.79 9.44]
```

---

# 207. Limites com `maximum()` e `minimum()`

```python
notas = np.array([
    4.0,
    6.0,
    8.0,
    12.0
])
```

Limitando o máximo:

```python
print(
    np.minimum(
        notas,
        10
    )
)
```

Resultado:

```text
[ 4.  6.  8. 10.]
```

Limitando o mínimo:

```python
print(
    np.maximum(
        notas,
        5
    )
)
```

Resultado:

```text
[ 5.  6.  8. 12.]
```

---

# 208. Produto escalar

Considere:

```python
vetor_a = np.array([
    1,
    2,
    3
])

vetor_b = np.array([
    4,
    5,
    6
])
```

Produto escalar:

```python
resultado = np.dot(
    vetor_a,
    vetor_b
)

print(resultado)
```

Cálculo:

```text
1 × 4
+
2 × 5
+
3 × 6
```

Resultado:

```text
32
```

---

# 209. Conexão com Redes Neurais

Em um neurônio artificial, as entradas são multiplicadas por pesos.

Representação:

```text
x1 × peso1
+
x2 × peso2
+
x3 × peso3
```

Essa operação é semelhante a um produto escalar.

Exemplo:

```python
entradas = np.array([
    0.5,
    0.8,
    0.2
])

pesos = np.array([
    0.4,
    0.7,
    0.1
])

resultado = np.dot(
    entradas,
    pesos
)

print(resultado)
```

---

# 210. Adicionando o bias

Um neurônio também pode utilizar um valor chamado `bias`.

```python
bias = 0.2
```

Resultado do neurônio:

```python
saida = np.dot(
    entradas,
    pesos
) + bias

print(saida)
```

Não construiremos uma Rede Neural nesta aula.

O objetivo é perceber que operações estudadas com NumPy aparecem na base matemática das Redes Neurais.

---

# 211. Multiplicação de matrizes

A multiplicação elemento a elemento utiliza:

```python
A * B
```

A multiplicação matricial utiliza:

```python
A @ B
```

ou:

```python
np.matmul(A, B)
```

Exemplo:

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])
```

Elemento a elemento:

```python
print(A * B)
```

Resultado:

```text
[[ 5 12]
 [21 32]]
```

Multiplicação matricial:

```python
print(A @ B)
```

Resultado:

```text
[[19 22]
 [43 50]]
```

---

# 212. Por que distinguir essas operações?

Em Machine Learning e Redes Neurais, a multiplicação matricial é muito importante.

Dados podem ser organizados em matrizes.

Pesos também podem ser organizados em matrizes.

O processamento pode envolver:

```text
matriz de entradas
×
matriz de pesos
```

Por isso, é importante não confundir:

```python
A * B
```

com:

```python
A @ B
```

---

# 213. Exemplo de várias amostras e pesos

```python
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
```

Temos:

```text
2 amostras
3 features
```

Pesos:

```python
pesos = np.array([
    [0.2],
    [0.5],
    [0.1]
])
```

Formato:

```text
3 pesos
1 saída
```

Multiplicação:

```python
resultado = X @ pesos

print(resultado)
```

Formato do resultado:

```text
(2, 1)
```

Cada amostra produziu uma saída.

---

# 214. Verificando os formatos antes da multiplicação

```python
print("X:", X.shape)
print("Pesos:", pesos.shape)
```

Resultado:

```text
X: (2, 3)
Pesos: (3, 1)
```

As dimensões internas são compatíveis:

```text
(2, 3) × (3, 1)
```

O resultado será:

```text
(2, 1)
```

---

# 215. Erro de multiplicação matricial

Considere:

```python
X.shape
```

```text
(2, 3)
```

e:

```python
pesos_errados.shape
```

```text
(2, 1)
```

Ao tentar:

```python
X @ pesos_errados
```

será gerado um erro porque:

```text
3 é diferente de 2
```

As dimensões internas precisam ser iguais.

---

# 216. Regra da multiplicação matricial

Para multiplicar:

```text
(m, n) × (n, p)
```

o resultado terá:

```text
(m, p)
```

Exemplo:

```text
(2, 3) × (3, 1) = (2, 1)
```

---

# 217. Transposição

A transposição troca linhas por colunas.

```python
dados = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

Formato original:

```python
print(dados.shape)
```

Resultado:

```text
(2, 3)
```

Transpondo:

```python
dados_transpostos = dados.T

print(dados_transpostos)
```

Resultado:

```text
[[1 4]
 [2 5]
 [3 6]]
```

Formato:

```text
(3, 2)
```

---

# 218. Aplicação da transposição

A transposição pode ser utilizada para:

* reorganizar dados;
* ajustar formatos;
* realizar multiplicações matriciais;
* trocar linhas e colunas;
* preparar estruturas matemáticas.

Outra forma:

```python
np.transpose(dados)
```

---

# 219. Operações vetorizadas e desempenho

Operações vetorizadas costumam ser mais eficientes do que laços escritos manualmente em Python.

Exemplo com laço:

```python
resultado = []

for valor in notas:
    resultado.append(valor * 2)
```

Exemplo vetorizado:

```python
resultado = notas * 2
```

O segundo código é:

* menor;
* mais legível;
* mais próximo da matemática;
* geralmente mais rápido.

---

# 220. Medindo o tempo

Em um notebook, podemos utilizar:

```python
import time
```

Criando dados:

```python
valores = list(
    range(1_000_000)
)
```

Com lista:

```python
inicio = time.time()

resultado_lista = []

for valor in valores:
    resultado_lista.append(
        valor * 2
    )

fim = time.time()

print(
    "Tempo com lista:",
    fim - inicio
)
```

Com NumPy:

```python
array = np.array(valores)

inicio = time.time()

resultado_numpy = array * 2

fim = time.time()

print(
    "Tempo com NumPy:",
    fim - inicio
)
```

Os resultados podem variar conforme o ambiente.

---

# 221. Cuidado ao interpretar desempenho

NumPy tende a ser mais eficiente em grandes operações numéricas.

Entretanto:

* arrays pequenos podem não apresentar diferença perceptível;
* o tempo varia entre máquinas;
* o ambiente Colab pode oscilar;
* criar o array também possui custo;
* nem todo problema precisa ser resolvido com NumPy.

O objetivo é entender que NumPy foi projetado para cálculos numéricos em grande escala.

---

# 222. Prática guiada — Estatística dos alunos

Considere:

```python
alunos = np.array([
    [18, 8.5, 95],
    [19, 6.0, 75],
    [20, 9.0, 98],
    [21, 7.5, 85],
    [22, 5.0, 60]
])
```

Extraia as colunas:

```python
idades = alunos[:, 0]
notas = alunos[:, 1]
frequencias = alunos[:, 2]
```

Calcule:

```python
print("Média das idades:", np.mean(idades))
print("Média das notas:", np.mean(notas))
print("Média das frequências:", np.mean(frequencias))
```

Depois:

```python
print("Menor nota:", np.min(notas))
print("Maior nota:", np.max(notas))
print("Mediana das notas:", np.median(notas))
print("Desvio padrão:", np.std(notas))
```

---

# 223. Prática guiada — Estatística por coluna

```python
medias = np.mean(
    alunos,
    axis=0
)

minimos = np.min(
    alunos,
    axis=0
)

maximos = np.max(
    alunos,
    axis=0
)

desvios = np.std(
    alunos,
    axis=0
)
```

Exiba:

```python
print("Médias:", medias)
print("Mínimos:", minimos)
print("Máximos:", maximos)
print("Desvios:", desvios)
```

---

# 224. Prática guiada — Aumento das notas

Crie uma cópia:

```python
alunos_ajustados = alunos.copy()
```

Aumente as notas:

```python
alunos_ajustados[:, 1] += 0.5
```

Limite:

```python
alunos_ajustados[:, 1] = np.clip(
    alunos_ajustados[:, 1],
    0,
    10
)
```

Exiba:

```python
print(alunos_ajustados)
```

---

# 225. Prática guiada — Centralização

```python
medias = np.mean(
    alunos,
    axis=0
)

alunos_centralizados = (
    alunos - medias
)

print(alunos_centralizados)
```

Verifique:

```python
print(
    np.mean(
        alunos_centralizados,
        axis=0
    )
)
```

---

# 226. Prática guiada — Normalização

```python
minimos = np.min(
    alunos,
    axis=0
)

maximos = np.max(
    alunos,
    axis=0
)

alunos_normalizados = (
    alunos - minimos
) / (
    maximos - minimos
)

print(alunos_normalizados)
```

---

# 227. Comparando dados originais e normalizados

Dados originais:

```python
print(alunos)
```

Dados normalizados:

```python
print(
    np.round(
        alunos_normalizados,
        2
    )
)
```

Os dados normalizados ficarão entre 0 e 1.

---

# 228. Atividade prática 1 — Operações básicas

Considere:

```python
valores = np.array([
    10,
    20,
    30,
    40,
    50
])
```

Resolva:

1. Some 5 a todos os valores.
2. Subtraia 10 de todos os valores.
3. Multiplique todos os valores por 3.
4. Divida todos os valores por 10.
5. Eleve todos os valores ao quadrado.
6. Calcule a raiz quadrada.
7. Calcule a soma total.
8. Calcule a média.
9. Encontre o menor valor.
10. Encontre o maior valor.

---

# 229. Atividade prática 2 — Estatística das notas

Considere:

```python
notas = np.array([
    4.5,
    5.5,
    6.0,
    7.0,
    8.5,
    9.0,
    10.0
])
```

Calcule:

1. soma;
2. média;
3. mediana;
4. mínimo;
5. máximo;
6. amplitude;
7. desvio padrão;
8. variância;
9. primeiro quartil;
10. terceiro quartil.

---

# 230. Atividade prática 3 — Eixos

Considere:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.0, 70],
    [20, 9.0, 98],
    [21, 7.0, 85]
])
```

Resolva:

1. Calcule a média de todos os valores.
2. Calcule a média por coluna.
3. Calcule a soma por coluna.
4. Calcule o mínimo por coluna.
5. Calcule o máximo por coluna.
6. Calcule o desvio padrão por coluna.
7. Explique o significado de cada resultado.
8. Calcule a média por linha.
9. Explique por que a média por linha não é adequada nesse exemplo.
10. Extraia somente nota e frequência e calcule as médias.

---

# 231. Atividade prática 4 — Broadcasting

Utilize:

```python
dados = np.array([
    [18, 8.0, 90],
    [19, 6.5, 75],
    [20, 9.0, 98]
])
```

Resolva:

1. Some 1 a toda a matriz.
2. Explique por que essa operação pode não fazer sentido.
3. Crie o vetor:

```python
ajustes = np.array([
    0,
    0.5,
    2
])
```

4. Aplique os ajustes.
5. Explique o que aconteceu em cada coluna.
6. Calcule as médias de cada coluna.
7. Subtraia as médias da matriz.
8. Verifique se as novas médias estão próximas de zero.

---

# 232. Atividade prática 5 — Produto escalar

Considere:

```python
entradas = np.array([
    0.5,
    0.8,
    0.2
])

pesos = np.array([
    0.4,
    0.7,
    0.1
])
```

Resolva:

1. Multiplique os arrays elemento a elemento.
2. Some os resultados.
3. Faça o mesmo cálculo com `np.dot()`.
4. Compare os resultados.
5. Adicione um bias de `0.2`.
6. Explique como esse cálculo se relaciona com um neurônio artificial.

---

# 233. Desafio — Preparação numérica de dados escolares

Considere:

```python
dados = np.array([
    [16, 5.0, 60, 2],
    [17, 7.5, 85, 8],
    [18, 8.0, 90, 10],
    [19, 6.0, 72, 5],
    [20, 9.0, 98, 12],
    [21, 4.5, 55, 1],
    [22, 7.0, 80, 7],
    [23, 8.5, 92, 11]
])
```

Colunas:

```text
0 → idade
1 → nota
2 → frequência
3 → atividades entregues
```

---

## 233.1 Tarefas

1. Exiba o `shape`.
2. Exiba o `dtype`.
3. Calcule as médias de cada coluna.
4. Calcule os valores mínimos.
5. Calcule os valores máximos.
6. Calcule os desvios padrão.
7. Identifique o aluno com maior nota.
8. Identifique o aluno com menor frequência.
9. Ordene os alunos pela nota.
10. Ordene os alunos pela frequência.
11. Normalize todas as colunas entre 0 e 1.
12. Padronize todas as colunas.
13. Verifique as médias após a padronização.
14. Verifique os desvios após a padronização.
15. Crie um target de risco escolar.

Considere aluno em risco quando:

```text
nota < 6
ou
frequência < 70
ou
atividades < 4
```

---

# 234. Criando o target do desafio

```python
notas = dados[:, 1]
frequencias = dados[:, 2]
atividades = dados[:, 3]
```

Máscara:

```python
risco = (
    (notas < 6)
    |
    (frequencias < 70)
    |
    (atividades < 4)
)
```

Target:

```python
y = risco.astype(int)
```

Verifique:

```python
print(y)
print(y.shape)
```

---

# 235. Comparando o dataset original e transformado

Dataset original:

```python
X_original = dados.copy()
```

Dataset normalizado:

```python
X_normalizado = (
    dados - np.min(
        dados,
        axis=0
    )
) / (
    np.max(
        dados,
        axis=0
    )
    -
    np.min(
        dados,
        axis=0
    )
)
```

Compare:

```python
print("Original:")
print(X_original)

print("Normalizado:")
print(
    np.round(
        X_normalizado,
        2
    )
)
```

---

# 236. Reflexão sobre a transformação

Responda:

1. Os alunos mudaram após a normalização?
2. O significado das features mudou?
3. A ordem dos alunos mudou?
4. Os valores passaram a ter a mesma escala?
5. Por que isso pode ser importante para KNN?
6. Por que isso pode ser importante para Redes Neurais?
7. A Árvore de Decisão precisa obrigatoriamente dessa transformação?
8. Devemos normalizar antes ou depois de separar treino e teste?
9. Qual risco existe ao usar informações do conjunto de teste na preparação?
10. O que é vazamento de dados?

---

# 237. Introdução ao vazamento de dados

Vazamento de dados, ou `data leakage`, acontece quando informações que não deveriam estar disponíveis durante o treinamento influenciam o modelo.

Exemplo incorreto:

```text
1. Calcular média usando todos os dados.
2. Separar treino e teste.
3. Utilizar essa média na transformação.
```

Nesse caso, o conjunto de teste influenciou a transformação.

---

# 238. Ordem recomendada

Uma sequência mais adequada é:

```text
Dados
  ↓
Separação entre treino e teste
  ↓
Calcular parâmetros no treino
  ↓
Transformar o treino
  ↓
Aplicar os mesmos parâmetros no teste
```

Exemplo:

```text
média do treino
desvio padrão do treino
mínimo do treino
máximo do treino
```

Esses valores são usados para transformar treino e teste.

---

# 239. Conexão com Scikit-Learn

Na próxima aula, utilizaremos ferramentas como:

```python
from sklearn.preprocessing import StandardScaler
```

e:

```python
from sklearn.preprocessing import MinMaxScaler
```

Essas ferramentas ajudam a:

* calcular os parâmetros no treino;
* aplicar as mesmas transformações no teste;
* organizar o pré-processamento;
* reduzir erros;
* evitar vazamento de dados.

---

# 240. Conexão com KNN

O KNN utiliza distância entre exemplos.

Por isso, a escala das features influencia o resultado.

Considere:

```text
idade → 16 a 60
nota → 0 a 10
frequência → 0 a 100
```

Sem transformação, a frequência pode dominar o cálculo da distância.

Na aula de KNN, retomaremos:

* normalização;
* padronização;
* distância;
* escolha do número de vizinhos.

---

# 241. Conexão com Deep Learning

Redes Neurais também realizam muitos cálculos com matrizes.

Os dados entram em formato numérico.

Pesos são armazenados em matrizes.

O treinamento envolve:

* multiplicações;
* somas;
* produtos escalares;
* operações vetorizadas;
* transformação de matrizes.

Um fluxo simplificado:

```text
Matriz de entradas
        ↓
Multiplicação por pesos
        ↓
Soma do bias
        ↓
Função de ativação
        ↓
Saída
```

---

# 242. O que já conseguimos compreender

Mesmo sem construir uma Rede Neural, já podemos compreender a base numérica de um neurônio:

```python
saida = (
    np.dot(
        entradas,
        pesos
    )
    +
    bias
)
```

Esse cálculo utiliza conceitos estudados nesta aula:

* arrays;
* vetores;
* multiplicação;
* soma;
* produto escalar;
* operações vetorizadas.

---

# 243. Erros comuns nesta etapa

## Erro 1 — Misturar features sem analisar a escala

```python
np.mean(dados)
```

Pode misturar idade, nota e frequência sem significado.

---

## Erro 2 — Utilizar o eixo incorreto

```python
np.mean(dados, axis=1)
```

pode produzir médias por aluno quando o objetivo era média por feature.

---

## Erro 3 — Confundir `*` com `@`

```python
A * B
```

é multiplicação elemento a elemento.

```python
A @ B
```

é multiplicação matricial.

---

## Erro 4 — Dividir por desvio padrão zero

Isso pode gerar:

```text
nan
inf
```

---

## Erro 5 — Normalizar antes de separar treino e teste

Isso pode gerar vazamento de dados.

---

## Erro 6 — Alterar o array original sem perceber

```python
dados[:, 1] += 1
```

modifica diretamente o array.

Para preservar:

```python
dados_novos = dados.copy()
```

---

## Erro 7 — Aplicar operação sem considerar o significado

Somar o mesmo valor em idade, nota e frequência pode não fazer sentido.

---

# 244. Checklist de compreensão da Parte 4

* [ ] Sei realizar soma em arrays.
* [ ] Sei realizar subtração.
* [ ] Sei realizar multiplicação.
* [ ] Sei realizar divisão.
* [ ] Sei utilizar potência.
* [ ] Sei calcular média.
* [ ] Sei calcular mediana.
* [ ] Sei encontrar mínimo e máximo.
* [ ] Sei calcular amplitude.
* [ ] Sei calcular desvio padrão.
* [ ] Sei calcular variância.
* [ ] Sei utilizar `axis=0`.
* [ ] Sei utilizar `axis=1`.
* [ ] Sei explicar a diferença entre operação por linha e por coluna.
* [ ] Sei utilizar `argmin()` e `argmax()`.
* [ ] Sei ordenar arrays.
* [ ] Sei explicar broadcasting.
* [ ] Sei centralizar dados.
* [ ] Sei normalizar entre 0 e 1.
* [ ] Sei padronizar dados.
* [ ] Sei explicar por que a escala importa.
* [ ] Sei diferenciar `*` e `@`.
* [ ] Sei calcular produto escalar.
* [ ] Sei relacionar produto escalar a um neurônio artificial.
* [ ] Sei explicar por que treino e teste devem ser separados antes da transformação.
* [ ] Entendo o conceito inicial de vazamento de dados.

---

# 245. Resumo da Parte 4

Nesta parte, aprendemos:

* operações vetorizadas;
* operações entre arrays;
* soma, subtração, multiplicação e divisão;
* potência e raiz quadrada;
* funções estatísticas;
* média e mediana;
* mínimo e máximo;
* amplitude;
* desvio padrão;
* variância;
* operações por eixo;
* médias por coluna;
* médias por linha;
* índices de valores mínimos e máximos;
* ordenação;
* percentis;
* broadcasting;
* centralização;
* padronização;
* normalização;
* produto escalar;
* multiplicação matricial;
* transposição;
* relação entre NumPy e Redes Neurais;
* relação entre escala e KNN;
* introdução ao vazamento de dados.

---

# 246. Encerramento da Aula 04

Nesta aula, compreendemos como os dados são organizados numericamente antes de serem utilizados por um modelo de Machine Learning.

Estudamos:

* arrays NumPy;
* vetores e matrizes;
* tipos de dados;
* dimensões e formatos;
* acesso a linhas e colunas;
* filtros e máscaras booleanas;
* operações vetorizadas;
* estatísticas;
* eixos;
* broadcasting;
* normalização;
* padronização;
* conversão entre Pandas e NumPy;
* preparação de dados para algoritmos de Machine Learning.

Esses conceitos serão importantes nas próximas aulas, especialmente no estudo de algoritmos baseados em distância, como o KNN.

Na próxima aula, aprenderemos a reorganizar e combinar arrays. Também iniciaremos o estudo de estruturas com mais dimensões, como imagens e tensores.

---

# Próxima aula

## Aula 05 — Reorganização de Dados e Introdução aos Tensores

Na Aula 05, estudaremos:

* alteração de formato com `reshape()`;
* transformação de matrizes em vetores;
* combinação de arrays;
* divisão de arrays;
* transposição;
* arrays tridimensionais;
* representação numérica de imagens;
* canais de cores;
* lotes de imagens;
* introdução aos tensores;
* formatos utilizados em Deep Learning.

Esses conteúdos criarão uma ponte entre o Machine Learning tradicional e a disciplina de Redes Neurais.

© @karizeviecelli - 2026

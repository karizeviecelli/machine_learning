<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 01 – Preparando os Dados para o Modelo

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 01 – Preparando os Dados para o Modelo

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender como um algoritmo recebe os dados.
- Identificar quais colunas serão utilizadas para o treinamento.
- Separar Features e Target.
- Preparar o dataset para criar o primeiro modelo de Machine Learning.

---

# 📍 Onde estamos?

Na Aula 02 realizamos uma limpeza completa dos dados.

✔ Corrigimos valores nulos.

✔ Removemos registros duplicados.

✔ Padronizamos textos.

✔ Corrigimos tipos de dados.

✔ Corrigimos valores inválidos.

Ao final da aula geramos o arquivo:

```text
alunos_tratado.csv
```

Agora esse arquivo será utilizado para ensinar um computador.

---

# Situação-Problema

Imagine que a direção da escola fez o seguinte pedido.

> "Gostaríamos de prever se um novo aluno possui chances de ser aprovado."

Mas existe uma pergunta importante.

Como o computador descobrirá isso?

A resposta está nos dados.

---

# Como um computador aprende?

Imagine um professor corrigindo provas.

Durante vários anos ele observa:

- idade dos alunos;
- quantidade de faltas;
- notas;
- frequência.

Depois de analisar centenas de alunos.

Ele começa a perceber padrões.

O computador faz exatamente a mesma coisa.

Ele observa exemplos.

Depois aprende.

---

# O papel do dataset

Observe nossa tabela.

| Idade | Nota | Frequência | Aprovado |
|------:|------:|-----------:|-----------|
| 18 | 8.5 | 95 | Sim |
| 20 | 6.2 | 70 | Não |
| 19 | 9.0 | 96 | Sim |
| 22 | 5.8 | 65 | Não |

Cada linha representa um exemplo.

Quanto mais exemplos.

Melhor o computador poderá aprender.

---

# O que o algoritmo precisa aprender?

Queremos responder apenas uma pergunta.

```
Será aprovado?
```

Essa resposta recebe um nome muito importante.

---

# Target

Em Machine Learning chamamos essa resposta de:

## Target

Também conhecido como:

- variável alvo;
- variável resposta;
- classe.

É exatamente aquilo que desejamos prever.

No nosso exemplo.

```text
Aprovado
```

é o Target.

---

# Features

Agora pense.

Como o computador descobrirá se um aluno será aprovado?

Ele observará algumas informações.

No nosso exemplo.

- Idade
- Nota
- Frequência

Essas informações recebem o nome de:

## Features

São as características utilizadas pelo algoritmo para aprender.

---

# Resumindo

| Coluna | Tipo |
|----------|------|
| Idade | Feature |
| Nota | Feature |
| Frequência | Feature |
| Aprovado | Target |

Essa separação é utilizada em praticamente todos os algoritmos de Machine Learning.

---

# Abrindo o dataset

Primeiro vamos importar o Pandas.

```python
import pandas as pd
```

Agora vamos abrir o arquivo.

```python
df = pd.read_csv("alunos_tratado.csv")
```

Visualize o DataFrame.

```python
df.head()
```

Confira se tudo está correto.

---

# Separando as Features

Agora vamos criar uma variável chamada:

```python
X
```

Por convenção.

A letra **X** representa as entradas do algoritmo.

No nosso exemplo.

```python
X = df[["Idade", "Nota", "Frequencia"]]
```

Observe.

Estamos selecionando apenas as colunas que ajudam o computador a aprender.

---

# Visualizando as Features

Execute.

```python
X.head()
```

Resultado esperado.

| Idade | Nota | Frequência |
|------:|------:|-----------:|
| 18 | 8.5 | 95 |
| 20 | 6.2 | 70 |
| 19 | 9.0 | 96 |

Perceba.

A coluna **Aprovado** não aparece.

---

# Separando o Target

Agora criaremos outra variável.

```python
y
```

Por convenção.

A letra **y** representa a resposta correta.

```python
y = df["Aprovado"]
```

---

# Visualizando o Target

Execute.

```python
y.head()
```

Resultado.

```text
0    Sim

1    Não

2    Sim

3    Não
```

Agora temos exatamente aquilo que queremos ensinar ao computador.

---

# Comparando

## Features

```python
X
```

São as perguntas.

```
Idade

Nota

Frequência
```

---

## Target

```python
y
```

É a resposta.

```
Aprovado
```

---

# Analogia

Imagine um professor aplicando uma prova.

As perguntas representam:

```
Features
```

O gabarito representa:

```
Target
```

O algoritmo aprende comparando as perguntas com o gabarito.

---

# Laboratório 01

Abra o arquivo.

```text
alunos_tratado.csv
```

Execute.

```python
import pandas as pd

df = pd.read_csv("alunos_tratado.csv")
```

Depois.

```python
df.head()
```

Agora crie.

```python
X
```

e

```python
y
```

Visualize os dois.

---

# Laboratório 02

Responda.

Quais colunas poderiam ser utilizadas como Features?

Existe alguma coluna que não faria sentido utilizar?

Explique sua resposta.

---

# ⚠️ Erro comum

Muitos iniciantes fazem.

```python
X = df
```

Isso faz com que a coluna:

```
Aprovado
```

também seja enviada ao algoritmo.

Nesse caso.

O computador teria acesso à resposta antes do treinamento.

O modelo aprenderia de forma incorreta.

Sempre remova o Target das Features.

---

# 💼 No mercado

Uma das decisões mais importantes em um projeto de Machine Learning é escolher corretamente as Features.

Nem toda informação disponível deve ser utilizada.

Selecionar boas Features pode melhorar significativamente o desempenho do modelo.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ O que são Features.

✅ O que é Target.

✅ Como abrir o dataset tratado.

✅ Como separar os dados para treinamento.

Agora temos tudo preparado para criar nosso primeiro modelo de Machine Learning.

---

# 🚀 Preparando o próximo módulo

No próximo módulo conheceremos a biblioteca mais utilizada em Machine Learning com Python.

A **Scikit-Learn**.

Você aprenderá:

- como importar um algoritmo;
- o que é uma Árvore de Decisão;
- como criar seu primeiro modelo utilizando:

```python
from sklearn.tree import DecisionTreeClassifier
```

Pela primeira vez, faremos o computador aprender com os nossos dados.

---

**© @karizeviecelli - 2026**


<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 02 – Criando o Primeiro Modelo

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 02 – Criando o Primeiro Modelo de Machine Learning

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Conhecer a biblioteca Scikit-Learn.
- Importar um algoritmo de Machine Learning.
- Criar seu primeiro modelo.
- Entender o que acontece durante o treinamento.
- Utilizar o método `fit()`.

---

# 📍 Onde estamos?

No módulo anterior preparamos nossos dados.

Agora temos duas variáveis.

```python
X
```

Contém as Features.

```python
y
```

Contém o Target.

Agora chegou a hora de ensinar o computador.

---

# Conhecendo o Scikit-Learn

Até agora utilizamos apenas uma biblioteca.

```python
Pandas
```

Ela nos ajudou a trabalhar com tabelas.

Agora conheceremos outra biblioteca muito importante.

```
Scikit-Learn
```

Ela já possui diversos algoritmos de Machine Learning prontos para serem utilizados.

---

# O que existe dentro do Scikit-Learn?

Entre os principais algoritmos estão:

- Árvore de Decisão
- Regressão Linear
- KNN
- Random Forest
- SVM
- Naive Bayes

Neste curso começaremos pela Árvore de Decisão.

Porque ela é simples de entender e muito utilizada no aprendizado inicial.

---

# O que é uma Árvore de Decisão?

Imagine que você deseja descobrir se um aluno será aprovado.

Você começa fazendo perguntas.

```
A nota é maior que 7?

↓

SIM

↓

A frequência é maior que 75%?

↓

SIM

↓

Aprovado
```

Agora imagine outra situação.

```
Nota maior que 7?

↓

NÃO

↓

Reprovado
```

O computador faz exatamente isso.

Ele cria uma sequência de perguntas até chegar a uma resposta.

Por isso o algoritmo recebe o nome de:

**Árvore de Decisão.**

---

# Analogia

Imagine um jogo de adivinhação.

```
É um animal?

↓

Sim

↓

Tem asas?

↓

Não

↓

Tem quatro patas?

↓

Sim

↓

É um cachorro.
```

A Árvore de Decisão funciona de maneira muito parecida.

Cada pergunta reduz as possibilidades até encontrar a resposta.

---

# Importando o algoritmo

Vamos importar nosso primeiro algoritmo.

```python
from sklearn.tree import DecisionTreeClassifier
```

---

# Entendendo linha por linha

Observe.

```python
from
```

Significa.

```
Da biblioteca...
```

---

```python
sklearn
```

É o nome da biblioteca.

---

```python
tree
```

É o módulo que contém algoritmos baseados em árvores.

---

```python
import
```

Significa importar.

---

```python
DecisionTreeClassifier
```

É a classe responsável por criar uma Árvore de Decisão para problemas de classificação.

---

# Criando o modelo

Agora vamos construir nosso modelo.

```python
modelo = DecisionTreeClassifier()
```

---

# O que aconteceu?

Ainda não treinamos nada.

Criamos apenas um objeto.

Pense nele como um aluno que acabou de entrar na escola.

Ele ainda não estudou.

Ainda não aprendeu.

Ainda não sabe responder nenhuma pergunta.

---

# Analogia

Imagine comprar um computador novo.

Ele acabou de sair da caixa.

Ainda não possui seus arquivos.

Ainda não possui programas instalados.

Nosso modelo está exatamente assim.

Ele existe.

Mas ainda não sabe nada.

---

# Treinando o modelo

Agora vem uma das linhas mais importantes de todo o curso.

```python
modelo.fit(X, y)
```

---

# O que significa fit?

Em inglês.

```
fit

↓

ajustar
```

Nesse momento o algoritmo observa:

```python
X
```

e compara com:

```python
y
```

Tentando descobrir padrões.

---

# O que acontece durante o fit()?

Imagine nossa tabela.

| Idade | Nota | Frequência | Aprovado |
|------:|------:|-----------:|-----------|
|18|8.5|95|Sim|
|20|6.0|70|Não|
|22|9.2|98|Sim|

O algoritmo analisa cada linha.

Ele procura responder.

> Quais características aparecem com mais frequência entre os alunos aprovados?

Depois faz a mesma análise para os alunos reprovados.

A partir dessas observações ele cria sua Árvore de Decisão.

---

# Importante

O método:

```python
fit()
```

não mostra nenhuma saída.

Isso é normal.

Quando o treinamento termina sem erros.

O modelo já aprendeu.

---

# Código completo

```python
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("alunos_tratado.csv")

X = df[["Idade", "Nota", "Frequencia"]]

y = df["Aprovado"]

modelo = DecisionTreeClassifier()

modelo.fit(X, y)
```

Se tudo estiver correto.

Nenhuma mensagem será exibida.

Isso significa que o treinamento foi realizado com sucesso.

---

# Laboratório 01

Abra o arquivo.

```
alunos_tratado.csv
```

Execute todo o código apresentado.

Ao final.

Responda.

O modelo exibiu alguma mensagem?

Se não.

O treinamento foi concluído corretamente.

---

# Laboratório 02

Altere uma das Features.

Por exemplo.

Inclua.

```python
Curso
```

Observe o que acontece.

O algoritmo conseguiu treinar?

Por quê?

Discuta sua resposta com os colegas.

---

# ⚠️ Erro comum

Muitos iniciantes tentam utilizar colunas de texto diretamente.

Por exemplo.

```python
Curso
```

Como o algoritmo trabalha com números.

Ele normalmente apresentará erro.

Mais adiante aprenderemos como transformar textos em números.

---

# 💼 No mercado

Grande parte dos projetos de Machine Learning utiliza a biblioteca Scikit-Learn.

Ela oferece dezenas de algoritmos prontos e uma interface muito parecida entre eles.

Por isso, ao aprender um algoritmo, fica mais fácil aprender os demais.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ O que é o Scikit-Learn.

✅ O que é uma Árvore de Decisão.

✅ Como importar um algoritmo.

✅ Como criar um modelo.

✅ Como utilizar `fit()`.

Agora nosso computador já aprendeu com os dados.

---

# 🚀 Preparando o próximo módulo

No próximo módulo faremos a pergunta mais importante.

> **"O modelo realmente aprendeu?"**

Você criará novos alunos e utilizará:

```python
modelo.predict()
```

para descobrir se eles possuem chances de aprovação.

Será a primeira previsão realizada pela sua Inteligência Artificial.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 03 – Fazendo a Primeira Previsão

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 03 – Fazendo a Primeira Previsão

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Utilizar o método `predict()`.
- Fazer previsões utilizando um modelo treinado.
- Interpretar o resultado retornado pelo algoritmo.
- Testar diferentes cenários.

---

# 📍 Onde estamos?

No módulo anterior criamos nosso primeiro modelo.

```python
modelo = DecisionTreeClassifier()
```

Depois treinamos utilizando.

```python
modelo.fit(X, y)
```

Agora chegou a hora de descobrir se ele realmente aprendeu.

---

# O que significa prever?

Imagine um professor experiente.

Depois de vários anos dando aula.

Ele observa um novo aluno.

Antes mesmo da prova ele pensa.

> "Acredito que esse aluno será aprovado."

Como ele chegou a essa conclusão?

Porque comparou o novo aluno com centenas de alunos anteriores.

Nosso algoritmo fará exatamente isso.

---

# Conhecendo o predict()

Para fazer previsões utilizamos.

```python
modelo.predict()
```

Em inglês.

```
Predict

↓

Prever
```

---

# Como funciona?

Imagine este novo aluno.

| Idade | Nota | Frequência |
|------:|------:|-----------:|
|20|8.5|92|

Queremos descobrir.

```
Será aprovado?
```

---

# Criando um novo aluno

No Python.

```python
novo_aluno = [[20, 8.5, 92]]
```

---

# Por que existem dois colchetes?

Essa é uma dúvida muito comum.

Observe.

Um aluno.

```python
[20, 8.5, 92]
```

É apenas uma lista.

Mas o Scikit-Learn espera receber uma tabela.

Mesmo contendo apenas um registro.

Visualmente.

| Idade | Nota | Frequência |
|------:|------:|-----------:|
|20|8.5|92|

Por isso utilizamos.

```python
[[20,8.5,92]]
```

Uma lista contendo outra lista.

---

# Fazendo a previsão

Agora basta executar.

```python
resultado = modelo.predict(novo_aluno)
```

---

# O que aconteceu?

O algoritmo comparou o novo aluno com todos os exemplos utilizados durante o treinamento.

Depois retornou sua previsão.

---

# Visualizando o resultado

```python
print(resultado)
```

Saída.

```text
['Sim']
```

Isso significa.

Segundo o algoritmo.

Esse aluno possui características semelhantes às de alunos aprovados.

---

# Outro exemplo

Agora vamos testar outro aluno.

```python
novo_aluno = [[19, 5.5, 62]]

resultado = modelo.predict(novo_aluno)

print(resultado)
```

Resultado.

```text
['Não']
```

Observe.

O modelo mudou sua resposta.

Porque os dados mudaram.

---

# Fazendo vários testes

Também podemos prever vários alunos ao mesmo tempo.

```python
novos_alunos = [

    [18, 9.2, 98],

    [20, 5.8, 70],

    [22, 7.5, 80],

    [19, 6.0, 65]

]
```

Agora.

```python
resultado = modelo.predict(novos_alunos)

print(resultado)
```

Resultado.

```text
['Sim'
 'Não'
 'Sim'
 'Não']
```

Cada posição corresponde a um aluno.

---

# Entendendo o resultado

Observe.

```python
resultado
```

É uma lista.

Podemos acessar cada previsão.

Primeiro aluno.

```python
resultado[0]
```

Segundo.

```python
resultado[1]
```

Terceiro.

```python
resultado[2]
```

---

# Exibindo uma mensagem amigável

Podemos utilizar um IF.

```python
if resultado[0] == "Sim":
    print("O aluno provavelmente será aprovado.")
else:
    print("O aluno provavelmente não será aprovado.")
```

Agora a resposta fica muito mais fácil de entender.

---

# Código completo

```python
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("alunos_tratado.csv")

X = df[["Idade", "Nota", "Frequencia"]]

y = df["Aprovado"]

modelo = DecisionTreeClassifier()

modelo.fit(X, y)

novo_aluno = [[20, 8.5, 92]]

resultado = modelo.predict(novo_aluno)

print(resultado)
```

---

# Laboratório 01

Utilizando o modelo treinado.

Faça previsões para os seguintes alunos.

Aluno 1

```text
Idade: 18

Nota: 9.5

Frequência: 98
```

Aluno 2

```text
Idade: 19

Nota: 5.5

Frequência: 60
```

Aluno 3

```text
Idade: 21

Nota: 7.2

Frequência: 82
```

Anote todas as respostas.

---

# Laboratório 02

Crie três novos alunos.

Escolha livremente.

- idade;
- nota;
- frequência.

Depois faça as previsões.

Os resultados fazem sentido?

Discuta com os colegas.

---

# Desafio

Crie uma pequena lista contendo cinco novos alunos.

Utilize apenas um comando.

```python
modelo.predict()
```

Para prever todos de uma única vez.

Depois monte uma tabela.

| Idade | Nota | Frequência | Previsão |
|------:|------:|-----------:|-----------|
|18|9.0|95|Sim|
|20|5.5|65|Não|
|...|...|...|...|

---

# ⚠️ Erro comum

Alguns alunos fazem.

```python
novo_aluno = [20,8.5,92]
```

Depois.

```python
modelo.predict(novo_aluno)
```

Resultado.

Erro.

Porque o algoritmo espera uma tabela.

Mesmo contendo apenas um registro.

O correto é.

```python
novo_aluno = [[20,8.5,92]]
```

---

# Outro erro comum

As Features devem aparecer exatamente na mesma ordem utilizada no treinamento.

Treinamento.

```python
X = df[
    [
        "Idade",
        "Nota",
        "Frequencia"
    ]
]
```

Então.

A previsão também deve seguir.

```python
[[Idade, Nota, Frequencia]]
```

Nunca altere essa ordem.

---

# 💼 No mercado

Empresas utilizam exatamente esse processo.

Um novo cliente.

↓

O sistema coleta informações.

↓

O modelo faz uma previsão.

↓

O sistema toma uma decisão.

Isso acontece diariamente em:

- bancos;
- hospitais;
- seguradoras;
- lojas virtuais;
- aplicativos de transporte.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ Como utilizar `predict()`.

✅ Como criar novos registros.

✅ Como interpretar a resposta do modelo.

✅ Como realizar previsões para um ou vários registros.

---

# 🚀 Preparando o próximo módulo

Agora sabemos fazer previsões.

Mas surgiu uma nova pergunta.

> **Podemos confiar nessas respostas?**

No próximo módulo aprenderemos como medir a qualidade de um modelo utilizando um conjunto de teste.

Você conhecerá funções como:

```python
train_test_split()
```

e começará a entender como avaliar um modelo de Machine Learning antes de utilizá-lo em situações reais.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 04 – Separando Dados de Treino e Teste

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 04 – Separando Dados de Treino e Teste

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender por que não devemos treinar utilizando todos os dados.
- Separar um dataset em treino e teste.
- Utilizar a função `train_test_split()`.
- Preparar corretamente os dados para avaliar um modelo.

---

# 📍 Uma pergunta importante

Até agora fizemos isto.

```python
modelo.fit(X, y)
```

Depois.

```python
modelo.predict(...)
```

Tudo funcionou.

Mas...

**Como sabemos que o modelo realmente aprendeu?**

Talvez ele apenas tenha decorado os exemplos.

---

# Uma analogia

Imagine um professor preparando um aluno para uma prova.

Ele entrega exatamente as mesmas perguntas que cairão na avaliação.

O aluno estuda.

Chega o dia da prova.

Recebe exatamente as mesmas questões.

Tira nota 10.

Podemos dizer que ele realmente aprendeu?

Provavelmente não.

Ele apenas decorou as respostas.

---

# O mesmo acontece com o computador

Imagine este conjunto de dados.

| Idade | Nota | Frequência | Aprovado |
|------:|------:|-----------:|-----------|
|18|8.5|95|Sim|
|20|6.0|70|Não|
|19|9.2|98|Sim|
|22|5.8|65|Não|

Treinamos utilizando todos esses exemplos.

Depois fazemos a previsão utilizando exatamente os mesmos dados.

Naturalmente.

O resultado será muito bom.

Mas isso não significa que o modelo conseguirá prever novos alunos.

---

# Como resolver esse problema?

A solução é muito simples.

Dividimos o dataset em duas partes.

```text
Dataset

↓

Treino

↓

Teste
```

---

# Conjunto de treino

É utilizado para ensinar o algoritmo.

```
O computador aprende aqui.
```

---

# Conjunto de teste

É utilizado somente depois do treinamento.

Serve para responder uma pergunta.

```
O modelo consegue prever dados que nunca viu?
```

---

# Fluxo correto

```text
Dataset

↓

Separação

↓

Treino

↓

Modelo

↓

Teste

↓

Resultado
```

Esse é o fluxo utilizado em praticamente todos os projetos de Machine Learning.

---

# Conhecendo o train_test_split()

A biblioteca Scikit-Learn possui uma função pronta.

```python
from sklearn.model_selection import train_test_split
```

Ela faz toda a divisão automaticamente.

---

# Separando os dados

```python
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Calma.

Vamos entender linha por linha.

---

# O que significa test_size?

```python
test_size = 0.2
```

Significa.

```
20%
```

dos dados serão utilizados para teste.

Os outros.

```
80%
```

serão utilizados para treinamento.

---

# Exemplo

Imagine um dataset com.

```
100 alunos
```

A divisão ficará assim.

```
80 alunos

↓

Treinamento
```

```
20 alunos

↓

Teste
```

---

# O que significa random_state?

Observe.

```python
random_state = 42
```

O computador faz uma divisão aleatória.

Se não informarmos um valor.

Cada execução produzirá uma divisão diferente.

Ao utilizar.

```python
42
```

Garantimos que todos obterão exatamente a mesma divisão.

Isso facilita estudos e comparações.

---

# Visualizando os conjuntos

Quantidade utilizada para treinamento.

```python
X_treino.shape
```

Exemplo.

```text
(80,3)
```

---

Quantidade utilizada para teste.

```python
X_teste.shape
```

Resultado.

```text
(20,3)
```

---

# Treinando corretamente

Agora.

Em vez de utilizar.

```python
modelo.fit(X, y)
```

Utilizamos.

```python
modelo.fit(
    X_treino,
    y_treino
)
```

Observe.

O modelo aprende somente com os dados de treinamento.

---

# Fazendo previsões

Depois do treinamento.

Utilizamos.

```python
previsoes = modelo.predict(X_teste)
```

Agora.

As previsões serão realizadas utilizando registros que o modelo nunca viu.

É exatamente isso que queremos.

---

# Comparando

Antes.

```python
Treinamento

↓

Todos os dados
```

Agora.

```python
Treinamento

↓

80%
```

Depois.

```python
Teste

↓

20%
```

Muito mais confiável.

---

# Código completo

```python
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

df = pd.read_csv("alunos_tratado.csv")

X = df[["Idade", "Nota", "Frequencia"]]

y = df["Aprovado"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo = DecisionTreeClassifier()

modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
```

---

# Laboratório 01

Utilizando o arquivo.

```text
alunos_tratado.csv
```

Faça a divisão em:

- treino;
- teste.

Depois responda.

- Quantos registros ficaram no treino?

- Quantos ficaram no teste?

---

# Laboratório 02

Altere.

```python
test_size=0.3
```

Agora responda.

Quanto ficou para treino?

Quanto ficou para teste?

Repita utilizando:

```
0.4
```

---

# Desafio

Faça três divisões diferentes.

```python
0.1

0.2

0.3
```

Qual delas produziu o maior conjunto de treinamento?

Explique.

---

# ⚠️ Erro comum

Alguns iniciantes treinam utilizando:

```python
X
```

e depois testam utilizando:

```python
X
```

Isso faz com que o algoritmo seja avaliado utilizando exatamente os mesmos exemplos que utilizou para aprender.

O resultado normalmente parece excelente.

Mas não representa o desempenho em situações reais.

---

# 💼 No mercado

Modelos de Machine Learning são treinados para resolver problemas futuros.

Por isso.

Sempre devem ser avaliados utilizando dados que não participaram do treinamento.

Essa prática evita um problema conhecido como:

```
Overfitting
```

Por enquanto basta entender que um modelo pode decorar os exemplos em vez de aprender padrões.

Estudaremos esse conceito mais adiante.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ Por que dividir o dataset.

✅ O que é conjunto de treino.

✅ O que é conjunto de teste.

✅ Como utilizar `train_test_split()`.

✅ O significado de `test_size`.

✅ O significado de `random_state`.

---

# 🚀 Preparando o próximo módulo

Agora já sabemos separar os dados corretamente.

Mas ainda falta responder à pergunta mais importante.

> **O modelo acertou ou errou?**

No próximo módulo aprenderemos a medir a qualidade das previsões utilizando métricas como:

```python
accuracy_score()
```

e construiremos nossa primeira avaliação de um modelo de Machine Learning.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 05 – Avaliando o Modelo com Acurácia

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 05 – Avaliando o Modelo com Acurácia

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender por que um modelo precisa ser avaliado.
- Comparar previsões com os resultados reais.
- Utilizar a função `accuracy_score()`.
- Interpretar o resultado da acurácia.

---

# 📍 Onde estamos?

Até aqui nosso fluxo foi:

```
Dataset

↓

Limpeza

↓

Features e Target

↓

Treinamento

↓

Previsão
```

Agora precisamos descobrir se o modelo realmente aprendeu.

---

# Uma pergunta importante

Imagine que seu modelo respondeu:

```
Sim

Não

Sim

Não
```

Como saber se essas respostas estão corretas?

Precisamos compará-las com a realidade.

---

# Um exemplo

Imagine que o professor entregou uma lista com quatro alunos.

O modelo respondeu.

| Aluno | Previsão |
|--------|----------|
| Ana | Sim |
| João | Não |
| Pedro | Sim |
| Maria | Sim |

Agora vamos comparar com a realidade.

| Aluno | Realidade |
|--------|-----------|
| Ana | Sim |
| João | Não |
| Pedro | Não |
| Maria | Sim |

Observe.

O modelo acertou três respostas.

Errou apenas uma.

---

# O que é Acurácia?

Acurácia responde a uma pergunta muito simples.

> **De todas as previsões realizadas, quantas estavam corretas?**

É uma das métricas mais utilizadas para iniciar estudos em Machine Learning.

---

# Calculando manualmente

Imagine.

```
10 previsões
```

Dessas.

```
8 corretas
```

Então.

```
8 ÷ 10 = 0.80
```

Ou.

```
80%
```

Essa é a acurácia.

---

# A função accuracy_score()

O Scikit-Learn já possui uma função pronta.

Primeiro fazemos a importação.

```python
from sklearn.metrics import accuracy_score
```

---

# Utilizando a função

Depois das previsões.

```python
previsoes = modelo.predict(X_teste)
```

Calculamos.

```python
acuracia = accuracy_score(
    y_teste,
    previsoes
)
```

Observe a ordem dos parâmetros.

Primeiro.

```python
y_teste
```

São as respostas verdadeiras.

Depois.

```python
previsoes
```

São as respostas produzidas pelo modelo.

---

# Exibindo o resultado

```python
print(acuracia)
```

Resultado.

```text
0.90
```

O que significa?

```
90%
```

das previsões estavam corretas.

---

# Exibindo em porcentagem

Podemos deixar mais amigável.

```python
print(f"Acurácia: {acuracia:.2%}")
```

Resultado.

```text
Acurácia: 90.00%
```

Muito mais fácil de interpretar.

---

# Código completo

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.datasets import load_iris

# ESSA LINHA CRIA A VARIÁVEL 'dados':
dados = load_iris() 
df = pd.read_csv("alunos_tratado.csv")

X = df[["Idade", "Nota", "Frequencia"]]
y = df["Aprovado"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = DecisionTreeClassifier()
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia: {acuracia:.2%}")
```

---

# Interpretando o resultado

Imagine que seu modelo apresentou.

```
Acurácia = 100%
```

Significa que ele é perfeito?

Nem sempre.

Talvez o conjunto de teste seja pequeno.

Talvez os dados sejam muito simples.

Por isso.

A acurácia é apenas uma das formas de avaliar um modelo.

Nas próximas aulas conheceremos outras métricas.

---

# Laboratório 01

Execute todo o código.

Depois responda.

- Qual foi a acurácia obtida?

- O resultado ficou acima de 80%?

---

# Laboratório 02

Altere.

```python
test_size = 0.3
```

Treine novamente.

A acurácia mudou?

Por quê?

Discuta com a turma.

---

# Desafio

Faça três testes.

```python
random_state = 10
```

Depois.

```python
random_state = 20
```

Depois.

```python
random_state = 42
```

A acurácia permaneceu igual?

Explique.

---

# ⚠️ Erro comum

Alguns alunos fazem.

```python
accuracy_score(
    previsoes,
    y_teste
)
```

Embora muitas vezes o resultado seja o mesmo, a convenção correta é:

```python
accuracy_score(
    y_teste,
    previsoes
)
```

Primeiro.

Resposta correta.

Depois.

Resposta prevista.

---

# 💼 No mercado

Nenhum modelo é colocado em produção apenas porque conseguiu fazer previsões.

Antes disso.

Ele passa por uma avaliação utilizando diversas métricas.

A acurácia normalmente é a primeira delas.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ O que é acurácia.

✅ Como calcular a acurácia.

✅ Como utilizar `accuracy_score()`.

✅ Como interpretar o resultado.

Agora já sabemos não apenas fazer previsões.

Também sabemos medir a qualidade do nosso modelo.

---

# 🧪 Mini Projeto

Treine novamente o modelo utilizando:

```
70% treino

30% teste
```

Depois responda.

- Qual foi a acurácia?

- Você considera esse resultado satisfatório?

- Como poderíamos tentar melhorar esse modelo?

---

# 🚀 Preparando o próximo módulo

Até agora utilizamos apenas uma Árvore de Decisão.

Na próxima aula conheceremos outro algoritmo muito utilizado.

**K-Nearest Neighbors (KNN)**

Você descobrirá como um algoritmo pode tomar decisões analisando os exemplos mais parecidos com um novo registro e comparará seus resultados com os da Árvore de Decisão.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 06 – Visualizando a Árvore de Decisão

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 06 – Visualizando a Árvore de Decisão

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender como uma Árvore de Decisão toma decisões.
- Visualizar graficamente o modelo treinado.
- Interpretar os nós da árvore.
- Descobrir quais características são mais importantes.

---

# 📍 Onde estamos?

Já conseguimos:

✅ limpar os dados

✅ criar um modelo

✅ treinar

✅ fazer previsões

✅ calcular a acurácia

Mas ainda existe uma pergunta.

> **Como o computador chegou naquela resposta?**

Vamos descobrir.

---

# A Inteligência Artificial pensa?

Na verdade...

Não.

Ela apenas segue regras matemáticas.

Uma Árvore de Decisão faz isso através de perguntas.

Imagine uma árvore como esta.

```
               Nota > 7 ?

             /            \

          NÃO             SIM

     Reprovado      Frequência > 75 ?

                     /              \

                  NÃO              SIM

             Reprovado         Aprovado
```

Perceba.

Ela apenas faz perguntas.

Nada mais.

---

# O que é um nó?

Cada pergunta recebe o nome de:

```
Nó
```

Exemplo.

```
Nota > 7 ?
```

é um nó.

---

# O que são folhas?

Quando a árvore chega a uma resposta.

Temos uma folha.

```
Aprovado
```

ou

```
Reprovado
```

---

# Visualizando a árvore

O Scikit-Learn consegue desenhar a árvore.

Primeiro importamos.

```python
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

```

---

# Desenhando

Depois do treinamento.

```python
plt.figure(figsize=(15,8))

plot_tree(

    modelo,

    feature_names=dados.feature_names,

    class_names=dados.target_names,

    filled=True,

    rounded=True,

     fontsize=10
)

plt.show()

```

---

# O que significa cada parâmetro?

## modelo

É a árvore treinada.

---

## feature_names

Mostra os nomes das colunas.

Sem isso.

A árvore mostraria apenas números.

---

## class_names

Mostra os nomes das classes.

```
Não

Sim
```

---

## filled=True

Colora os nós.

Facilitando a leitura.

---

# O resultado

Você verá algo parecido com isto.

```
                Nota <= 7.2

               /            \

          Não             Frequência <= 74

                        /               \

                    Não               Sim
```

Observe.

A árvore criou perguntas automaticamente.

Nós não escrevemos nenhuma delas.

---

# Como ela escolheu essas perguntas?

Durante o treinamento.

O algoritmo analisou centenas de possibilidades.

Depois escolheu aquelas que melhor separavam:

```
Aprovados

↓

Reprovados
```

---

# Lendo uma árvore

Imagine esta.

```
Nota <= 7 ?

↓

Sim

↓

Reprovado
```

Significa.

Sempre que a nota for menor ou igual a sete.

O algoritmo acredita que o aluno será reprovado.

---

Outro caminho.

```
Nota > 7

↓

Frequência > 80

↓

Aprovado
```

Observe.

Agora são duas perguntas.

Quanto maior a árvore.

Mais perguntas existirão.

---

# As perguntas são fixas?

Não.

Cada dataset produz uma árvore diferente.

Se mudarmos os dados.

A árvore também muda.

---

# Laboratório 01

Execute.

```python
plt.figure(figsize=(15,8))

plot_tree(

    modelo,

    feature_names=X.columns,

    class_names=["Não","Sim"],

    filled=True

)

plt.show()
```

Observe.

Qual foi a primeira pergunta feita pela árvore?

---

# Laboratório 02

Treine novamente.

Agora altere.

```python
random_state
```

Depois desenhe novamente.

A árvore mudou?

Compare com seus colegas.

---

# Descobrindo a importância das Features

Existe outra informação muito interessante.

Execute.

```python
modelo.feature_importances_
```

Resultado.

```
[0.12

0.58

0.30]
```

Mas...

O que isso significa?

---

# Associando nomes

Faça.

```python
for coluna, importancia in zip(

    X.columns,

    modelo.feature_importances_

):

    print(coluna, importancia)
```

Resultado.

```
Idade 0.10

Nota 0.61

Frequência 0.29
```

---

# Interpretando

Observe.

```
Nota

↓

0.61
```

Foi a característica mais importante.

Já.

```
Idade

↓

0.10
```

Influenciou pouco.

---

# 💼 No mercado

Empresas utilizam essas informações para entender o comportamento do modelo.

Isso ajuda a responder perguntas como.

> O que mais influencia a aprovação dos alunos?

> Qual variável tem maior impacto?

Essa interpretação é muito importante quando o modelo será utilizado para apoiar decisões.

---

# ⚠️ Erro comum

Alguns alunos acreditam que:

```
0.61

↓

61%
```

Não.

Esses valores representam uma **importância relativa**.

Quanto maior.

Maior a influência daquela Feature.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ Como visualizar uma Árvore de Decisão.

✅ O que é um nó.

✅ O que é uma folha.

✅ Como interpretar perguntas da árvore.

✅ Como descobrir a importância das Features.

Agora você consegue enxergar como a Inteligência Artificial tomou suas decisões.

---

# 🧪 Mini Projeto

Desenhe a árvore.

Depois responda.

- Qual foi a primeira pergunta?

- Qual Feature foi considerada mais importante?

- Você concorda com essa decisão?

Explique.

---

# 🚀 Preparando a Aula 04

Até agora utilizamos apenas uma Árvore de Decisão.

Na próxima aula conheceremos um algoritmo completamente diferente.

O **K-Nearest Neighbors (KNN)**.

Você verá que, enquanto a Árvore toma decisões fazendo perguntas, o KNN decide observando os exemplos mais parecidos.

Compararemos os dois algoritmos utilizando exatamente o mesmo dataset.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 03 – Construindo o Primeiro Modelo de Machine Learning
Módulo 07 – Salvando e Reutilizando um Modelo

© @karizeviecelli - 2026
=========================================================
-->

# Aula 03
# Módulo 07 – Salvando e Reutilizando um Modelo

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender por que salvar um modelo treinado.
- Utilizar a biblioteca Joblib.
- Salvar um modelo em um arquivo.
- Carregar um modelo salvo.
- Fazer previsões sem treinar novamente.

---

# 📍 Uma pergunta importante

Imagine que você treinou um modelo.

O treinamento demorou:

```
3 horas
```

Agora imagine que amanhã um novo aluno precisa ser avaliado.

Você precisa treinar tudo novamente?

**Não.**

O modelo já aprendeu.

Basta reutilizá-lo.

---

# Como funciona?

O processo é semelhante ao que fazemos com um documento.

Você escreve um relatório.

↓

Salva no computador.

↓

Fecha o arquivo.

↓

No outro dia.

↓

Abre novamente.

O modelo de Machine Learning funciona exatamente assim.

---

# Fluxo de trabalho

```
Dataset

↓

Treinamento

↓

Modelo

↓

Salvar

↓

Arquivo
```

Mais tarde.

```
Arquivo

↓

Carregar

↓

Modelo

↓

Previsão
```

---

# Conhecendo o Joblib

Para salvar modelos utilizaremos a biblioteca:

```python
joblib
```

Ela é muito utilizada em projetos desenvolvidos com Scikit-Learn.

---

# Importando

```python
import joblib
```

---

# Salvando o modelo

Depois do treinamento.

```python
joblib.dump(
    modelo,
    "modelo_aprovacao.pkl"
)
```

---

# O que aconteceu?

Foi criado um arquivo.

```text
modelo_aprovacao.pkl
```

Esse arquivo contém tudo o que o algoritmo aprendeu.

---

# O que significa .pkl?

É a extensão de um arquivo serializado.

Serializar significa transformar um objeto da memória em um arquivo.

Assim ele pode ser armazenado e reutilizado posteriormente.

---

# Onde o arquivo será salvo?

No Google Colab.

Ele aparecerá na área de arquivos.

Também poderá ser baixado para o computador.

---

# Carregando o modelo

Imagine que amanhã queremos fazer novas previsões.

Não precisamos executar:

```python
modelo.fit()
```

Basta carregar.

```python
modelo = joblib.load(
    "modelo_aprovacao.pkl"
)
```

Pronto.

O modelo já está pronto para uso.

---

# Fazendo novas previsões

Agora podemos utilizar normalmente.

```python
novo_aluno = [[19, 8.2, 90]]

resultado = modelo.predict(
    novo_aluno
)

print(resultado)
```

Observe.

Nenhum treinamento foi realizado.

Mesmo assim.

O modelo respondeu.

---

# Código completo

```python
import pandas as pd

import joblib

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("alunos_tratado.csv")

X = df[
    [
        "Idade",
        "Nota",
        "Frequencia"
    ]
]

y = df["Aprovado"]

modelo = DecisionTreeClassifier()

modelo.fit(X, y)

joblib.dump(
    modelo,
    "modelo_aprovacao.pkl"
)
```

---

# Utilizando o modelo salvo

```python
import joblib

modelo = joblib.load(
    "modelo_aprovacao.pkl"
)

novo_aluno = [[20, 9.0, 96]]

resultado = modelo.predict(
    novo_aluno
)

print(resultado)
```

---

# Laboratório 01

Treine novamente o modelo.

Depois salve utilizando:

```python
joblib.dump()
```

Verifique se o arquivo apareceu na lista de arquivos do Google Colab.

---

# Laboratório 02

Reinicie o ambiente do Colab.

Importe apenas o Joblib.

Carregue o modelo salvo.

Faça uma nova previsão.

Perceba que não foi necessário treinar novamente.

---

# Desafio

Treine um modelo utilizando o dataset:

```text
clientes_tratado.csv
```

Salve-o com o nome.

```text
modelo_clientes.pkl
```

Depois carregue esse modelo e faça uma previsão.

---

# ⚠️ Erro comum

Alguns alunos tentam fazer previsões antes de carregar o modelo.

Por exemplo.

```python
resultado = modelo.predict(...)
```

Sem executar.

```python
modelo = joblib.load(...)
```

O Python informará que a variável não existe.

Sempre carregue o modelo antes de utilizá-lo.

---

# 💼 No mercado

Em aplicações reais.

O treinamento normalmente acontece apenas quando novos dados são disponibilizados.

O sistema utilizado pelos usuários trabalha apenas carregando o modelo salvo.

Isso torna as previsões muito mais rápidas.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ Como salvar um modelo.

✅ Como carregar um modelo.

✅ Como reutilizar um modelo treinado.

✅ Como fazer previsões utilizando um arquivo `.pkl`.

Agora você já conhece o ciclo completo de um modelo de Machine Learning.

---

# 🧪 Mini Projeto

Crie um sistema simples.

1. Treine o modelo.
2. Salve o arquivo.
3. Feche o notebook.
4. Abra novamente.
5. Carregue o modelo.
6. Faça uma previsão para um novo aluno.

Explique por que não foi necessário realizar um novo treinamento.

---

# 🏁 Encerramento da Aula 03

Parabéns!

Nesta aula você:

✅ Preparou os dados para o treinamento.

✅ Separou Features e Target.

✅ Criou uma Árvore de Decisão.

✅ Treinou um modelo.

✅ Fez previsões.

✅ Avaliou a acurácia.

✅ Visualizou a árvore.

✅ Descobriu a importância das Features.

✅ Salvou e reutilizou um modelo.

Você já domina o fluxo básico de um projeto de Machine Learning utilizando o Scikit-Learn.

---

# 🚀 Preparando a Aula 04

Na próxima aula conheceremos um algoritmo completamente diferente da Árvore de Decisão.

Enquanto a árvore toma decisões por meio de perguntas, o **K-Nearest Neighbors (KNN)** toma decisões comparando um novo registro com os exemplos mais parecidos.

Você verá que dois algoritmos diferentes podem resolver o mesmo problema utilizando estratégias completamente distintas.

---

**© @karizeviecelli - 2026**


<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 01 – Por que limpar os dados?

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 01 – Por que limpar os dados?

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender por que limpar um dataset antes do treinamento.
- Reconhecer problemas comuns em bases de dados.
- Explicar o conceito de GIGO.
- Identificar situações que prejudicam um modelo de Machine Learning.

---

## 📍 Onde estamos?

Na aula anterior aprendemos a:

- importar um arquivo CSV;
- abrir um dataset com Pandas;
- explorar informações utilizando `head()`, `info()`, `shape()` e `describe()`.

Hoje vamos dar o próximo passo.

Antes de criar um modelo de Machine Learning, precisamos responder uma pergunta muito importante:

> **Os nossos dados estão prontos para serem utilizados?**

Na maioria das vezes, a resposta é **não**.

---

# Situação-Problema

Imagine que você trabalha em uma escola.

O diretor entrega o seguinte arquivo.

| Nome | Idade | Curso | Nota |
|------|------:|--------|------:|
| Ana | 18 | Python | 8.5 |
| João | | Java | 7.2 |
| Maria | -5 | Python | 9.1 |
| Pedro | 20 | PYTHON | 8.0 |
| Pedro | 20 | PYTHON | 8.0 |

O diretor pergunta:

> **"Podemos utilizar esse arquivo para treinar uma Inteligência Artificial?"**

Observe atentamente.

O que chamou sua atenção?

Pense por alguns segundos antes de continuar.

---

# O que você encontrou?

Provavelmente você percebeu alguns problemas.

✔ Existe uma idade negativa.

✔ Existe uma idade em branco.

✔ O curso aparece escrito de formas diferentes.

✔ Existe um registro duplicado.

Esses problemas são muito comuns em empresas.

---

# O que acontece se ignorarmos esses erros?

Imagine que queremos ensinar uma criança a reconhecer frutas.

Mostramos várias imagens.

Mas algumas delas estão erradas.

```
🍎  → Maçã

🍌  → Banana

🚗  → Banana

🍎  → Carro
```

A criança ficará confusa.

Com o computador acontece exatamente a mesma coisa.

Se ensinarmos utilizando dados incorretos, ele aprenderá padrões incorretos.

---

# GIGO

Existe uma regra muito conhecida na Ciência de Dados.

```text
Garbage In

↓

Garbage Out
```

Em português.

> **Entrou lixo, saiu lixo.**

Ou seja.

```
Dados ruins

↓

Modelo ruim

↓

Resultados ruins
```

---

# Analogia

Imagine que você deseja preparar um bolo.

Você utiliza:

- farinha vencida;
- leite estragado;
- ovos quebrados.

O bolo ficará bom?

Claro que não.

Mesmo utilizando a melhor receita.

No Machine Learning acontece exatamente a mesma coisa.

Não importa o algoritmo.

Se os dados forem ruins.

O resultado também será.

---

# A limpeza de dados faz parte do trabalho

Muitas pessoas acreditam que um Cientista de Dados passa o dia criando Inteligência Artificial.

Na prática, grande parte do tempo é utilizada preparando os dados.

Uma estimativa bastante citada na área mostra que profissionais de dados podem gastar entre **60% e 80% do tempo** coletando, entendendo e preparando dados antes da etapa de modelagem.

Isso mostra que a qualidade dos dados é tão importante quanto a escolha do algoritmo.

---

# Quais problemas encontramos em um dataset?

Durante este curso aprenderemos a identificar os principais problemas.

| Problema | Exemplo |
|-----------|----------|
| Valores nulos | Nota em branco |
| Dados duplicados | Mesmo aluno cadastrado duas vezes |
| Texto despadronizado | Python, python, PYTHON |
| Tipos incorretos | "18" como texto em vez de número |
| Valores inválidos | Idade = -5 |
| Datas incorretas | 31/02/2026 |

Hoje conheceremos todos eles.

Nas próximas aulas aprenderemos como corrigi-los.

---

# Fluxo de preparação dos dados

Sempre siga esta sequência.

```text
Receber o dataset

↓

Explorar os dados

↓

Encontrar problemas

↓

Corrigir problemas

↓

Validar os dados

↓

Treinar o modelo
```

Esse será o fluxo utilizado durante todo o curso.

---

# 💻 Vamos para o laboratório!

Nesta aula utilizaremos o arquivo:

```text
alunos_sujo.csv
```

Esse arquivo foi criado especialmente para esta aula.

Ele possui diversos problemas que você aprenderá a identificar e corrigir.

---

# 🎯 Desafio Inicial

Antes de escrever qualquer código.

Abra o arquivo `alunos_sujo.csv` utilizando um editor de planilhas ou o próprio Google Colab.

Sem utilizar Python, anote todos os problemas que conseguir encontrar.

Depois compare sua lista com a dos colegas.

Quantos problemas diferentes vocês encontraram?

---

# 💼 No mercado

Imagine que uma empresa deseja criar um sistema para prever quais clientes irão cancelar um serviço.

Se o cadastro possui:

- nomes duplicados;
- datas erradas;
- idades inválidas;
- campos vazios;

o modelo fará previsões com baixa qualidade.

Por isso, empresas investem muito tempo na preparação dos dados antes de treinar qualquer algoritmo.

---

# 📌 Resumo

Neste módulo você aprendeu que:

✅ Dados de baixa qualidade geram resultados de baixa qualidade.

✅ GIGO significa **Garbage In, Garbage Out**.

✅ Antes de treinar um modelo precisamos verificar se os dados fazem sentido.

✅ Os principais problemas encontrados em datasets são:

- valores nulos;
- registros duplicados;
- textos despadronizados;
- tipos incorretos;
- valores inválidos.

---

# 🚀 Preparando o próximo módulo

Agora que entendemos por que limpar os dados é importante.

Vamos abrir o arquivo **alunos_sujo.csv** utilizando Pandas e responder à primeira pergunta de um Cientista de Dados:

> **"Quais problemas realmente existem neste dataset?"**

No próximo módulo aprenderemos a investigar um dataset utilizando:

```python
df.info()

df.shape

df.isnull()

df.describe()
```

Esses comandos serão nossas primeiras ferramentas para encontrar problemas automaticamente.

---

**© @karizeviecelli - 2026**


<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 02 – Investigando um Dataset

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 02 – Investigando um Dataset

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Abrir um dataset no Google Colab.
- Identificar possíveis problemas utilizando Pandas.
- Localizar valores nulos.
- Conhecer a estrutura de um DataFrame.
- Descobrir informações importantes antes da limpeza dos dados.

---

# Situação-Problema

Imagine que você recebeu o arquivo:

```text
alunos_sujo.csv
```

Você sabe que ele possui alguns problemas.

Mas...

**Quais problemas são esses?**

Antes de corrigir qualquer coisa, precisamos investigar.

É exatamente isso que faremos agora.

---

# Etapa 1 — Importando o Pandas

Como já aprendemos na Aula 01, o primeiro passo é importar a biblioteca Pandas.

```python
import pandas as pd
```

### O que este comando faz?

Importa a biblioteca Pandas para que possamos trabalhar com tabelas.

---

# Etapa 2 — Abrindo o arquivo

Agora vamos abrir o dataset.

```python
df = pd.read_csv("alunos_sujo.csv")
```

### Entendendo o código

```python
pd.read_csv()
```

Lê um arquivo CSV.

---

```python
"alunos_sujo.csv"
```

É o nome do arquivo.

---

```python
df
```

Receberá o DataFrame criado.

---

# Etapa 3 — Visualizando a tabela

Agora basta escrever:

```python
df
```

Ou

```python
df.head()
```

Resultado esperado:

| Nome | Idade | Curso | Nota | Frequencia |
|------|------:|--------|------:|-----------:|
| Ana | 18 | Python | 8.5 | 95 |
| João | NaN | Java | 7.2 | 88 |
| Maria | -5 | Python | NaN | 96 |
| Pedro | 20 | PYTHON | 8.0 | 91 |
| Pedro | 20 | PYTHON | 8.0 | 91 |

Observe.

Você já consegue perceber alguns problemas.

---

# Primeira pergunta

## Quantas linhas e colunas existem?

Execute.

```python
df.shape
```

Resultado.

```text
(25,6)
```

O primeiro número representa:

```
Linhas
```

O segundo número representa:

```
Colunas
```

Sempre leia assim.

```
(Linhas, Colunas)
```

---

# Segunda pergunta

## Quais colunas existem?

```python
df.columns
```

Resultado.

```text
Index([
'Nome',
'Idade',
'Curso',
'Nota',
'Frequencia',
'Aprovado'
])
```

Agora sabemos exatamente quais informações estão armazenadas.

---

# Terceira pergunta

## Que tipo de informação existe em cada coluna?

Execute.

```python
df.dtypes
```

Resultado.

```text
Nome            object
Idade            int64
Curso           object
Nota           float64
Frequencia       int64
Aprovado        object
```

---

## Interpretando

| Tipo | Significado |
|-------|-------------|
| object | Texto |
| int64 | Número inteiro |
| float64 | Número decimal |

Saber o tipo de cada coluna será muito importante nas próximas aulas.

---

# Quarta pergunta

## O DataFrame possui informações ausentes?

Agora vamos utilizar um comando muito importante.

```python
df.info()
```

Exemplo de saída:

```text
<class 'pandas.core.frame.DataFrame'>

RangeIndex: 25 entries

Data columns (total 6 columns)

Nome          25 non-null

Idade         24 non-null

Curso         25 non-null

Nota          23 non-null

Frequencia    25 non-null

Aprovado      25 non-null
```

---

## O que significa "non-null"?

**Non-null** significa:

```
Não vazio
```

Observe.

Se existem 25 registros.

Mas a coluna **Nota** possui apenas 23 valores.

Isso significa que:

```
Existem duas células vazias.
```

---

# Quinta pergunta

## Onde estão os valores nulos?

Agora vamos localizar exatamente onde estão.

Execute.

```python
df.isnull()
```

Resultado.

```text
Nome    Idade   Curso   Nota

False   False   False   False

False   True    False   False

False   False   False   True
```

---

## Como interpretar?

```
False

↓

Existe valor
```

```
True

↓

Valor ausente
```

---

# Sexta pergunta

## Quantos valores nulos existem?

O comando anterior mostra linha por linha.

Agora queremos um resumo.

```python
df.isnull().sum()
```

Resultado.

```text
Nome           0

Idade          1

Curso          0

Nota           2

Frequencia     0

Aprovado       0
```

Muito mais fácil de interpretar.

---

# Sétima pergunta

## Como estão distribuídos os números?

Execute.

```python
df.describe()
```

Resultado.

| Estatística | Idade | Nota |
|-------------|------:|-----:|
| Média | 19.4 | 8.2 |
| Mínimo | -5 | 6.8 |
| Máximo | 23 | 9.9 |

Observe.

Existe uma idade igual a:

```
-5
```

Faz sentido?

Claro que não.

Acabamos de encontrar outro problema.

---

# O que descobrimos até agora?

Sem corrigir absolutamente nada.

Já encontramos.

✅ Valores nulos.

✅ Idade inválida.

✅ Possível duplicidade.

✅ Textos diferentes.

Tudo isso apenas utilizando comandos simples do Pandas.

---

# 💻 Laboratório 01

Abra o arquivo.

```
alunos_sujo.csv
```

Execute os seguintes comandos.

```python
df.head()
```

```python
df.shape
```

```python
df.columns
```

```python
df.dtypes
```

```python
df.info()
```

```python
df.isnull()
```

```python
df.isnull().sum()
```

```python
df.describe()
```

Complete a tabela.

| Pergunta | Resposta |
|-----------|----------|
| Quantas linhas existem? | |
| Quantas colunas? | |
| Existe valor nulo? | |
| Em quais colunas? | |
| Existe idade inválida? | |
| Existem textos diferentes? | |

---

# ⚠️ Erro comum

Muitos iniciantes executam:

```python
df.info()
```

e ignoram o resultado.

Não faça isso.

Esse comando fornece um resumo completo do DataFrame.

Sempre analise cuidadosamente:

- quantidade de linhas;
- quantidade de colunas;
- tipos de dados;
- quantidade de valores preenchidos.

---

# 💼 No mercado

Receber um dataset e executar:

```python
df.head()

df.info()

df.describe()

df.isnull().sum()
```

é uma prática comum em praticamente todos os projetos de Ciência de Dados.

Esses comandos ajudam o profissional a entender rapidamente a qualidade dos dados antes de qualquer tratamento.

---

# 📌 Resumo

Neste módulo você aprendeu a investigar um dataset utilizando:

| Comando | Finalidade |
|----------|------------|
| `head()` | Visualizar as primeiras linhas |
| `shape` | Descobrir linhas e colunas |
| `columns` | Listar colunas |
| `dtypes` | Verificar tipos de dados |
| `info()` | Obter um resumo do DataFrame |
| `isnull()` | Localizar valores nulos |
| `isnull().sum()` | Contar valores nulos |
| `describe()` | Exibir estatísticas numéricas |

---

# 🚀 Preparando o próximo módulo

Agora já sabemos quais problemas existem.

No próximo módulo aprenderemos a resolver o primeiro deles.

> **Valores nulos (Missing Values)**

Você aprenderá quando remover registros e quando preencher informações ausentes utilizando o Pandas.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 03 – Tratando Valores Nulos

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 03 – Tratando Valores Nulos (Missing Values)

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender o que são valores nulos.
- Identificar campos vazios em um dataset.
- Utilizar `fillna()` para preencher informações.
- Utilizar `dropna()` para remover registros incompletos.
- Escolher a melhor estratégia para cada situação.

---

# Situação-Problema

Você recebeu o arquivo:

```text
alunos_sujo.csv
```

Durante a investigação descobriu que algumas informações estão faltando.

Veja um exemplo.

| Nome | Idade | Curso | Nota |
|------|------:|--------|------:|
| Ana | 18 | Python | 8.5 |
| João | 20 | Java | |
| Maria | 19 | Python | 9.2 |

Observe.

João não possui nota.

O computador não sabe se:

- a nota foi esquecida;
- o aluno não fez a prova;
- ocorreu erro na importação.

Esses valores recebem o nome de **Valores Nulos**.

---

# O que significa NaN?

Quando um valor está ausente, o Pandas normalmente exibe:

```text
NaN
```

NaN significa:

```
Not a Number
```

Ou seja.

"O valor não está disponível."

Não significa necessariamente que seja um número errado.

Significa apenas que aquela informação não existe.

---

# Por que aparecem valores nulos?

Em empresas isso acontece diariamente.

Alguns exemplos.

✔ Cliente esqueceu de informar um telefone.

✔ Sistema antigo perdeu informações.

✔ Funcionário digitou apenas parte do cadastro.

✔ Arquivo foi importado de maneira incorreta.

✔ Sensor deixou de enviar dados.

Portanto.

Encontrar valores nulos é completamente normal.

---

# Como descobrir quantos valores nulos existem?

Já vimos este comando.

```python
df.isnull().sum()
```

Resultado.

```text
Nome            0

Idade           1

Curso           0

Nota            2

Frequencia      0

Aprovado        0
```

Observe.

A coluna **Nota** possui dois valores ausentes.

---

# Primeira estratégia

## Preencher os valores

Em algumas situações podemos substituir os valores vazios.

Para isso utilizamos:

```python
df.fillna(0)
```

Resultado.

| Nome | Nota |
|------|------:|
| Ana | 8.5 |
| João | 0.0 |
| Maria | 9.2 |

Todos os campos vazios passam a valer zero.

---

# Atenção!

Nem sempre preencher com zero é a melhor escolha.

Imagine um cadastro de pessoas.

| Nome | Idade |
|------|------:|
| João | |

Faz sentido substituir por:

```
0 anos?
```

Provavelmente não.

Precisamos analisar o contexto.

---

# Preenchendo com um texto

Quando a coluna é textual.

Podemos fazer.

```python
df["Curso"] = df["Curso"].fillna("Não informado")
```

Resultado.

| Nome | Curso |
|------|--------|
| Ana | Python |
| João | Não informado |

---

# Preenchendo com a média

Essa é uma estratégia muito utilizada.

Imagine.

| Nota |
|------:|
| 8.5 |
| 9.0 |
| |
| 7.5 |

Podemos substituir o valor vazio pela média.

Código.

```python
media = df["Nota"].mean()

df["Nota"] = df["Nota"].fillna(media)
```

---

## O que significa `mean()`?

Em inglês.

```
Mean

↓

Média
```

O Pandas calcula automaticamente a média da coluna.

Depois.

`fillna()` utiliza esse valor para preencher os espaços vazios.

---

# Visualizando o resultado

Antes.

| Nota |
|------:|
| 8.5 |
| |
| 9.0 |

Depois.

| Nota |
|------:|
| 8.5 |
| 8.75 |
| 9.0 |

---

# Segunda estratégia

## Remover registros

Em alguns casos.

Pode ser melhor remover as linhas incompletas.

Para isso utilizamos.

```python
df.dropna()
```

Resultado.

Todas as linhas contendo valores nulos são removidas.

---

# Quando utilizar?

Imagine um dataset com:

```
500.000 registros
```

Apenas cinco possuem problemas.

Nesse caso.

Remover essas cinco linhas dificilmente afetará o resultado.

---

Agora imagine outro cenário.

```
200 registros
```

E.

```
120 registros possuem valores nulos.
```

Remover tudo seria um grande prejuízo.

Nesse caso.

É melhor preencher os valores.

---

# Comparando

| Estratégia | Quando utilizar |
|------------|----------------|
| `fillna()` | Quando é possível substituir a informação |
| `dropna()` | Quando poucos registros possuem problemas |

---

# Laboratório 01

Abra o arquivo.

```text
alunos_sujo.csv
```

Execute.

```python
df.isnull().sum()
```

Anote.

Quais colunas possuem valores nulos?

---

Agora.

Crie uma nova variável.

```python
novo_df = df.fillna(0)
```

Exiba.

```python
novo_df
```

Observe.

O DataFrame original continua igual.

---

Agora.

Preencha a coluna Nota utilizando a média.

```python
media = df["Nota"].mean()

df["Nota"] = df["Nota"].fillna(media)
```

Verifique novamente.

```python
df.isnull().sum()
```

---

# Laboratório 02

Abra o dataset.

```text
clientes_sujo.csv
```

Descubra.

- Existem valores nulos?

Depois.

Preencha-os utilizando a estratégia que você considera mais adequada.

Explique sua escolha.

---

# ⚠️ Erro comum

Muitos iniciantes fazem:

```python
df.fillna(0)
```

E acreditam que o DataFrame foi alterado.

Na verdade.

Esse comando retorna um novo DataFrame.

Se quiser salvar a alteração.

Faça.

```python
df = df.fillna(0)
```

Ou.

```python
df.fillna(0, inplace=True)
```

---

# 💼 No mercado

Não existe uma única maneira correta de tratar valores nulos.

A decisão depende do problema.

Um Cientista de Dados analisa:

- quantidade de registros;
- importância da coluna;
- impacto da informação ausente.

Só então escolhe a melhor estratégia.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ O que são valores nulos.

✅ O significado de NaN.

✅ Como localizar valores ausentes.

✅ Como preencher utilizando:

- zero;
- texto;
- média.

✅ Como remover registros incompletos.

---

# 🚀 Preparando o próximo módulo

Agora nosso dataset possui menos problemas.

O próximo passo será localizar registros repetidos.

Você aprenderá a identificar e remover duplicidades utilizando:

```python
df.duplicated()

df.drop_duplicates()
```

Esses comandos ajudam a evitar que um mesmo registro influencie o treinamento mais de uma vez.

---

**© @karizeviecelli - 2026**


<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 04 – Registros Duplicados

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 04 – Registros Duplicados

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Identificar registros duplicados.
- Entender por que eles prejudicam um modelo de Machine Learning.
- Utilizar `duplicated()`.
- Utilizar `drop_duplicates()`.
- Escolher quando remover registros repetidos.

---

# Situação-Problema

Imagine que uma escola possui o seguinte cadastro.

| Nome | Curso | Nota |
|------|--------|------:|
| Ana | Python | 8.5 |
| João | Java | 7.2 |
| Pedro | Python | 8.0 |
| Pedro | Python | 8.0 |
| Maria | Front-end | 9.1 |

Observe atentamente.

Existe algum problema?

Sim.

O aluno **Pedro** aparece duas vezes.

---

# O que é um registro duplicado?

Um registro duplicado ocorre quando uma mesma informação aparece mais de uma vez no dataset.

Exemplo.

| Nome | Curso |
|------|--------|
| Ana | Python |
| Ana | Python |

Nem sempre isso significa erro.

Mas, na maioria das vezes, precisamos investigar.

---

# Por que isso acontece?

Em empresas é muito comum.

Algumas situações.

- O usuário clicou duas vezes no botão "Salvar".
- Dois funcionários cadastraram o mesmo cliente.
- O sistema importou o arquivo duas vezes.
- A integração entre sistemas repetiu informações.

---

# Qual o problema?

Imagine uma pesquisa com apenas quatro alunos.

| Nome | Nota |
|------|------:|
| Ana | 8 |
| João | 6 |
| Pedro | 10 |
| Maria | 7 |

Agora imagine que Pedro foi cadastrado novamente.

| Nome | Nota |
|------|------:|
| Ana | 8 |
| João | 6 |
| Pedro | 10 |
| Pedro | 10 |
| Maria | 7 |

Perceba.

Pedro passou a influenciar o conjunto de dados duas vezes.

Isso pode alterar estatísticas e também o treinamento do modelo.

---

# Como localizar registros duplicados?

O Pandas possui um comando específico.

```python
df.duplicated()
```

---

## O que esse comando faz?

Ele verifica linha por linha.

Quando encontra um registro repetido retorna:

```text
True
```

Caso contrário.

```text
False
```

---

## Exemplo

```python
df.duplicated()
```

Resultado.

```text
0    False

1    False

2    False

3     True

4    False
```

Observe.

A linha 3 é uma duplicata.

---

# Como visualizar apenas os registros repetidos?

Podemos utilizar.

```python
df[df.duplicated()]
```

Resultado.

| Nome | Curso | Nota |
|------|--------|------:|
| Pedro | Python | 8.0 |

Muito mais fácil de analisar.

---

# Quantos registros duplicados existem?

Execute.

```python
df.duplicated().sum()
```

Resultado.

```text
1
```

Esse comando conta quantas duplicidades existem.

---

# Removendo duplicados

Depois da análise.

Podemos remover.

```python
df = df.drop_duplicates()
```

Resultado.

| Nome | Curso | Nota |
|------|--------|------:|
| Ana | Python | 8.5 |
| João | Java | 7.2 |
| Pedro | Python | 8.0 |
| Maria | Front-end | 9.1 |

Agora o registro repetido foi eliminado.

---

# Atenção!

Nunca remova registros duplicados sem analisar.

Imagine um hospital.

| Paciente | Data |
|-----------|------|
| João | 10/05 |
| João | 18/05 |

São dois atendimentos diferentes.

Apesar do mesmo nome.

Os registros são válidos.

Sempre analise antes de excluir.

---

# Laboratório 01

Abra o arquivo.

```text
alunos_sujo.csv
```

Execute.

```python
df.duplicated()
```

Agora.

Conte.

```python
df.duplicated().sum()
```

Depois.

Visualize apenas as duplicatas.

```python
df[df.duplicated()]
```

---

# Laboratório 02

Remova as duplicatas.

```python
df = df.drop_duplicates()
```

Agora execute novamente.

```python
df.duplicated().sum()
```

Qual foi o resultado?

---

# Desafio

Abra o arquivo.

```text
clientes_sujo.csv
```

Descubra.

- Existem registros duplicados?
- Quantos?
- Quais são eles?

Depois.

Remova-os.

---

# ⚠️ Erro comum

Alguns alunos fazem.

```python
df.drop_duplicates()
```

E acreditam que o DataFrame foi alterado.

Na verdade.

O comando retorna um novo DataFrame.

Faça.

```python
df = df.drop_duplicates()
```

Ou.

```python
df.drop_duplicates(inplace=True)
```

---

# 💼 No mercado

Remover duplicidades é uma etapa comum em sistemas de:

- CRM
- ERP
- E-commerce
- Hospitais
- Bancos
- Indústrias

Duplicidades podem gerar relatórios incorretos e prejudicar modelos de Machine Learning.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ O que são registros duplicados.

✅ Como localizar duplicidades.

✅ Como contar registros repetidos.

✅ Como remover duplicatas.

✅ Quando NÃO remover duplicidades.

---

# 🚀 Preparando o próximo módulo

Agora nosso dataset já possui menos problemas.

O próximo desafio será padronizar os textos.

Você aprenderá a transformar situações como:

```text
Python

python

PYTHON

 Python
```

em um único padrão utilizando o Pandas.

Utilizaremos funções como:

```python
str.upper()

str.lower()

str.title()

str.strip()
```

---

**© @karizeviecelli - 2026**


<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 05 – Padronizando Dados de Texto

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 05 – Padronizando Dados de Texto

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender por que textos despadronizados prejudicam análises.
- Remover espaços em branco desnecessários.
- Converter textos para maiúsculas, minúsculas ou formato de título.
- Padronizar informações utilizando o Pandas.

---

# Situação-Problema

Imagine que uma escola cadastrou os cursos dos alunos.

Observe a coluna abaixo.

| Curso |
|--------|
| Python |
| python |
| PYTHON |
| Python |
|  Python |
| python  |

Você considera que existem seis cursos diferentes?

Claro que não.

Todos representam o mesmo curso.

Mas o computador enxerga cada um deles como um valor diferente.

---

# Por que isso acontece?

Para o computador:

```text
Python
```

é diferente de

```text
python
```

que também é diferente de

```text
PYTHON
```

e diferente de

```text
 Python
```

Mesmo que para nós todos representem a mesma informação.

---

# Problemas causados

Imagine que você deseja contar quantos alunos estudam Python.

Observe.

| Curso |
|--------|
| Python |
| python |
| PYTHON |

O resultado esperado seria:

```
Python

↓

3 alunos
```

Mas o computador pode interpretar como:

```
Python

↓

1
```

```
python

↓

1
```

```
PYTHON

↓

1
```

A análise ficará incorreta.

---

# Nosso objetivo

Transformar isto:

```text
Python

python

PYTHON

 Python

python
```

Em apenas um padrão.

Por exemplo.

```text
Python
```

---

# Primeiro problema

## Espaços em branco

Observe.

```text
Python
```

e

```text
 Python
```

São diferentes.

Também.

```text
Python
```

e

```text
Python
```

(com espaço no final)

São diferentes.

---

# Removendo espaços

Utilizamos.

```python
df["Curso"] = df["Curso"].str.strip()
```

---

## O que significa strip()?

A função:

```python
strip()
```

Remove espaços no início e no final do texto.

Exemplo.

Antes.

```text
" Python "
```

Depois.

```text
"Python"
```

---

# Vamos testar

Execute.

```python
print(df["Curso"])
```

Depois.

```python
df["Curso"] = df["Curso"].str.strip()
```

Exiba novamente.

```python
print(df["Curso"])
```

Observe a diferença.

---

# Segundo problema

## Letras maiúsculas

Observe.

```text
Python

PYTHON
```

Ainda são considerados valores diferentes.

---

# Transformando tudo em maiúsculas

```python
df["Curso"] = df["Curso"].str.upper()
```

Resultado.

```text
PYTHON

PYTHON

PYTHON
```

Agora todos seguem o mesmo padrão.

---

# O que faz upper()?

A função:

```python
upper()
```

Converte todas as letras para maiúsculas.

Exemplo.

Antes.

```text
Python
```

Depois.

```text
PYTHON
```

---

# Transformando tudo em minúsculas

Também podemos fazer.

```python
df["Curso"] = df["Curso"].str.lower()
```

Resultado.

```text
python

python

python
```

---

# O que faz lower()?

Converte todas as letras para minúsculas.

Antes.

```text
JAVA
```

Depois.

```text
java
```

---

# Formato de título

Outra opção muito utilizada.

```python
df["Curso"] = df["Curso"].str.title()
```

Resultado.

```text
Python

Banco De Dados

Front-End
```

A primeira letra de cada palavra fica maiúscula.

---

# Comparando

| Função | Resultado |
|---------|-----------|
| `upper()` | PYTHON |
| `lower()` | python |
| `title()` | Python |
| `strip()` | Remove espaços |

---

# Aplicando várias funções

Podemos combinar funções.

```python
df["Curso"] = df["Curso"].str.strip().str.title()
```

O que acontece?

Primeiro.

Remove os espaços.

Depois.

Padroniza o texto.

É uma prática muito comum em projetos reais.

---

# Exemplo completo

Antes.

| Curso |
|--------|
| python |
| PYTHON |
|  Python |
| python  |

Depois.

| Curso |
|--------|
| Python |
| Python |
| Python |
| Python |

Agora todos os registros seguem o mesmo padrão.

---

# Laboratório 01

Abra o arquivo.

```text
alunos_sujo.csv
```

Visualize apenas a coluna Curso.

```python
df["Curso"]
```

Observe os diferentes formatos.

Agora execute.

```python
df["Curso"] = df["Curso"].str.strip()
```

Depois.

```python
df["Curso"] = df["Curso"].str.title()
```

Visualize novamente.

```python
df["Curso"]
```

O que mudou?

---

# Laboratório 02

Abra.

```text
clientes_sujo.csv
```

Padronize a coluna Cidade.

Depois.

Padronize qualquer outra coluna de texto.

---

# Desafio

Utilizando apenas os comandos aprendidos.

Padronize todas as colunas de texto do dataset.

Dica.

Você pode utilizar.

```python
df.columns
```

para descobrir quais colunas existem.

---

# ⚠️ Erro comum

Alguns alunos fazem.

```python
df.upper()
```

Isso gera erro.

Por quê?

Porque:

```python
upper()
```

é uma função para textos.

Quando trabalhamos com uma coluna textual no Pandas devemos utilizar:

```python
df["Curso"].str.upper()
```

Observe o uso de:

```python
.str
```

Ela informa ao Pandas que queremos utilizar funções específicas para texto.

---

# 💼 No mercado

Padronização de textos é uma das primeiras etapas em projetos de:

- CRM
- E-commerce
- Sistemas hospitalares
- Recursos Humanos
- Marketing
- Bancos

Sem padronização, relatórios podem apresentar informações duplicadas ou incorretas.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ Como remover espaços desnecessários.

✅ Como converter textos para maiúsculas.

✅ Como converter textos para minúsculas.

✅ Como utilizar o formato de título.

✅ Como combinar funções para padronizar dados.

---

# 🧪 Mini Projeto

O arquivo abaixo possui diversos problemas de padronização.

```text
Curso

Python

 python

PYTHON

Python

JAVA

java

 Java
```

Escreva um código que transforme todos os registros em:

```text
Python

Python

Python

Python

Java

Java

Java
```

---

# 🚀 Preparando o próximo módulo

Agora nosso dataset já possui:

✅ Valores nulos tratados.

✅ Registros duplicados removidos.

✅ Textos padronizados.

O próximo passo será verificar se cada coluna possui o tipo de dado correto.

Aprenderemos funções como:

```python
astype()

to_numeric()

to_datetime()
```

Essas funções serão fundamentais para preparar nossos dados para o primeiro modelo de Machine Learning.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 06 – Corrigindo Tipos de Dados

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 06 – Corrigindo Tipos de Dados

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Entender o que são tipos de dados.
- Identificar quando uma coluna possui um tipo incorreto.
- Converter textos em números.
- Converter textos em datas.
- Alterar o tipo de uma coluna utilizando Pandas.

---

# Situação-Problema

Imagine que você recebeu o seguinte arquivo.

| Nome | Idade |
|------|-------|
| Ana | 18 |
| João | 20 |
| Maria | 25 |

Visualmente parece correto.

Mas existe um detalhe importante.

Observe o resultado do comando:

```python
df.dtypes
```

Resultado.

```text
Nome      object

Idade     object
```

Espere...

A idade não deveria ser um número?

Sim.

Mas ela foi importada como texto.

---

# Por que isso acontece?

Isso pode acontecer quando:

- o arquivo foi exportado incorretamente;
- existem letras misturadas com números;
- existem espaços em branco;
- existe algum valor inválido.

Veja um exemplo.

| Idade |
|-------|
| 18 |
| 20 |
| vinte |
| 25 |

Como existe um texto ("vinte"), o Pandas entende que toda a coluna deve ser tratada como texto.

---

# Por que isso é um problema?

Imagine calcular a média da idade.

```python
df["Idade"].mean()
```

Se a coluna for texto.

O Pandas exibirá erro.

Algumas operações matemáticas só funcionam com números.

---

# Como descobrir o tipo de cada coluna?

Utilize:

```python
df.dtypes
```

Exemplo.

```text
Nome            object

Idade            int64

Nota           float64

Data            object
```

---

## Relembrando

| Tipo | Significado |
|-------|-------------|
| object | Texto |
| int64 | Número inteiro |
| float64 | Número decimal |
| bool | Verdadeiro ou falso |

---

# Convertendo para número inteiro

Utilizamos.

```python
df["Idade"] = df["Idade"].astype(int)
```

Resultado.

A coluna passa a ser do tipo:

```text
int64
```

---

# Convertendo para número decimal

```python
df["Nota"] = df["Nota"].astype(float)
```

Resultado.

```text
float64
```

Agora será possível calcular médias.

---

# Quando usar astype()?

Sempre que você tiver certeza de que todos os valores podem ser convertidos.

Por exemplo.

```
18

20

25
```

Não haverá problemas.

---

# E se existir um valor inválido?

Observe.

| Idade |
|-------|
| 18 |
| 20 |
| vinte |
| 25 |

Agora execute.

```python
df["Idade"].astype(int)
```

Resultado.

```text
ValueError
```

O Pandas não consegue transformar a palavra "vinte" em número.

---

# A solução

Utilizamos.

```python
pd.to_numeric()
```

Exemplo.

```python
df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")
```

---

## O que significa?

### pd.to_numeric()

Converte textos em números.

---

### errors="coerce"

Quando encontrar um valor inválido.

Transforma em:

```text
NaN
```

Assim.

Não ocorre erro.

Depois podemos tratar esses valores utilizando os comandos aprendidos no módulo anterior.

---

# Exemplo

Antes.

| Idade |
|-------|
| 18 |
| 20 |
| vinte |
| 25 |

Depois.

| Idade |
|-------:|
| 18 |
| 20 |
| NaN |
| 25 |

Muito melhor.

Agora conseguimos localizar o problema.

---

# Trabalhando com datas

Imagine esta coluna.

| Data |
|-----------|
| 15/03/2026 |
| 18/03/2026 |
| 25/03/2026 |

Visualmente parece uma data.

Mas para o Pandas.

Ela pode ser apenas um texto.

---

# Convertendo para data

Utilizamos.

```python
df["Data"] = pd.to_datetime(df["Data"])
```

Agora o Pandas entende que aquela coluna representa datas.

---

# Informando o formato brasileiro

Quando a data estiver no formato:

```text
15/03/2026
```

É recomendado utilizar.

```python
df["Data"] = pd.to_datetime(
    df["Data"],
    format="%d/%m/%Y"
)
```

---

# Principais códigos

| Código | Significado |
|---------|-------------|
| %d | Dia |
| %m | Mês |
| %Y | Ano com quatro dígitos |

---

# Comparando

| Função | Utilização |
|---------|------------|
| `astype(int)` | Número inteiro |
| `astype(float)` | Número decimal |
| `pd.to_numeric()` | Texto para número |
| `pd.to_datetime()` | Texto para data |

---

# Laboratório 01

Abra.

```text
clientes_sujo.csv
```

Execute.

```python
df.dtypes
```

Responda.

- Quais colunas são texto?
- Quais são números?

---

Agora tente converter uma coluna numérica.

```python
df["Renda"] = df["Renda"].astype(float)
```

Verifique novamente.

```python
df.dtypes
```

---

# Laboratório 02

Abra.

```text
vendas.csv
```

Observe a coluna:

```text
Data
```

Converta-a para data.

```python
df["Data"] = pd.to_datetime(
    df["Data"],
    format="%d/%m/%Y"
)
```

Depois execute.

```python
df.dtypes
```

O tipo mudou?

---

# Desafio

Imagine que um arquivo possui a coluna.

| Idade |
|-------|
| 18 |
| 22 |
| vinte |
| 30 |
| 25 |

Utilize.

```python
pd.to_numeric()
```

Para converter a coluna.

Depois descubra.

```python
df.isnull().sum()
```

Quantos valores inválidos existiam?

---

# ⚠️ Erro comum

Muitos iniciantes fazem.

```python
df["Idade"] = df["Idade"].astype(int)
```

Mesmo existindo textos.

O resultado será um erro.

Sempre que existir dúvida.

Utilize.

```python
pd.to_numeric(
    errors="coerce"
)
```

É muito mais seguro.

---

# 💼 No mercado

Grande parte dos problemas encontrados em datasets está relacionada a tipos incorretos.

É comum receber:

- números armazenados como texto;
- datas armazenadas como texto;
- valores monetários contendo símbolos.

Corrigir esses tipos é uma das primeiras tarefas de um Analista de Dados.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ Como verificar tipos de dados.

✅ Como converter textos em números.

✅ Como converter textos em datas.

✅ Como utilizar `astype()`.

✅ Como utilizar `pd.to_numeric()`.

✅ Como utilizar `pd.to_datetime()`.

---

# 🧪 Mini Projeto

O arquivo abaixo possui vários problemas.

| Produto | Valor | Data |
|----------|-------|------------|
| Mouse | 120.50 | 15/03/2026 |
| Teclado | 89.90 | 20/03/2026 |
| Monitor | cento e vinte | 25/03/2026 |

Sua missão é:

- Converter a coluna **Valor** para número.
- Descobrir quais valores são inválidos.
- Converter a coluna **Data** para o tipo data.
- Verificar os tipos utilizando `df.dtypes`.

---

# 🚀 Preparando o próximo módulo

Nosso dataset já está muito melhor.

Agora vamos aprender a identificar outro problema bastante comum:

- idades negativas;
- notas acima de 10;
- frequências maiores que 100%;
- valores extremamente altos ou baixos.

Esses são chamados de **valores inválidos** e **outliers simples**, e aprenderemos como identificá-los utilizando filtros no Pandas.

---

**© @karizeviecelli - 2026**

<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 07 – Identificando Valores Inválidos

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 07 – Identificando Valores Inválidos

---

## 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Identificar valores que não fazem sentido em um dataset.
- Utilizar filtros para localizar registros incorretos.
- Corrigir ou remover valores inválidos.
- Aplicar boas práticas antes de treinar um modelo de Machine Learning.

---

# 📍 Onde estamos?

Até agora aprendemos a:

✔ Corrigir valores nulos.

✔ Remover registros duplicados.

✔ Padronizar textos.

✔ Corrigir tipos de dados.

Agora falta verificar se os valores armazenados realmente fazem sentido.

---

# Situação-Problema

Observe a tabela abaixo.

| Nome | Idade | Nota | Frequência |
|------|------:|------:|-----------:|
| Ana | 18 | 8.5 | 95 |
| João | -5 | 7.2 | 82 |
| Maria | 22 | 15.0 | 88 |
| Pedro | 20 | 8.0 | 120 |

Visualmente a tabela parece correta.

Mas será que realmente está?

---

# O que há de errado?

Observe cada coluna.

## Idade

```
-5
```

Pode existir uma pessoa com idade negativa?

❌ Não.

---

## Nota

```
15
```

Nossa escola utiliza notas de:

```
0 até 10
```

Logo.

```
15
```

É um valor inválido.

---

## Frequência

```
120%
```

Um aluno pode ter frequência maior que:

```
100%
```

❌ Não.

---

# O que é um valor inválido?

Um valor inválido é uma informação que existe no dataset, mas não representa a realidade.

Exemplos.

| Coluna | Valor | Problema |
|---------|------:|----------|
| Idade | -3 | Não existe idade negativa |
| Nota | 15 | Nota máxima é 10 |
| Frequência | 120 | Máximo é 100% |
| Salário | -2500 | Salário negativo normalmente é inválido |

---

# Como localizar valores inválidos?

No Pandas podemos utilizar filtros.

---

# Exemplo 1

## Idades negativas

```python
df[df["Idade"] < 0]
```

### Como funciona?

Observe.

```python
df["Idade"] < 0
```

O Pandas verifica cada linha.

Se a idade for menor que zero.

Retorna:

```
True
```

Depois.

```python
df[ ... ]
```

Mostra apenas essas linhas.

---

# Resultado

| Nome | Idade |
|------|------:|
| João | -5 |

Encontramos o problema.

---

# Exemplo 2

## Notas maiores que 10

```python
df[df["Nota"] > 10]
```

Resultado.

| Nome | Nota |
|------|-----:|
| Maria | 15 |

---

# Exemplo 3

## Frequência maior que 100

```python
df[df["Frequencia"] > 100]
```

Resultado.

| Nome | Frequência |
|------|-----------:|
| Pedro | 120 |

---

# Combinando condições

Imagine que queremos localizar:

- idade negativa

OU

- nota maior que 10.

Podemos utilizar.

```python
df[
    (df["Idade"] < 0)
    |
    (df["Nota"] > 10)
]
```

---

## O que significa o símbolo |

No Pandas.

```text
|
```

Significa:

**OU**

---

# Utilizando E

Agora queremos.

Idade maior que 18

E

Nota maior que 8.

```python
df[
    (df["Idade"] > 18)
    &
    (df["Nota"] > 8)
]
```

---

## O que significa &

```text
&
```

Significa.

**E**

---

# Corrigindo um valor

Imagine que encontramos.

```
Idade = -5
```

Podemos corrigir manualmente.

```python
df.loc[1, "Idade"] = 18
```

---

## O que significa?

```python
loc
```

Permite alterar um valor específico.

No exemplo.

```
Linha 1

↓

Coluna Idade

↓

18
```

---

# Outra possibilidade

Em alguns casos.

É melhor remover o registro.

```python
df = df[df["Idade"] >= 0]
```

Assim permanecem apenas registros válidos.

---

# Laboratório 01

Abra.

```text
alunos_sujo.csv
```

Descubra.

- Existem idades negativas?
- Existem notas maiores que 10?
- Existe frequência maior que 100?

Utilize.

```python
df[df["Idade"] < 0]
```

```python
df[df["Nota"] > 10]
```

```python
df[df["Frequencia"] > 100]
```

---

# Laboratório 02

Abra.

```text
clientes_sujo.csv
```

Descubra.

- Existe renda negativa?
- Existe idade menor que zero?
- Existe algum valor muito estranho?

Depois.

Corrija ou remova esses registros.

---

# Desafio

O arquivo abaixo representa vendas.

| Produto | Quantidade | Valor |
|----------|-----------:|------:|
| Mouse | 2 | 120 |
| Teclado | -1 | 90 |
| Monitor | 3 | -1500 |
| SSD | 1 | 450 |

Responda.

1. Quais registros possuem problemas?

2. Como você corrigiria esses dados?

3. Em quais situações seria melhor remover o registro?

---

# ⚠️ Erro comum

Alguns alunos acreditam que todo número é válido.

Mas um computador não sabe que:

```
Idade = -10
```

é impossível.

Cabe ao Analista de Dados identificar esses problemas antes do treinamento.

---

# 💼 No mercado

Empresas trabalham diariamente com milhões de registros.

É comum encontrar:

- salários negativos;
- datas impossíveis;
- idades incompatíveis;
- valores financeiros incorretos;
- sensores com leituras absurdas.

Antes de qualquer análise, esses dados precisam ser revisados.

---

# 📌 Resumo

Neste módulo você aprendeu:

✅ O que são valores inválidos.

✅ Como localizar registros utilizando filtros.

✅ Como combinar condições com:

- **E** (`&`)
- **OU** (`|`)

✅ Como corrigir valores específicos com `loc`.

✅ Como remover registros inválidos.

---

# 🧪 Mini Projeto

Utilizando o arquivo:

```text
alunos_sujo.csv
```

Realize as seguintes tarefas.

✔ Localize idades negativas.

✔ Localize notas maiores que 10.

✔ Localize frequências maiores que 100.

✔ Corrija os valores encontrados.

✔ Exiba novamente o DataFrame.

---

# ✅ Checklist Profissional

Antes de treinar um modelo de Machine Learning, pergunte sempre:

☐ Existem valores nulos?

☐ Existem registros duplicados?

☐ Existem textos despadronizados?

☐ Os tipos de dados estão corretos?

☐ Existem valores inválidos?

☐ O dataset representa a realidade?

Se todas as respostas forem **SIM**, seu conjunto de dados estará muito mais preparado para treinar um modelo.

---

# 🚀 Preparando o próximo módulo

Nos próximos módulos vamos reunir tudo o que aprendemos.

Você realizará uma limpeza completa do arquivo **alunos_sujo.csv**, aplicando todas as técnicas estudadas nesta aula.

Ao final, terá um dataset pronto para construir seu primeiro modelo de Machine Learning.

---

**© @karizeviecelli - 2026**


<!--
=========================================================
Curso: Machine Learning com Python
Aula 02 – Limpeza e Preparação de Dados
Módulo 08 – Laboratório Prático: Limpando um Dataset Completo

© @karizeviecelli - 2026
=========================================================
-->

# Aula 02
# Módulo 08 – Laboratório Prático

---

# 🎯 Objetivos

Ao final deste módulo você será capaz de:

- Aplicar todas as técnicas aprendidas na Aula 02.
- Investigar um dataset.
- Identificar problemas.
- Corrigir problemas.
- Gerar um dataset pronto para Machine Learning.

---

# Situação-Problema

Você acaba de ser contratado como Analista de Dados.

Seu gerente entrega o arquivo:

```text
alunos_sujo.csv
```

Ele informa:

> "Precisamos utilizar Machine Learning para prever quais alunos serão aprovados.

> Antes disso precisamos limpar completamente esta base."

A partir deste momento a responsabilidade é sua.

---

# Missão

Você deverá:

✅ Encontrar problemas

✅ Corrigir problemas

✅ Validar os dados

✅ Gerar um novo dataset

---

# Passo 1

## Importando a biblioteca

```python
import pandas as pd
```

---

# Passo 2

## Abrindo o arquivo

```python
df = pd.read_csv("alunos_sujo.csv")
```

---

# Passo 3

## Visualizando os dados

```python
df.head()
```

Observe.

Não tente corrigir nada ainda.

---

# Passo 4

## Investigando o DataFrame

Execute.

```python
df.info()
```

Depois.

```python
df.shape
```

Depois.

```python
df.describe()
```

Agora responda.

- Quantas linhas existem?

- Quantas colunas?

- Existem valores nulos?

- Existem números estranhos?

---

# Passo 5

## Procurando valores nulos

```python
df.isnull().sum()
```

Anote.

Em quais colunas existem problemas?

---

# Passo 6

## Corrigindo valores nulos

Para as colunas numéricas.

```python
df["Nota"] = df["Nota"].fillna(
    df["Nota"].mean()
)
```

Para colunas de texto.

```python
df["Curso"] = df["Curso"].fillna(
    "Não informado"
)
```

---

# Passo 7

## Procurando duplicidades

```python
df.duplicated().sum()
```

Agora visualize.

```python
df[df.duplicated()]
```

Remova.

```python
df = df.drop_duplicates()
```

---

# Passo 8

## Padronizando textos

Removendo espaços.

```python
df["Curso"] = df["Curso"].str.strip()
```

Padronizando.

```python
df["Curso"] = df["Curso"].str.title()
```

Visualize.

```python
df["Curso"]
```

---

# Passo 9

## Verificando tipos

```python
df.dtypes
```

Existe alguma coluna que deveria ser número?

Existe alguma coluna que deveria ser data?

Caso necessário.

```python
df["Idade"] = df["Idade"].astype(int)
```

---

# Passo 10

## Procurando valores inválidos

Idades negativas.

```python
df[df["Idade"] < 0]
```

Notas acima de 10.

```python
df[df["Nota"] > 10]
```

Frequências acima de 100.

```python
df[df["Frequencia"] > 100]
```

Corrija ou remova os registros.

---

# Passo 11

## Conferência Final

Execute novamente.

```python
df.info()
```

Depois.

```python
df.describe()
```

Depois.

```python
df.isnull().sum()
```

Compare com o início da aula.

O dataset melhorou?

---

# Passo 12

## Salvando o novo dataset

Agora vamos gerar uma nova versão.

```python
df.to_csv(
    "alunos_tratado.csv",
    index=False
)
```

Pronto.

Você criou um novo arquivo.

```
alunos_tratado.csv
```

Esse será utilizado nas próximas aulas.

---

# Fluxo completo

```text
Receber o dataset

↓

Abrir o arquivo

↓

Investigar

↓

Encontrar problemas

↓

Corrigir

↓

Validar

↓

Salvar novo dataset
```

Esse fluxo será repetido durante todo o curso.

---

# Desafio

Repita exatamente os mesmos passos utilizando:

```text
clientes_sujo.csv
```

Não consulte o material.

Utilize apenas seus conhecimentos.

---

# Desafio Extra

Faça a limpeza do arquivo:

```text
filmes_sujo.csv
```

Ao terminar responda.

- Existem valores nulos?

- Existem duplicidades?

- Existem textos despadronizados?

- Existem valores inválidos?

---

# 💼 No mercado

É muito comum que empresas mantenham duas versões do mesmo conjunto de dados.

```
clientes.csv
```

Dados originais.

↓

Após limpeza.

```
clientes_tratado.csv
```

Essa prática evita perder os dados originais e permite repetir o processo caso seja necessário.

---

# 📌 Resumo

Hoje você realizou todas as etapas de preparação dos dados.

✔ Investigou o dataset.

✔ Corrigiu valores nulos.

✔ Removeu duplicidades.

✔ Padronizou textos.

✔ Corrigiu tipos de dados.

✔ Localizou valores inválidos.

✔ Gerou um novo dataset.

Parabéns!

Seu conjunto de dados agora está muito mais preparado para ser utilizado em Machine Learning.

---

# 🏆 Missão Concluída

Você acaba de realizar exatamente uma das tarefas mais comuns de um Analista de Dados.

Antes de qualquer algoritmo ser treinado, alguém precisa preparar os dados.

Hoje esse profissional foi você.

---

# 🚀 Preparando a Aula 03

Na próxima aula utilizaremos o arquivo:

```text
alunos_tratado.csv
```

Pela primeira vez construiremos um modelo de Machine Learning utilizando a biblioteca **Scikit-Learn**.

Você aprenderá:

- O que são **Features** e **Target** na prática.
- Como separar os dados para treinamento.
- Como criar sua primeira **Árvore de Decisão (Decision Tree)**.
- Como fazer previsões com novos dados.

---

**© @karizeviecelli - 2026**

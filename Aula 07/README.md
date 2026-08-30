# AULA 07 --- Introdução ao NLP: Como a IA trabalha com textos?

**Trilha de IA Aplicada --- Base Comum dos Projetos**\
**Carga horária sugerida:** 4 horas\
**Pré-requisitos:** features, target, classificação, treino/teste e
métricas básicas.

**[Click aqui - Link Colab](https://colab.research.google.com/drive/1siIiWtC72rY-qq2hdU1G-Dl5oGhrwp8w?usp=sharing)**

------------------------------------------------------------------------

## 1. Situação-problema

Até agora nossas features eram principalmente números.

Mas os projetos também possuem textos.

Exemplos:

``` text
"Paciente recusou o almoço."

"Estamos trabalhando com equipe reduzida."

"Tem um sofá abandonado perto da escola."

"Tenho dentista quinta às duas."
```

Para uma pessoa, essas frases possuem significado.

Mas um algoritmo como os que estudamos trabalha com:

``` text
NÚMEROS
```

Então surge nossa pergunta:

> **Como transformar texto em dados que um modelo consiga utilizar?**

------------------------------------------------------------------------

# 2. O que é NLP?

NLP significa:

> **Natural Language Processing**

Em português:

> **Processamento de Linguagem Natural**

É uma área da Inteligência Artificial que trabalha com linguagem humana.

Exemplos de aplicações:

-   classificar mensagens;
-   identificar assuntos;
-   detectar spam;
-   analisar comentários;
-   extrair informações;
-   resumir textos;
-   tradução;
-   assistentes conversacionais.

Nesta aula vamos trabalhar com uma tarefa básica:

> **classificação de textos.**

------------------------------------------------------------------------

# 3. NLP não significa necessariamente chatbot

Quando ouvimos "linguagem natural", é comum pensar imediatamente em
chatbot.

Mas NLP também pode ser:

``` text
Comentário
    ↓
SOBRECARGA
```

ou:

``` text
Mensagem
    ↓
COMPROMISSO
```

ou:

``` text
Descrição
    ↓
MÓVEIS
```

Portanto:

> Um sistema pode utilizar NLP sem possuir um chatbot.

------------------------------------------------------------------------

# 4. Aplicação nos projetos

## Cuidadores

``` text
"Paciente recusou o almoço."
            ↓
      ALIMENTAÇÃO
```

## NR-1

``` text
"Estamos trabalhando com equipe reduzida."
                 ↓
            SOBRECARGA
```

Os comentários devem ser tratados de forma compatível com anonimato e
análise agregada.

## Fiscalização Colaborativa

``` text
"Tem um sofá abandonado na calçada."
                 ↓
              MÓVEIS
```

## Agenda Inteligente

``` text
"Tenho dentista quinta às duas."
                 ↓
           COMPROMISSO
```

O mesmo conceito pode atender problemas diferentes.

------------------------------------------------------------------------

# 5. Primeiro: regras simples

Antes de Machine Learning, podemos escrever:

``` python
texto = "Paciente recusou o almoço".lower()

if "almoço" in texto:
    print("Alimentação")
```

Funciona?

Sim.

É Machine Learning?

``` text
NÃO.
```

Nós programamos diretamente a regra.

------------------------------------------------------------------------

# 6. Regra × Machine Learning

## Regra programada

``` text
SE encontrar "almoço"
ENTÃO alimentação
```

O programador definiu a decisão.

## Machine Learning

Fornecemos exemplos:

``` text
"recusou o almoço" → alimentação
"comeu pouco" → alimentação
"dormiu mal" → sono
"acordou várias vezes" → sono
```

O algoritmo utiliza esses exemplos para construir um modelo.

------------------------------------------------------------------------

# 7. O grande problema

Algoritmos não recebem naturalmente:

``` text
"Paciente dormiu pouco"
```

como uma feature numérica.

Precisamos transformar o texto.

Fluxo:

``` text
TEXTO
  ↓
PREPARAÇÃO
  ↓
VETORIZAÇÃO
  ↓
NÚMEROS
  ↓
MODELO
  ↓
CATEGORIA
```

------------------------------------------------------------------------

# 8. Preparação simples do texto

Podemos começar transformando tudo em letras minúsculas:

``` python
texto = "Paciente DORMIU Pouco"

texto = texto.lower()

print(texto)
```

Resultado:

``` text
paciente dormiu pouco
```

Isso reduz diferenças desnecessárias entre:

``` text
Dormiu
DORMIU
dormiu
```

------------------------------------------------------------------------

# 9. Token

Um conceito básico de NLP é o:

> **token**

Simplificando nesta aula, podemos pensar em tokens como unidades do
texto, frequentemente palavras.

Frase:

``` text
paciente dormiu pouco
```

Tokens:

``` text
paciente
dormiu
pouco
```

------------------------------------------------------------------------

# 10. Vocabulário

Imagine estas frases:

``` text
"dormiu pouco"
"recusou almoço"
"tomou medicamento"
```

Podemos criar um vocabulário:

``` text
dormiu
pouco
recusou
almoço
tomou
medicamento
```

Agora podemos verificar quais palavras aparecem em cada frase.

------------------------------------------------------------------------

# 11. Bag of Words

Uma técnica introdutória é:

> **Bag of Words --- saco de palavras**

A ideia é representar o texto pela ocorrência das palavras.

Exemplo:

  Frase                 dormiu   almoço   medicamento
  ------------------- -------- -------- -------------
  dormiu pouco               1        0             0
  recusou almoço             0        1             0
  tomou medicamento          0        0             1

Transformamos:

``` text
TEXTO
```

em:

``` text
NÚMEROS
```

------------------------------------------------------------------------

# 12. O Bag of Words entende o significado?

Não como uma pessoa.

Ele trabalha principalmente com ocorrência/frequência de termos.

Por exemplo:

``` text
"cachorro morde homem"
```

e:

``` text
"homem morde cachorro"
```

podem ter representações muito parecidas em uma abordagem simples.

Isso mostra uma limitação.

Mas é excelente para aprender o princípio:

> **texto precisa ser representado numericamente.**

------------------------------------------------------------------------

# 13. CountVectorizer

O Scikit-Learn possui:

``` python
from sklearn.feature_extraction.text import CountVectorizer
```

Criamos:

``` python
vectorizador = CountVectorizer()
```

Depois:

``` python
X = vectorizador.fit_transform(textos)
```

------------------------------------------------------------------------

# 14. O que o fit_transform faz aqui?

## fit

Aprende o vocabulário existente nos textos de treinamento.

## transform

Transforma os textos em vetores numéricos.

Podemos observar:

``` python
print(
    vectorizador.get_feature_names_out()
)
```

------------------------------------------------------------------------

# 15. Exemplo completo de vetorização

``` python
from sklearn.feature_extraction.text import CountVectorizer

textos = [
    "dormiu pouco",
    "recusou almoço",
    "tomou medicamento"
]

vectorizador = CountVectorizer()

X = vectorizador.fit_transform(textos)

print(
    vectorizador.get_feature_names_out()
)

print(
    X.toarray()
)
```

Agora o computador possui uma matriz numérica.

------------------------------------------------------------------------

# 16. Nosso dataset textual

Exemplo:

  texto                          categoria
  ------------------------------ -------------
  dormiu pouco durante a noite   sono
  acordou várias vezes           sono
  recusou o almoço               alimentação
  comeu todo o café              alimentação
  medicamento administrado       medicação
  remédio atrasou                medicação

Temos novamente:

``` text
FEATURE
```

e:

``` text
TARGET
```

Mas agora a feature original é texto.

------------------------------------------------------------------------

# 17. Feature e target no NLP

``` python
X_texto = dados["texto"]

y = dados["categoria"]
```

Antes do modelo, `X_texto` precisa ser vetorizado.

------------------------------------------------------------------------

# 18. Separar treino e teste primeiro

Uma forma didaticamente correta é separar os textos antes de aprender o
vocabulário.

``` python
from sklearn.model_selection import train_test_split

X_treino_texto, X_teste_texto, y_treino, y_teste = train_test_split(
    X_texto,
    y,
    test_size=0.2,
    random_state=42
)
```

------------------------------------------------------------------------

# 19. Vetorizando treino e teste

``` python
from sklearn.feature_extraction.text import CountVectorizer

vectorizador = CountVectorizer(
    lowercase=True
)

X_treino = vectorizador.fit_transform(
    X_treino_texto
)

X_teste = vectorizador.transform(
    X_teste_texto
)
```

Observe novamente:

``` text
TREINO → fit_transform

TESTE → transform
```

A lógica é semelhante ao que vimos com `StandardScaler`.

------------------------------------------------------------------------

# 20. Por que não fazemos fit no teste?

Porque o teste deve representar dados separados do processo de
aprendizado.

O vocabulário deve ser aprendido com:

``` text
TREINO
```

Depois aplicamos esse vocabulário ao:

``` text
TESTE
```

------------------------------------------------------------------------

# 21. Qual modelo usar?

Para uma introdução a classificação de texto, podemos usar:

``` python
from sklearn.naive_bayes import MultinomialNB
```

O objetivo desta aula não é aprofundar a matemática do Naive Bayes.

Queremos observar o pipeline:

``` text
texto
 ↓
vetor
 ↓
classificador
 ↓
categoria
```

------------------------------------------------------------------------

# 22. Treinando

``` python
modelo = MultinomialNB()

modelo.fit(
    X_treino,
    y_treino
)
```

------------------------------------------------------------------------

# 23. Prevendo

``` python
previsoes = modelo.predict(
    X_teste
)

print(previsoes)
```

------------------------------------------------------------------------

# 24. Avaliando

Podemos utilizar o que aprendemos na Aula 02:

``` python
from sklearn.metrics import accuracy_score

acuracia = accuracy_score(
    y_teste,
    previsoes
)

print(
    "Acurácia:",
    acuracia
)
```

Também podemos usar:

``` python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_teste,
        previsoes
    )
)
```

Agora as aulas começam a se conectar.

------------------------------------------------------------------------

# 25. Nova frase

Imagine:

``` text
"paciente não quis comer o almoço"
```

Primeiro:

``` python
nova_frase = [
    "paciente não quis comer o almoço"
]
```

Depois precisamos utilizar o **mesmo vectorizer**:

``` python
nova_frase_vetor = vectorizador.transform(
    nova_frase
)
```

Finalmente:

``` python
resultado = modelo.predict(
    nova_frase_vetor
)

print(resultado)
```

------------------------------------------------------------------------

# 26. Erro importante

Errado:

``` python
novo_vectorizer = CountVectorizer()

novo_vectorizer.fit_transform(
    nova_frase
)
```

Por quê?

Porque o modelo foi treinado com o vocabulário criado anteriormente.

A nova frase deve passar pelo:

``` python
vectorizador.transform()
```

já treinado.

------------------------------------------------------------------------

# 27. Palavras desconhecidas

E se aparecer uma palavra que não existia no treinamento?

Exemplo:

``` text
"paciente beliscou o jantar"
```

Se `beliscou` nunca apareceu no vocabulário, o `CountVectorizer` não
terá aprendido essa palavra.

Isso ajuda a compreender uma ideia importante:

> **A qualidade e variedade dos exemplos de treinamento importam.**

------------------------------------------------------------------------

# 28. Dataset pequeno

Se treinarmos:

``` text
"almoço" → alimentação
"jantar" → alimentação
```

e quase nenhum outro exemplo, o modelo terá pouca informação.

Machine Learning precisa de:

-   exemplos;
-   diversidade;
-   qualidade;
-   categorias bem definidas.

------------------------------------------------------------------------

# 29. Uma frase pode ter mais de um assunto

Exemplo:

``` text
"Dormiu pouco e recusou o almoço."
```

Temos:

``` text
sono
+
alimentação
```

Nosso modelo introdutório pode estar preparado para retornar apenas uma
categoria.

Esse é um limite do experimento.

Existe uma abordagem chamada:

> **classificação multirrótulo**

Mas ela ficará como evolução futura.

------------------------------------------------------------------------

# 30. NLP no projeto NR-1: cuidado especial

Comentários podem conter informações sensíveis ou permitir identificação
indireta.

Portanto, um sistema real deve considerar:

-   anonimização;
-   acesso restrito;
-   análise agregada;
-   grupos mínimos;
-   LGPD;
-   revisão das regras de uso.

A IA deve apoiar a análise organizacional, não diagnosticar pessoas.

------------------------------------------------------------------------

# 31. NLP no projeto Fiscalização

O projeto também pode combinar:

``` text
TEXTO
+
IMAGEM
+
LOCALIZAÇÃO
```

Nesta disciplina básica, podemos começar somente pelo texto.

Exemplo:

``` text
"Tem uma geladeira abandonada."
        ↓
resíduo eletrônico / móvel
```

Reconhecimento visual é uma especialização posterior.

------------------------------------------------------------------------

# 32. NLP na Agenda

Uma mensagem como:

``` text
"Tenho dentista quinta às duas."
```

pode futuramente ser decomposta em:

``` text
tipo → compromisso
assunto → dentista
data → quinta
hora → 14:00
```

Nesta aula, começamos pelo problema mais simples:

``` text
classificar o texto
```

Extração de entidades pode ser uma evolução.

------------------------------------------------------------------------

# 33. Atividade de fixação 1

Classifique cada abordagem como:

``` text
REGRA
```

ou:

``` text
MACHINE LEARNING
```

1.  

``` python
if "sofá" in texto:
    categoria = "móveis"
```

2.  

Um modelo recebe 500 mensagens previamente classificadas e aprende a
prever categorias.

3.  

``` python
if "reunião" in texto:
    tipo = "compromisso"
```

4.  

Um classificador recebe textos vetorizados e aprende com `fit()`.

------------------------------------------------------------------------

# 34. Atividade de fixação 2

Dado o vocabulário:

``` text
sono
almoço
remédio
```

Represente:

``` text
"sono remédio"
```

usando:

``` text
[sono, almoço, remédio]
```

Resposta esperada conceitualmente:

``` text
[1, 0, 1]
```

------------------------------------------------------------------------

# 35. Prática principal

Cada grupo cria um dataset simples com pelo menos:

``` text
30 a 50 frases
```

e:

``` text
3 categorias
```

Sugestões:

### Cuidadores

``` text
sono
alimentação
medicação
```

### NR-1

``` text
sobrecarga
apoio
comunicação
```

### Fiscalização

``` text
móveis
entulho
eletrônicos
```

### Agenda

``` text
compromisso
tarefa
lembrete
```

------------------------------------------------------------------------

# 36. Etapas da prática

1.  Criar/carregar o dataset.
2.  Observar as categorias.
3.  Separar treino e teste.
4.  Criar `CountVectorizer`.
5.  `fit_transform()` no treino.
6.  `transform()` no teste.
7.  Treinar o classificador.
8.  Fazer previsões.
9.  Calcular acurácia.
10. Observar métricas.
11. Testar frases novas.
12. Registrar erros.

------------------------------------------------------------------------

# 37. Erros são material de estudo

Quando o modelo errar, não apague o erro.

Pergunte:

``` text
Por que ele errou?
```

Possibilidades:

-   poucos exemplos;
-   palavra nunca vista;
-   categorias parecidas;
-   frase ambígua;
-   dataset desbalanceado;
-   texto com mais de um assunto.

Os erros ajudam a entender o modelo.

------------------------------------------------------------------------

# 38. Erros comuns

## Erro 1

Enviar texto diretamente ao classificador numérico.

## Erro 2

Criar um novo `CountVectorizer` para cada frase.

## Erro 3

Fazer `fit_transform()` no teste.

## Erro 4

Treinar com pouquíssimos exemplos e esperar alta generalização.

## Erro 5

Confundir NLP com IA generativa.

## Erro 6

Achar que o Bag of Words compreende linguagem como uma pessoa.

------------------------------------------------------------------------

# 39. Questões de revisão

1.  O que significa NLP?
2.  NLP é sinônimo de chatbot?
3.  O que é um token?
4.  O que é vocabulário?
5.  Por que precisamos transformar texto em números?
6.  O que é Bag of Words?
7.  Para que serve `CountVectorizer`?
8.  O que o `fit()` aprende no vectorizer?
9.  Por que usamos `transform()` no teste?
10. O que acontece com palavras nunca vistas no treinamento?
11. Uma regra com `if` é Machine Learning?
12. Por que precisamos de variedade no dataset?
13. O que é classificação de texto?
14. O que acontece quando uma frase pertence a mais de uma categoria?
15. Qual a diferença entre NLP e IA generativa?

------------------------------------------------------------------------

# 40. Desafio final

Cada grupo deverá coletar cinco frases escritas por colegas que não
participaram da criação do dataset.

Teste o modelo.

Registre:

  Frase   Real   Previsto   Acertou?
  ------- ------ ---------- ----------
                            

Depois explique três erros ou acertos interessantes.

------------------------------------------------------------------------

# 41. Entrega

Notebook:

``` text
03_nlp_classificacao_textos.ipynb
```

Deve conter:

-   dataset;
-   categorias;
-   treino/teste;
-   `CountVectorizer`;
-   vetorização;
-   modelo;
-   previsões;
-   métricas;
-   cinco testes externos;
-   análise dos erros;
-   conclusão.

------------------------------------------------------------------------

# 42. Resumo

``` text
LINGUAGEM HUMANA
       ↓
      NLP
       ↓
PREPARAÇÃO DO TEXTO
       ↓
   VOCABULÁRIO
       ↓
  BAG OF WORDS
       ↓
 COUNTVECTORIZER
       ↓
     NÚMEROS
       ↓
     MODELO
       ↓
   CATEGORIA
```

Na próxima aula veremos outra abordagem:

> Em vez de apenas classificar uma entrada, podemos pedir a um modelo
> para **gerar conteúdo**.

Entraremos em:

> **IA Generativa e Prompt Engineering.**

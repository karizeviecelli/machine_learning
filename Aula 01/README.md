<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 01 – Bem-vindo ao Machine Learning

© @karizeviecelli - 2026
=========================================================
-->

# Aula 01 – Introdução ao Machine Learning

# Módulo 01 –

---

> **Curso:** Machine Learning com Python
>
> **Nível:** Iniciante
>
> **Ferramenta utilizada:** Google Colab
>
> **Autor:** @karizeviecelli - 2026

---

# O que você aprenderá neste módulo

Ao finalizar este módulo você será capaz de:

✔ Entender o objetivo do curso.

✔ Compreender por que Machine Learning se tornou uma das áreas mais importantes da tecnologia.

✔ Identificar aplicações de Machine Learning presentes no seu dia a dia.

✔ Conhecer a metodologia utilizada durante todo o curso.

✔ Preparar o ambiente que será utilizado nas próximas aulas.

---

# Pré-requisitos

Nenhum.

Este curso foi desenvolvido para estudantes que nunca tiveram contato com Machine Learning.

Se você já conhece Python, ótimo.

Se nunca programou para Inteligência Artificial, melhor ainda.

Vamos construir esse conhecimento passo a passo.

---

# Uma conversa antes de começarmos...

Talvez você tenha decidido estudar Machine Learning porque ouviu falar de Inteligência Artificial.

Ou talvez porque viu notícias sobre ChatGPT, carros autônomos, robôs, reconhecimento facial ou geração de imagens.

Independentemente do motivo, existe uma boa notícia.

Você não precisa ser um especialista para começar.

Na verdade, todo profissional da área começou exatamente do mesmo ponto em que você está agora.

Sem conhecer algoritmos.

Sem saber o que é um modelo.

Sem entender palavras como *dataset*, *feature*, *target* ou *treinamento*.

O conhecimento é construído em etapas.

E este curso foi organizado exatamente dessa forma.

Cada aula acrescentará uma nova peça ao quebra-cabeça.

Ao final, você será capaz de construir seus próprios modelos de Machine Learning.

---

# O que realmente é Machine Learning?

Antes de responder essa pergunta, imagine a seguinte situação.

Você trabalha em uma escola.

Todos os anos milhares de alunos realizam provas.

A direção da escola gostaria de descobrir antecipadamente quais alunos possuem maior chance de aprovação.

A primeira ideia seria criar uma regra.

Por exemplo:

```text
Se a nota for maior que 7
e a frequência for maior que 75%

Então

Aluno aprovado.
```

Funciona?

Sim.

Mas apenas para situações muito simples.

Agora imagine centenas de informações diferentes:

- idade;
- curso;
- número de atividades entregues;
- participação em aula;
- quantidade de faltas;
- notas de várias disciplinas;
- histórico escolar.

Como criar manualmente milhares de regras?

Seria praticamente impossível.

É justamente aqui que surge o Machine Learning.

---

# Uma primeira definição

Machine Learning é uma área da Inteligência Artificial que permite que computadores aprendam padrões utilizando exemplos, em vez de depender exclusivamente de regras escritas por programadores.

Em outras palavras...

Nós mostramos vários exemplos.

O algoritmo observa esses exemplos.

Depois tenta descobrir quais padrões existem.

Quando um novo caso aparece, ele utiliza o que aprendeu para fazer uma previsão.

---

# Uma analogia

Imagine uma criança pequena.

Você deseja ensinar o que é um cachorro.

Você poderia entregar um livro dizendo:

- possui quatro patas;
- possui focinho;
- possui orelhas;
- possui cauda.

Mas isso não garante que ela conseguirá reconhecer todos os cachorros.

Alguns são pequenos.

Outros enormes.

Alguns possuem muito pelo.

Outros quase nenhum.

O método mais eficiente normalmente é mostrar exemplos.

```
🐶 Cachorro

🐶 Cachorro

🐶 Cachorro

🐶 Cachorro

🐱 Gato

🐱 Gato

🐱 Gato
```

Depois de observar vários exemplos, a criança começa a perceber padrões.

Na próxima vez que encontrar um cachorro, conseguirá identificá-lo mesmo que nunca tenha visto aquele animal antes.

O Machine Learning funciona de maneira muito parecida.

---

# Pare e pense

Quando você aprende a reconhecer o rosto de um amigo, alguém escreveu uma lista de regras dentro do seu cérebro?

Ou você aprendeu observando exemplos ao longo do tempo?

O computador faz algo semelhante.

Ele observa exemplos.

Aprende padrões.

Depois tenta reconhecer novos casos.

---

# Machine Learning está muito mais perto do que você imagina

Mesmo que você nunca tenha estudado Inteligência Artificial, provavelmente utilizou Machine Learning hoje.

Veja alguns exemplos.

### Spotify

Quando o Spotify recomenda uma nova música, ele analisa seus hábitos.

Quanto mais você utiliza o aplicativo, melhores tendem a ser as recomendações.

---

### Netflix

A ordem dos filmes exibidos não é igual para todos.

Ela depende do seu histórico.

---

### YouTube

Os vídeos recomendados são diferentes para cada pessoa.

O sistema tenta prever aquilo que possui maior chance de despertar seu interesse.

---

### Bancos

Quando uma compra muito diferente do seu comportamento acontece, alguns bancos conseguem identificar uma possível fraude em poucos segundos.

---

### Redes sociais

O conteúdo mostrado para você normalmente é diferente daquele mostrado para outra pessoa.

O sistema aprende continuamente quais publicações chamam mais sua atenção.

---

# Curiosidade

Você perceberá uma característica interessante ao longo deste curso.

Quase todos os exemplos de Machine Learning possuem algo em comum.

Todos utilizam **dados**.

Sem dados...

Não existe Machine Learning.

Por esse motivo, aprenderemos primeiro a trabalhar com dados.

Somente depois construiremos modelos.

Essa é exatamente a mesma sequência utilizada por profissionais da área.

---

# Como este curso foi organizado

Durante o curso utilizaremos sempre o mesmo fluxo de trabalho.

```
Problema

↓

Dados

↓

Exploração

↓

Limpeza

↓

Preparação

↓

Treinamento

↓

Previsão

↓

Avaliação
```

Esse fluxo representa, de forma simplificada, o caminho percorrido em projetos reais.

Você perceberá que escrever algoritmos é apenas uma pequena parte do trabalho.

Grande parte do tempo será dedicada à compreensão dos dados.

---

# Dica do Professor

Não tenha pressa para aprender algoritmos.

Um modelo excelente treinado com dados ruins normalmente produz resultados ruins.

Primeiro aprendemos a trabalhar corretamente com os dados.

Depois treinaremos modelos cada vez mais inteligentes.

---

# Resumo do módulo

Neste primeiro contato você aprendeu que:

- Machine Learning faz parte da Inteligência Artificial.
- O computador aprende utilizando exemplos.
- Os dados são a matéria-prima do Machine Learning.
- O curso seguirá o mesmo fluxo utilizado em projetos profissionais.
- Ainda não construiremos modelos. Primeiro aprenderemos a compreender os dados.

---

# O que você já consegue fazer?

Marque os itens abaixo.

- [ ] Sei explicar, com minhas palavras, o que é Machine Learning.
- [ ] Consigo citar três aplicações de Machine Learning no dia a dia.
- [ ] Entendi que os dados são a base de qualquer projeto de IA.
- [ ] Compreendi como será organizado este curso.

---

# Próximo módulo

No próximo módulo responderemos uma pergunta fundamental:

> **O que é Inteligência Artificial e qual é a diferença entre IA, Machine Learning e Deep Learning?**

Esse conhecimento será utilizado durante todo o restante do curso.

---

© @karizeviecelli - 2026

<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 02 – Inteligência Artificial na Prática

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 02 – Inteligência Artificial na Prática

---

> **Nível:** 🟢 Iniciante

---

# Objetivos

Ao concluir este módulo você será capaz de:

✔ Explicar o que é Inteligência Artificial.

✔ Diferenciar Inteligência Artificial de Machine Learning.

✔ Entender por que os dados são tão importantes.

✔ Conhecer o conceito de GIGO.

---

# Antes de começarmos...

Feche os olhos por alguns segundos e tente responder mentalmente.

**Quantas vezes você utilizou Inteligência Artificial hoje?**

Talvez sua resposta seja:

> "Nenhuma."

Na verdade, é muito provável que você tenha utilizado dezenas de vezes.

Sem perceber.

---

# Onde a Inteligência Artificial está presente?

Observe algumas situações comuns.

## Ao desbloquear o celular

Quando o aparelho reconhece seu rosto.

Existe Inteligência Artificial trabalhando.

---

## Ao assistir Netflix

A ordem dos filmes muda para cada pessoa.

Isso não acontece por acaso.

Existe um algoritmo tentando descobrir quais filmes possuem maior chance de agradar você.

---

## Google Maps

Quando o aplicativo muda automaticamente a rota por causa do trânsito.

Existe Inteligência Artificial analisando milhares de informações em tempo real.

---

# Então...

A Inteligência Artificial não é um robô.

Ela não é apenas o ChatGPT.

Ela não é apenas um carro autônomo.

Na verdade, Inteligência Artificial é um conjunto enorme de técnicas utilizadas para resolver problemas.

Uma dessas técnicas chama-se Machine Learning.

---

# IA × Machine Learning

Imagine uma universidade.

```
Universidade

│

├── Engenharia

├── Direito

├── Medicina

└── Computação
```

Dentro da Computação existem vários cursos.

Da mesma forma acontece com a Inteligência Artificial.

```
Inteligência Artificial

│

├── Machine Learning

├── Deep Learning

├── Visão Computacional

├── Processamento de Linguagem Natural

└── Sistemas Especialistas
```

Machine Learning é apenas uma parte da Inteligência Artificial.

Ao longo deste curso estudaremos essa parte.

Mais adiante conheceremos Deep Learning.

---

# O ingrediente mais importante

Você pode imaginar que o segredo do Machine Learning está nos algoritmos.

Na verdade...

O ingrediente mais importante são os **dados**.

Sem dados...

Não existe aprendizado.

Imagine ensinar uma criança a reconhecer frutas.

Se ela nunca viu uma maçã.

Nunca viu uma banana.

Nunca viu uma laranja.

Ela conseguirá reconhecê-las?

Provavelmente não.

O computador funciona da mesma maneira.

Ele precisa observar exemplos.

Esses exemplos são chamados de **dados**.

---

# GIGO

Existe um princípio muito famoso na Computação.

Ele surgiu há muitas décadas.

Mesmo assim continua extremamente atual.

Seu nome é:

## GIGO

**Garbage In, Garbage Out**

Em português.

> **"Se entrar lixo, sairá lixo."**

---

# O que isso significa?

Imagine que você deseje ensinar um computador a reconhecer alunos aprovados.

Porém seu conjunto de dados possui vários problemas.

```
Nome           Nota

Ana             8,5

João            ???

Maria           25

Carlos          -8

Fernanda        abc
```

Observe os erros.

- notas inexistentes;

- textos onde deveriam existir números;

- valores negativos;

- notas maiores que dez.

Agora pense.

Mesmo utilizando o melhor algoritmo do mundo.

Ele aprenderá corretamente?

A resposta é:

**Não.**

---

# Analogia

Imagine um chef de cozinha.

Ele possui:

- excelente fogão;

- excelentes panelas;

- muita experiência.

Mas alguém entrega ingredientes estragados.

O resultado será um bom prato?

Claro que não.

Em Machine Learning acontece exatamente o mesmo.

```
Dados ruins

↓

Modelo ruim

↓

Previsões ruins
```

---

# Um conceito que você verá durante todo o curso

Nunca esqueça.

```
Qualidade dos Dados

>

Qualidade do Algoritmo
```

Muitas pessoas acreditam que trocar o algoritmo resolverá um problema.

Na maioria das vezes.

O verdadeiro problema está nos dados.

É por isso que Cientistas de Dados passam grande parte do tempo limpando e preparando informações.

Em muitos projetos reais, mais de 70% do tempo é dedicado à preparação dos dados.

---

# Curiosidade

Empresas como bancos, hospitais, seguradoras e indústrias possuem equipes inteiras dedicadas exclusivamente à qualidade dos dados.

Esses profissionais nem sempre desenvolvem modelos.

Seu trabalho é garantir que os dados utilizados pelos modelos sejam confiáveis.

---

# Pare e pense

Imagine que uma escola deseja prever quais alunos serão aprovados.

Se metade das notas estiver errada.

As previsões serão confiáveis?

Provavelmente não.

Essa é exatamente a importância do GIGO.

---

# No mercado

Uma frase muito conhecida entre Cientistas de Dados é:

> **"Os dados alimentam o modelo."**

Quanto melhor for a alimentação.

Melhor tende a ser o resultado.

---

# Resumo

Neste módulo você aprendeu que:

- Inteligência Artificial está presente em diversas aplicações do cotidiano.
- Machine Learning é uma área da Inteligência Artificial.
- Dados são a matéria-prima dos modelos.
- GIGO significa **Garbage In, Garbage Out**.
- Dados ruins produzem modelos ruins.

---

# Mini desafio

Pesquise cinco aplicativos que você utiliza diariamente.

Para cada um deles responda:

- Onde existe Inteligência Artificial?
- O sistema provavelmente utiliza Machine Learning?
- Quais dados ele deve utilizar para aprender?

Anote suas respostas.

Voltaremos a essa atividade nas próximas aulas.

---

© @karizeviecelli - 2026

<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 03 – Como um Computador Aprende?

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 03 – Como um Computador Aprende?

---

> **Nível:** 🟢 Iniciante

---

# Objetivos

Ao concluir este módulo você será capaz de:

✔ Entender como ocorre o aprendizado de um algoritmo.

✔ Compreender o conceito de padrão.

✔ Entender por que exemplos são importantes.

✔ Conhecer o ciclo básico de treinamento.

---


# O computador não pensa

Este é um dos maiores mitos sobre Inteligência Artificial.

Muitas pessoas dizem:

> "O computador pensa."

Na realidade...

O computador **não pensa como um ser humano**.

Ele faz cálculos.

Muitos cálculos.

Milhões deles em poucos segundos.

Esses cálculos permitem identificar padrões.

---

# O segredo está nos padrões

Imagine as seguintes informações.

| Horas de Estudo | Aprovado |
|----------------:|-----------|
| 1 | Não |
| 2 | Não |
| 3 | Não |
| 4 | Sim |
| 5 | Sim |
| 6 | Sim |

Observe atentamente.

Existe algum padrão?

Provavelmente você percebeu rapidamente que alunos que estudam mais possuem maior chance de aprovação.

Você descobriu um padrão.

O algoritmo faz exatamente isso.

---

# Outro exemplo

Agora imagine um banco.

Ele possui milhares de registros.

| Salário | Comprou um carro? |
|---------:|-------------------|
| 1800 | Não |
| 2200 | Não |
| 5000 | Sim |
| 6500 | Sim |
| 8000 | Sim |

Novamente existe um padrão.

Quanto maior o salário...

Maior parece ser a probabilidade de compra.

O algoritmo procura relações semelhantes.

---

# Atenção!

Observe que utilizamos a palavra:

> **Parece**

Isso acontece porque Machine Learning trabalha com probabilidades.

Nem toda pessoa que ganha bem compra um carro.

Nem toda pessoa que estuda muito será aprovada.

Os algoritmos procuram tendências.

Não certezas absolutas.

---

# Aprender é encontrar padrões

Podemos resumir o Machine Learning da seguinte forma.

```
Exemplos

↓

Análise

↓

Identificação de padrões

↓

Construção do modelo

↓

Novas previsões
```

Esse processo recebe o nome de **treinamento**.

---

# O que é treinamento?

Treinar um modelo significa apresentar diversos exemplos para que ele aprenda relações entre as informações.

Imagine um professor corrigindo milhares de provas.

Depois de algum tempo ele percebe padrões.

Alunos que estudam diariamente costumam obter melhores notas.

Alunos muito faltosos costumam apresentar maior dificuldade.

O algoritmo faz algo parecido.

Quanto mais exemplos recebe, melhor tende a compreender os padrões.

---

# Analogia

Imagine um médico recém-formado.

No primeiro dia de trabalho ele possui pouco conhecimento prático.

Após atender milhares de pacientes.

Observar exames.

Comparar sintomas.

Ele passa a reconhecer doenças com maior facilidade.

Isso aconteceu porque acumulou experiência.

O algoritmo também acumula experiência.

A diferença é que essa experiência vem dos dados.

---

# O ciclo do aprendizado

Sempre que treinamos um modelo, seguimos praticamente o mesmo processo.

```
Receber os dados

↓

Organizar

↓

Treinar

↓

Aprender padrões

↓

Realizar previsões
```

Mais adiante você perceberá que esse ciclo aparecerá em praticamente todos os algoritmos estudados.

---

# Curiosidade

Alguns modelos modernos são treinados utilizando milhões de exemplos.

Em projetos de reconhecimento de imagens, por exemplo, um algoritmo pode analisar milhões de fotografias antes de aprender a identificar corretamente um objeto.

Isso explica por que o treinamento pode levar horas ou até dias em projetos muito grandes.

---

# No mercado

Uma das primeiras perguntas feitas por um Cientista de Dados costuma ser:

> "Temos dados suficientes para treinar um modelo?"

Sem exemplos suficientes, o algoritmo terá dificuldade para aprender padrões confiáveis.

---

# Erro comum

Muitos iniciantes acreditam que basta utilizar um algoritmo famoso para obter excelentes resultados.

Na prática, isso raramente acontece.

Mesmo um algoritmo muito sofisticado produzirá resultados ruins se os exemplos apresentados forem insuficientes ou de baixa qualidade.

Lembre-se do conceito estudado anteriormente:

```
Garbage In

↓

Garbage Out
```

---

# Conectando os conceitos

Até agora aprendemos três ideias fundamentais.

```
Inteligência Artificial

↓

Machine Learning

↓

Dados
```

Agora acrescentamos uma quarta peça.

```
Dados

↓

Padrões

↓

Modelo

↓

Previsões
```

Esse será o fluxo utilizado durante todo o curso.

---

# O que aprenderemos a seguir?

Você já sabe que o computador aprende observando exemplos.

Mas...

Existem diferentes formas de aprender.

Alguns modelos recebem a resposta correta durante o treinamento.

Outros não.

Essa diferença dá origem aos principais tipos de Machine Learning.

É exatamente isso que estudaremos no próximo módulo.

---

# Resumo

Neste módulo você aprendeu que:

- O computador não pensa como um ser humano.
- Os algoritmos aprendem identificando padrões.
- O treinamento consiste em apresentar exemplos ao algoritmo.
- Machine Learning trabalha com probabilidades.
- Dados de qualidade são essenciais para que os padrões encontrados sejam confiáveis.

---

# Checklist

Antes de continuar, verifique se você consegue responder às seguintes perguntas:

- [ ] O computador realmente "pensa"?
- [ ] O que significa encontrar um padrão?
- [ ] O que é treinamento?
- [ ] Por que um algoritmo precisa de exemplos?
- [ ] Qual é a relação entre treinamento e previsão?

---

© @karizeviecelli - 2026

<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 04 – Conhecendo os Dados

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 04 – Conhecendo os Dados

---

> **Nível:** 🟢 Iniciante

---

# Onde estamos?

Observe novamente o ciclo de um projeto de Machine Learning.

```text
Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados   ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Perceba que **ainda não estamos treinando algoritmos**.

Antes disso, precisamos conhecer os dados.

Esse é exatamente o primeiro trabalho realizado por um Cientista de Dados.

---

# Imagine a seguinte situação

Você foi contratado por uma escola.

No primeiro dia de trabalho o diretor entrega um arquivo chamado:

```
alunos.csv
```

Ele diz:

> "Gostaríamos de utilizar Inteligência Artificial para prever quais alunos possuem maior chance de aprovação."

Você abre o arquivo.

E agora?

Você começa criando um algoritmo?

Não.

Primeiro você precisa entender o que recebeu.

---

# O que é um Dataset?

A palavra **dataset** pode ser traduzida como:

> **Conjunto de Dados**

Na prática, um dataset é um conjunto organizado de informações sobre algum assunto.

Por exemplo:

- alunos;
- clientes;
- produtos;
- pacientes;
- veículos;
- imóveis.

Sempre que existir um conjunto organizado de informações, provavelmente estamos diante de um dataset.

---

# Analogia

Imagine uma biblioteca.

Cada livro contém diversas informações.

- título;
- autor;
- editora;
- ano;
- quantidade de páginas.

Agora imagine uma tabela onde cada linha representa um livro.

Essa tabela é um dataset.

---

# Estrutura de um Dataset

Observe a tabela abaixo.

| ID | Nome | Idade | Curso | Nota |
|----|-------|--------|--------|------|
| 1 | Ana | 18 | Python | 8.5 |
| 2 | João | 20 | Java | 6.2 |
| 3 | Maria | 19 | Python | 9.4 |

Essa pequena tabela já é um dataset.

Mesmo contendo apenas três registros.

---

# Linhas

Cada linha representa um elemento do mundo real.

Neste exemplo.

```
Linha 1

↓

Ana
```

```
Linha 2

↓

João
```

Cada aluno ocupa uma linha.

Por isso também chamamos as linhas de:

- registros;
- observações;
- instâncias;
- amostras (dependendo do contexto).

Durante este curso utilizaremos principalmente o termo **registro**.

---

# Colunas

Agora observe apenas os títulos.

| ID | Nome | Idade | Curso | Nota |

Cada coluna representa uma característica.

Ou seja.

Cada coluna descreve alguma informação sobre o aluno.

---

# Registro × Coluna

É muito comum estudantes confundirem esses conceitos.

Observe.

```
                COLUNAS

 ID | Nome | Curso | Nota

----------------------------

1 | Ana | Python | 8.5

↑

Registro
```

Uma linha inteira corresponde a um registro.

Uma coluna corresponde a uma característica.

---

# Outro exemplo

Imagine um hospital.

| Nome | Idade | Peso | Pressão |

Cada paciente ocupa uma linha.

Cada informação ocupa uma coluna.

O conceito continua exatamente o mesmo.

---

# Quais tipos de informações podem existir?

Nem todas as colunas armazenam o mesmo tipo de dado.

Veja alguns exemplos.

| Coluna | Tipo |
|---------|------|
| Nome | Texto |
| Idade | Número inteiro |
| Nota | Número decimal |
| Data da Matrícula | Data |
| Aprovado | Verdadeiro/Falso ou Sim/Não |

Esses tipos serão muito importantes quando começarmos a utilizar o Pandas.

---

# Dados estruturados

São aqueles organizados em linhas e colunas.

Exemplos.

- CSV

- Excel

- Banco de Dados

São exatamente esses dados que utilizaremos durante boa parte deste curso.

---

# Dados não estruturados

Agora pense em uma fotografia.

Ela possui linhas?

Colunas?

Registros?

Não.

O mesmo acontece com:

- vídeos;

- áudios;

- documentos PDF;

- e-mails;

- mensagens.

Esses são chamados de **dados não estruturados**.

---

# Dados semiestruturados

Existe ainda um terceiro grupo.

Os dados semiestruturados.

Como.

```
JSON
```

```
XML
```

Eles possuem organização.

Mas não utilizam necessariamente linhas e colunas.

Mais adiante você verá que muitas APIs trabalham exatamente com esse formato.

---

# Curiosidade

Você sabia que grande parte dos dados produzidos atualmente não está em tabelas?

Fotos publicadas nas redes sociais.

Vídeos.

Mensagens.

Áudios.

Documentos.

Tudo isso também pode ser utilizado em projetos de Inteligência Artificial.

---

# No Mercado

Uma das primeiras perguntas feitas por um Cientista de Dados costuma ser:

> "Como esse dataset foi produzido?"

Dependendo da resposta.

Os cuidados durante a análise podem mudar completamente.

---

# Erro comum

Muitos iniciantes acreditam que basta abrir um arquivo CSV e começar a treinar um modelo.

Na prática.

Antes de qualquer treinamento precisamos responder perguntas como:

- Quantas linhas existem?

- Quantas colunas?

- Existem dados faltando?

- Existem informações duplicadas?

- Existem valores incorretos?

Essas perguntas serão respondidas nas próximas aulas.

---

# Pare e Pense

Você recebeu um arquivo chamado:

```
clientes.csv
```

Sem abrir o arquivo.

Responda.

Que tipo de informações você acredita que encontrará?

Discuta sua resposta com seus colegas.

---

# Resumo

Neste módulo você aprendeu que:

- Dataset significa conjunto de dados.
- Cada linha representa um registro.
- Cada coluna representa uma característica.
- Existem diferentes tipos de dados.
- Nem todos os dados estão organizados em tabelas.

---

# Preparando o próximo módulo

Agora que você já conhece a estrutura de um dataset, chegou o momento de descobrir que **nem todas as colunas possuem a mesma importância**.

Algumas serão utilizadas para ensinar o algoritmo.

Outras conterão exatamente a resposta que desejamos prever.

Esses dois conceitos recebem os nomes de:

- **Features**
- **Target**

Eles serão a base para todo o restante do curso.

---

© @karizeviecelli - 2026


<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 05 – Features e Target

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 05 – Features e Target


> **Nível:** 🟢 Iniciante

---

# Onde estamos?

Observe novamente o ciclo de um projeto de Machine Learning.

```text
CICLO DE VIDA DE UM PROJETO DE MACHINE LEARNING

Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features   ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Até agora aprendemos:

- o que é Machine Learning;
- como um computador aprende;
- o que é um dataset.

Agora precisamos responder outra pergunta.

> **Como o algoritmo sabe quais informações deve utilizar para aprender?**

---

# Vamos imaginar uma situação

Uma escola deseja prever quais alunos serão aprovados.

Ela possui a seguinte tabela.

| Nome | Idade | Frequência | Nota | Aprovado |
|------|-------:|-----------:|------:|-----------|
| Ana | 18 | 95 | 8.5 | Sim |
| João | 19 | 60 | 5.5 | Não |
| Maria | 20 | 92 | 9.4 | Sim |
| Pedro | 18 | 68 | 6.1 | Não |

Observe atentamente.

Qual informação queremos descobrir?

A resposta é:

```
Aprovado
```

Ainda não sabemos se um novo aluno será aprovado.

Queremos que o computador descubra isso.

---

# Como um professor faria?

Imagine um professor analisando os históricos dos alunos.

Ele observa:

- idade;
- frequência;
- nota.

Depois compara essas informações com o resultado final.

Com o tempo ele percebe padrões.

O algoritmo fará exatamente a mesma coisa.

---

# O que são Features?

**Feature** significa **característica**.

São as informações que descrevem um registro e que serão utilizadas pelo algoritmo durante o treinamento.

No nosso exemplo.

| Nome | Idade | Frequência | Nota | Aprovado |
|------|-------:|-----------:|------:|-----------|

As features podem ser:

- Idade
- Frequência
- Nota

Essas informações ajudam o algoritmo a encontrar padrões.

---

# O que NÃO é Feature?

Observe novamente.

| Nome | Idade | Frequência | Nota | Aprovado |
|------|-------:|-----------:|------:|-----------|

Será que o **Nome** ajuda o algoritmo?

Normalmente não.

Saber que o aluno se chama Ana ou João não indica, por si só, se ele será aprovado.

Por isso, em muitos projetos, colunas como:

- Nome
- CPF
- RG
- Matrícula
- Código

são removidas antes do treinamento.

Essas colunas identificam pessoas, mas não ajudam o modelo a aprender.

---

# O que é o Target?

O **Target** é exatamente aquilo que desejamos prever.

Em alguns materiais você também encontrará o termo **Label**.

No nosso exemplo.

```
Target

↓

Aprovado
```

O algoritmo observa as features para aprender a prever o target.

---

# Uma analogia

Imagine um médico.

Ele observa várias informações do paciente.

- temperatura;
- pressão arterial;
- idade;
- exames.

Essas são as **features**.

Depois responde:

```
Paciente possui gripe?

Sim

ou

Não
```

Essa resposta é o **target**.

---

# Outro exemplo

Imagine uma imobiliária.

Ela deseja prever o preço de uma casa.

| Área | Quartos | Garagem | Preço |
|------:|---------:|---------:|-------:|
| 90 | 2 | Sim | 350.000 |
| 180 | 3 | Sim | 690.000 |
| 60 | 1 | Não | 210.000 |

Quais são as features?

- Área
- Quartos
- Garagem

Qual é o target?

```
Preço
```

---

# Mais um exemplo

Agora pense em um hospital.

| Idade | Pressão | Febre | Doença |
|-------:|---------:|--------|----------|
| 18 | 12x8 | Sim | Gripe |
| 65 | 18x10 | Não | Hipertensão |

Features:

- Idade
- Pressão
- Febre

Target:

```
Doença
```

---

# Existe sempre um Target?

Depende.

Nos problemas de **Aprendizado Supervisionado**, sim.

Nos problemas de **Aprendizado Não Supervisionado**, não.

Mais adiante veremos exemplos em que o algoritmo recebe apenas as features e precisa descobrir sozinho padrões e agrupamentos.

---

# X e y

Você perceberá que praticamente todos os exemplos em Python utilizam duas letras:

```python
X
```

e

```python
y
```

Mas por quê?

Essa é uma convenção adotada na comunidade de Machine Learning.

Ela vem da Matemática e foi incorporada por bibliotecas como o Scikit-Learn.

Na prática:

```text
X

↓

Features
```

```text
y

↓

Target
```

Sempre que você encontrar um código como este:

```python
X = df[["Idade", "Frequencia", "Nota"]]

y = df["Aprovado"]
```

já saberá exatamente o que está acontecendo.

---

# Importante

Observe que:

```python
X
```

possui várias colunas.

Enquanto

```python
y
```

possui apenas uma.

Isso acontece porque normalmente utilizamos várias informações para prever apenas uma resposta.

---

# Curiosidade

Alguns modelos trabalham com centenas ou milhares de features.

Modelos utilizados para reconhecimento facial, por exemplo, analisam milhares de características de uma imagem antes de tomar uma decisão.

---

# No mercado

Uma frase muito conhecida entre Cientistas de Dados é:

> "A escolha das features é tão importante quanto a escolha do algoritmo."

Em muitos projetos, melhorar as features gera resultados melhores do que trocar o algoritmo.

---

# Erro comum

Um erro muito frequente é colocar o target dentro das features.

Imagine treinar um algoritmo utilizando a coluna **Aprovado** como feature e também como target.

Seria como entregar o gabarito da prova junto com as perguntas.

O modelo pareceria excelente durante o treinamento, mas falharia quando fosse utilizado na prática.

---

# Pare e Pense

Observe a tabela.

| Idade | Salário | Comprou |
|-------:|---------:|----------|
| 18 | 1800 | Não |
| 25 | 3500 | Sim |
| 32 | 5800 | Sim |

Responda.

Quais colunas representam as features?

____________________

Qual coluna representa o target?

____________________

---

# Resumo

Neste módulo você aprendeu que:

- Features são as informações utilizadas pelo algoritmo para aprender.
- Target é a resposta que desejamos prever.
- X representa o conjunto de features.
- y representa o target.
- Nem toda coluna deve ser utilizada durante o treinamento.

---

# Preparando o próximo módulo

Agora você já conhece:

- datasets;
- registros;
- colunas;
- features;
- target.

Chegou o momento de conhecer os três principais tipos de Machine Learning:

- Aprendizado Supervisionado;
- Aprendizado Não Supervisionado;
- Aprendizado por Reforço.

Depois disso estaremos prontos para escrever nosso primeiro código em Python utilizando a biblioteca Pandas.

---

© @karizeviecelli - 2026

<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 06 – Tipos de Machine Learning

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 06 – Tipos de Machine Learning

> **Nível:** 🟢 Iniciante

---

# Onde estamos?

```text
CICLO DE VIDA DE UM PROJETO DE MACHINE LEARNING

Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features
        │
        ▼
Treinamento ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Agora já sabemos:

- o que é um dataset;
- quais são as features;
- qual é o target.

Mas ainda existe uma pergunta importante.

> **Todo algoritmo aprende da mesma forma?**

A resposta é:

**Não.**

Existem diferentes maneiras de ensinar um computador.

---

# Existem três grandes formas de aprendizado

```
Machine Learning

│

├── Supervisionado

├── Não Supervisionado

└── Aprendizado por Reforço
```

Durante este curso utilizaremos principalmente o primeiro.

Mas conhecer os três é fundamental.

---

# Aprendizado Supervisionado

Imagine um professor corrigindo provas.

Cada questão possui uma resposta correta.

O professor consegue dizer ao aluno:

✔ Acertou.

❌ Errou.

Existe alguém orientando o aprendizado.

No Machine Learning acontece a mesma coisa.

O algoritmo recebe exemplos **e também recebe a resposta correta**.

Observe.

| Horas de Estudo | Frequência | Aprovado |
|----------------:|-----------:|-----------|
| 2 | 60 | Não |
| 4 | 82 | Sim |
| 6 | 95 | Sim |
| 1 | 50 | Não |

Perceba que existe uma coluna contendo a resposta.

```
Aprovado
```

Ela orienta o treinamento.

Por isso chamamos esse processo de **Aprendizado Supervisionado**.

---

# O que o algoritmo faz?

Durante o treinamento ele tenta responder à seguinte pergunta:

> Existe alguma relação entre as features e o target?

No nosso exemplo:

```
Horas de Estudo

+

Frequência

↓

Aprovado
```

O objetivo do algoritmo é descobrir essa relação.

---

# Onde usamos Aprendizado Supervisionado?

Ele é utilizado quando já conhecemos o resultado correto.

Alguns exemplos:

### Educação

Prever aprovação.

---

### Bancos

Detectar fraudes.

---

### Saúde

Auxiliar no diagnóstico de doenças.

---

### Comércio

Prever se um cliente fará uma compra.

---

### Marketing

Prever cancelamento de assinaturas.

---

# Dois tipos de problemas supervisionados

Dentro do Aprendizado Supervisionado existem dois grupos muito importantes.

---

## 1. Classificação

Na classificação queremos descobrir **uma categoria**.

Exemplos:

```
Sim

Não
```

```
Fraude

Normal
```

```
Aprovado

Reprovado
```

```
Gato

Cachorro
```

Observe.

Sempre escolhemos uma categoria.

---

## 2. Regressão

Na regressão queremos descobrir **um valor numérico**.

Exemplos.

Preço de uma casa.

```
R$ 420.000
```

Temperatura.

```
28 °C
```

Quantidade de vendas.

```
153 produtos
```

Agora não estamos escolhendo categorias.

Estamos prevendo números.

---

# Comparando

| Classificação | Regressão |
|---------------|-----------|
| Sim ou Não | Valor numérico |
| Fraude ou Normal | Temperatura |
| Gato ou Cachorro | Preço |
| Aprovado ou Reprovado | Salário |

---

# Aprendizado Não Supervisionado

Agora imagine outra situação.

Você recebe milhares de registros.

Mas ninguém informa qual é a resposta correta.

Veja.

| Cliente | Idade | Salário |
|----------|-------:|---------:|
| A | 20 | 2200 |
| B | 22 | 2500 |
| C | 60 | 12000 |
| D | 58 | 9800 |

Não existe uma coluna dizendo:

```
Cliente VIP

Sim

Não
```

O algoritmo precisa descobrir padrões sozinho.

---

# Analogia

Imagine entrar em uma gaveta cheia de meias.

Ninguém separou por cor.

Você começa naturalmente a criar grupos.

```
Pretas

↓

Brancas

↓

Coloridas
```

O algoritmo faz algo semelhante.

Ele agrupa elementos parecidos.

---

# Onde esse tipo de algoritmo é utilizado?

Segmentação de clientes.

Análise de comportamento.

Agrupamento de produtos.

Organização automática de documentos.

Sistemas de recomendação.

---

# Aprendizado por Reforço

Existe ainda um terceiro tipo.

O Aprendizado por Reforço.

Nesse caso.

O algoritmo aprende tentando.

Errando.

Corrigindo.

Tentando novamente.

Até encontrar a melhor solução.

---

# Analogia

Imagine ensinar um cachorro.

Quando ele realiza a ação correta.

Recebe uma recompensa.

Quando erra.

Não recebe.

Aos poucos aprende qual comportamento produz melhores resultados.

---

# Onde encontramos esse tipo de aprendizado?

Jogos.

Robótica.

Carros autônomos.

Controle de drones.

Automação industrial.

---

# Comparando os três tipos

| Tipo | Existe resposta correta? | Principal objetivo |
|-------|--------------------------|--------------------|
| Supervisionado | Sim | Fazer previsões |
| Não Supervisionado | Não | Descobrir padrões |
| Reforço | Não | Aprender por tentativa e erro |

---

# Curiosidade

Você já ouviu falar em AlphaGo?

Foi um sistema desenvolvido para jogar Go, um dos jogos de estratégia mais complexos do mundo.

Grande parte do aprendizado ocorreu por meio de Aprendizado por Reforço.

O sistema jogava milhões de partidas contra ele mesmo, aprendendo continuamente.

---

# No mercado

Quando você procurar vagas para Cientista de Dados ou Engenheiro de Machine Learning, verá frequentemente termos como:

- Classification
- Regression
- Clustering

Agora você já sabe que eles significam:

- Classificação;
- Regressão;
- Agrupamento.

---

# Erro comum

Muitos iniciantes acreditam que todos os algoritmos servem para qualquer problema.

Na prática.

Primeiro identificamos o tipo do problema.

Só depois escolhemos o algoritmo.

---

# Pare e Pense

Classifique os problemas abaixo.

| Problema | Tipo |
|-----------|------|
| Descobrir se um e-mail é spam | __________ |
| Prever o preço de um carro | __________ |
| Agrupar clientes semelhantes | __________ |
| Ensinar um robô a caminhar | __________ |

---

# Resumo

Neste módulo você aprendeu que existem três grandes tipos de Machine Learning.

```
Machine Learning

↓

Supervisionado

↓

Não Supervisionado

↓

Reforço
```

Também conheceu dois problemas muito importantes.

```
Classificação

↓

Categorias
```

```
Regressão

↓

Valores Numéricos
```

---

# Preparando o próximo módulo

Até agora aprendemos toda a base conceitual.

Agora chegou o momento de começar a programar.

No próximo módulo você conhecerá uma das bibliotecas mais importantes da Ciência de Dados.

> **Pandas**

Ela será nossa principal ferramenta para trabalhar com tabelas durante todo o curso.

---

© @karizeviecelli - 2026


<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 07 – Conhecendo o Pandas

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 07 – Conhecendo o Pandas

---

> **Nível:** 🟢 Iniciante

---

# Onde estamos?

```text
CICLO DE VIDA DE UM PROJETO DE MACHINE LEARNING

Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Chegou o momento de começar a trabalhar com dados utilizando Python.

Para isso utilizaremos uma biblioteca chamada **Pandas**.

---

# O que é uma biblioteca?

Antes de conhecer o Pandas, precisamos entender um conceito muito importante do Python.

Imagine que você queira construir uma casa.

Você poderia fabricar:

- os tijolos;
- o cimento;
- a areia;
- as portas;
- as janelas.

Mas isso levaria muito tempo.

O mais inteligente é utilizar materiais que já existem.

Na programação acontece exatamente a mesma coisa.

Milhares de desenvolvedores criam códigos reutilizáveis para resolver problemas comuns.

Esses conjuntos de códigos recebem o nome de **bibliotecas**.

---

# Analogia

Imagine uma caixa de ferramentas.

```
🧰 Caixa de Ferramentas

🔨 Martelo

🔧 Chave de Fenda

🪛 Chave Philips

📏 Trena
```

Cada ferramenta possui uma finalidade.

Uma biblioteca funciona como uma caixa de ferramentas para o programador.

Cada função resolve um problema específico.

---

# O que é o Pandas?

O Pandas é uma biblioteca desenvolvida para facilitar o trabalho com dados.

Ele permite:

✔ Ler arquivos CSV.

✔ Ler planilhas Excel.

✔ Organizar tabelas.

✔ Filtrar informações.

✔ Corrigir erros.

✔ Calcular médias.

✔ Preparar datasets para Machine Learning.

Em praticamente todo projeto de Ciência de Dados existe alguma etapa utilizando Pandas.

---

# Por que o nome "Pandas"?

Apesar de muitas pessoas associarem o nome ao animal panda, a origem está relacionada à expressão:

**PANel DAta**

ou seja, dados em formato tabular (linhas e colunas).

Com o tempo, o nome ficou conhecido simplesmente como **Pandas**.

---

# O Pandas substitui o Excel?

Não.

Na verdade, eles se complementam.

Veja uma comparação.

| Excel | Pandas |
|--------|--------|
| Interface gráfica | Código Python |
| Ideal para pequenas análises | Ideal para grandes volumes de dados |
| Uso manual | Automação |
| Fórmulas | Programação |

Imagine que você precise corrigir um arquivo com **500 mil linhas**.

No Excel isso pode ser lento e trabalhoso.

Com Pandas, podemos realizar a mesma tarefa utilizando poucas linhas de código.

---

# O que é um DataFrame?

O DataFrame é a principal estrutura de dados do Pandas.

Pense nele como uma planilha inteligente.

Observe.

| Nome | Idade | Curso |
|------|-------:|--------|
| Ana | 18 | Python |
| João | 20 | Java |
| Maria | 19 | Front-end |

Essa tabela é um DataFrame.

Quase tudo o que faremos durante este curso será realizado sobre DataFrames.

---

# Primeiro contato com o Google Colab

Durante este curso utilizaremos o Google Colab.

Ele é um ambiente gratuito que permite executar código Python diretamente pelo navegador.

Não será necessário instalar Python no computador.

Também não precisaremos configurar bibliotecas manualmente.

O Colab já possui o Pandas instalado.

---

# Abrindo o Google Colab

Acesse:

https://colab.research.google.com

Ao abrir o Colab você encontrará um notebook vazio.

Esse notebook é composto por células.

Existem dois tipos principais.

### Células de Texto

Utilizadas para documentação.

São escritas em Markdown.

---

### Células de Código

Executam comandos Python.

Durante este curso utilizaremos principalmente esse tipo de célula.

---

# Nosso primeiro código

Chegou o momento de escrever a primeira linha de código do curso.

```python
import pandas as pd
```

Pode parecer apenas uma linha.

Mas ela possui muito significado.

Vamos analisá-la.

---

# Linha por linha

```python
import pandas as pd
```

## import

A palavra **import** significa:

> "Trazer uma biblioteca para dentro do programa."

Sempre que precisarmos utilizar uma biblioteca, utilizaremos o comando `import`.

---

## pandas

É o nome da biblioteca.

Nesse momento estamos dizendo ao Python qual biblioteca desejamos utilizar.

---

## as

A palavra **as** significa:

> "Utilize um apelido."

---

## pd

Esse é o apelido escolhido para a biblioteca Pandas.

Poderíamos escrever:

```python
import pandas as biblioteca_de_dados
```

O código funcionaria.

Mas praticamente toda a comunidade Python utiliza `pd`.

Por isso seguiremos o mesmo padrão.

---

# Por que usamos um apelido?

Imagine escrever isso diversas vezes.

```python
pandas.DataFrame()
```

Agora compare.

```python
pd.DataFrame()
```

Muito mais simples.

Além disso, facilita a leitura e segue a convenção utilizada na documentação oficial do Pandas.

---

# Executando no Google Colab

Digite o código abaixo em uma célula.

```python
import pandas as pd
```

Depois pressione:

**Shift + Enter**

Se nenhuma mensagem de erro aparecer, significa que a biblioteca foi carregada com sucesso.

---

# Experimente

Altere o código para:

```python
import pandas as tabela
```

Agora tente executar.

Depois utilize:

```python
tabela.DataFrame()
```

Perceba que o Python aceita outro apelido.

Mas isso **não é recomendado**.

Sempre utilize `pd`.

---

# Erro comum

Um erro frequente entre iniciantes é esquecer de executar a célula antes de utilizar o Pandas.

Por exemplo:

```python
df = pd.DataFrame()
```

Sem executar primeiro:

```python
import pandas as pd
```

O Python exibirá um erro semelhante a este:

```text
NameError: name 'pd' is not defined
```

Isso acontece porque a biblioteca ainda não foi carregada.

---

# Curiosidade

O Pandas foi criado em 2008 por **Wes McKinney**, quando trabalhava com análise de dados financeiros.

Seu objetivo era criar uma ferramenta que facilitasse a manipulação de grandes volumes de dados utilizando Python.

Hoje, o Pandas é uma das bibliotecas mais utilizadas em Ciência de Dados no mundo.

---

# No mercado

É difícil encontrar uma vaga para Cientista de Dados, Analista de Dados ou Engenheiro de Machine Learning que não exija conhecimento em Pandas.

Dominar essa biblioteca é um dos primeiros passos para atuar na área.

---

# Resumo

Neste módulo você aprendeu que:

- Bibliotecas permitem reutilizar código.
- O Pandas é a principal biblioteca para manipulação de dados em Python.
- O DataFrame é sua estrutura de dados mais importante.
- `import` carrega uma biblioteca.
- `as` define um apelido.
- `pd` é a convenção utilizada pela comunidade.

---

# Mini desafio

Responda às perguntas.

1. O que é uma biblioteca em Python?

2. Qual é a função do Pandas?

3. O que significa a palavra `import`?

4. Por que utilizamos `as pd`?

5. O que aconteceria se tentássemos utilizar `pd.DataFrame()` antes de importar o Pandas?

---

# Preparando o próximo módulo

Agora que o Pandas já está carregado, construiremos nossa primeira tabela.

Você aprenderá:

- o que é um dicionário;
- como criar um DataFrame;
- como visualizar uma tabela no Google Colab.

Será o primeiro contato com dados utilizando Python.

---

© @karizeviecelli - 2026

<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 08 – Criando seu Primeiro DataFrame

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 08 – Criando seu Primeiro DataFrame

---

> **Nível:** 🟢 Iniciante

---

# Onde estamos?

```text
CICLO DE VIDA DE UM PROJETO DE MACHINE LEARNING

Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Até agora aprendemos:

✔ O que é Machine Learning.

✔ O que é um Dataset.

✔ O que é Pandas.

Agora construiremos nossa primeira tabela utilizando Python.

---

# Antes de escrever código...

Imagine uma planilha do Excel.

| Nome | Idade | Curso |
|------|-------:|--------|
| Ana | 18 | Python |
| João | 20 | Java |
| Maria | 19 | Front-end |

Essa tabela precisa existir dentro do Python.

Mas...

Como fazemos isso?

---

# O DataFrame nasce de uma estrutura de dados

O Pandas consegue criar tabelas utilizando diversas estruturas.

A mais utilizada por iniciantes é o **dicionário**.

Antes de conhecer o DataFrame.

Vamos entender rapidamente o que é um dicionário.

---

# O que é um dicionário?

Um dicionário é uma estrutura formada por:

```
CHAVE

↓

VALOR
```

Exemplo.

```python
aluno = {

    "nome": "Ana",

    "idade": 18

}
```

Observe.

```
nome

↓

Ana
```

```
idade

↓

18
```

Cada informação possui uma chave.

---

# Analogia

Imagine um armário.

Cada gaveta possui uma etiqueta.

```
Nome

↓

Ana
```

```
Curso

↓

Python
```

```
Idade

↓

18
```

As etiquetas são as chaves.

O conteúdo das gavetas são os valores.

---

# Criando nosso primeiro dicionário

Agora vamos criar um dicionário contendo vários alunos.

```python
dados = {

    "Nome": ["Ana", "João", "Maria"],

    "Idade": [18, 20, 19],

    "Curso": ["Python", "Java", "Front-end"]

}
```

---

# Entendendo linha por linha

Observe a primeira linha.

```python
dados = {
```

Estamos criando uma variável chamada **dados**.

Ela armazenará todas as informações.

---

Agora observe.

```python
"Nome":
```

Esta é a primeira chave.

Ela representa uma coluna da futura tabela.

---

Depois temos.

```python
["Ana", "João", "Maria"]
```

Essa é uma lista.

Cada elemento representa um registro.

Visualmente.

| Nome |
|------|
| Ana |
| João |
| Maria |

---

Agora observe.

```python
"Idade": [18,20,19]
```

Essa será outra coluna.

| Idade |
|-------:|
| 18 |
| 20 |
| 19 |

---

Perceba algo importante.

Todas as listas possuem exatamente a mesma quantidade de elementos.

```
Nome

↓

3 valores

Idade

↓

3 valores

Curso

↓

3 valores
```

Isso é obrigatório.

Caso uma coluna possua mais registros que outra.

O DataFrame não poderá ser criado.

---

# Criando o DataFrame

Agora utilizaremos o Pandas.

```python
import pandas as pd

dados = {

    "Nome": ["Ana", "João", "Maria"],

    "Idade": [18,20,19],

    "Curso": ["Python", "Java", "Front-end"]

}

df = pd.DataFrame(dados)

df
```

---

# O que aconteceu?

Observe a última linha.

```python
df
```

O Google Colab exibe automaticamente o DataFrame.

O resultado será semelhante a isto.

| | Nome | Idade | Curso |
|--|------|-------:|--------|
|0|Ana|18|Python|
|1|João|20|Java|
|2|Maria|19|Front-end|

---

# Linha por linha

## Importação

```python
import pandas as pd
```

Carrega a biblioteca.

---

## Dicionário

```python
dados
```

Armazena as informações.

---

## DataFrame

```python
pd.DataFrame()
```

Transforma o dicionário em uma tabela.

---

## Variável df

```python
df
```

Recebe a tabela criada.

Observe.

```
dados

↓

DataFrame

↓

df
```

---

# Por que utilizamos "df"?

Assim como "pd".

Também existe uma convenção.

Grande parte dos programadores utiliza:

```python
df
```

Significa.

```
DataFrame
```

Você poderá utilizar outro nome.

Mas durante este curso seguiremos o padrão utilizado pela comunidade.

---

# Experimente

Altere.

```python
"Maria"
```

para.

```python
"Carlos"
```

Execute novamente.

Observe que a tabela muda imediatamente.

---

Agora adicione uma nova coluna.

```
Nota
```

Tente fazer sozinho.

---

Depois.

Adicione um quarto aluno.

---

# Erro comum

Observe este código.

```python
dados = {

"Nome":["Ana","João"],

"Idade":[18]

}
```

A coluna Nome possui dois valores.

A coluna Idade possui apenas um.

Ao executar.

O Pandas exibirá um erro.

Isso acontece porque todas as colunas precisam possuir o mesmo número de registros.

---

# Curiosidade

O DataFrame pode ser criado utilizando:

✔ Dicionários

✔ Arquivos CSV

✔ Excel

✔ Banco de Dados

✔ JSON

✔ APIs

Hoje começamos pelo método mais simples.

Nas próximas aulas aprenderemos os demais.

---

# Laboratório 01

🎯 Objetivo

Construir seu primeiro DataFrame.

Tempo estimado.

15 minutos.

---

## Parte 1

Crie um DataFrame contendo.

- Nome

- Idade

- Curso

- Cidade

Com cinco alunos.

---

## Parte 2

Adicione.

```
Nota
```

---

## Parte 3

Adicione.

```
Frequência
```

---

## Parte 4

Exiba o resultado.

---

# Desafio

Uma locadora deseja cadastrar filmes.

Crie um DataFrame contendo.

- Filme

- Ano

- Gênero

- Nota IMDb

- Duração

Cadastre pelo menos cinco filmes.

---

# No Mercado

Grande parte dos Cientistas de Dados cria pequenos DataFrames manualmente para testar algoritmos antes de utilizar grandes bases de dados.

Essa prática ajuda a encontrar erros rapidamente e facilita a validação das primeiras ideias.

---

# Resumo

Neste módulo você aprendeu.

✔ O que é um dicionário.

✔ Como criar um dicionário.

✔ Como transformar um dicionário em um DataFrame.

✔ O significado de `pd.DataFrame()`.

✔ Por que utilizamos a variável `df`.

---

# Preparando o próximo módulo

Até agora criamos uma tabela manualmente.

Mas na vida real quase nunca digitamos os dados.

Normalmente recebemos arquivos já prontos.

No próximo módulo aprenderemos a abrir nosso primeiro arquivo CSV utilizando o Google Colab.

---

© @karizeviecelli - 2026


<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 09 – Trabalhando com Arquivos no Google Colab

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 09 – Trabalhando com Arquivos no Google Colab

---

> **Nível:** 🟢 Iniciante

---

# Onde estamos?

```text
CICLO DE VIDA DE UM PROJETO DE MACHINE LEARNING

Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Agora que sabemos criar um DataFrame manualmente, chegou o momento de trabalhar com dados reais.

Na maioria dos projetos, os dados não são digitados pelo Cientista de Dados.

Eles já chegam prontos em arquivos.

---

# De onde vêm os dados?

Em projetos reais, os dados podem vir de diferentes fontes.

```text
                Dados

                  │

      ┌───────────┼────────────┐

      ▼           ▼            ▼

 Arquivos      Banco       APIs

CSV/Excel      Dados      Sistemas

                  │

                  ▼

             Google Colab

                  │

                  ▼

              Pandas
```

Durante este curso utilizaremos principalmente arquivos CSV.

Mais adiante conheceremos bancos de dados e APIs.

---

# O que é um arquivo CSV?

CSV significa:

**Comma-Separated Values**

Ou, em português:

**Valores Separados por Vírgula**

Apesar do nome, alguns arquivos CSV utilizam ponto e vírgula (`;`) como separador, dependendo da configuração regional.

Um arquivo CSV é um arquivo de texto que organiza dados em formato de tabela.

Veja um exemplo:

```csv
Nome,Idade,Curso,Nota
Ana,18,Python,8.5
João,20,Java,6.8
Maria,19,Front-end,9.2
```

Quando o Pandas lê esse arquivo, ele transforma essas linhas em um DataFrame.

---

# Por que utilizamos CSV?

O CSV é um dos formatos mais utilizados porque:

- é simples;
- ocupa pouco espaço;
- pode ser aberto em diferentes programas;
- é compatível com praticamente todas as linguagens de programação.

---

# Conhecendo o Google Colab

O Google Colab é um ambiente de desenvolvimento executado na nuvem.

Isso significa que o código roda em um computador disponibilizado pelo Google, e não diretamente no seu computador.

Essa característica traz uma consequência importante.

---

# Atenção!

Os arquivos enviados para o Google Colab ficam armazenados apenas durante a sessão.

Quando a sessão é encerrada, eles podem ser removidos.

Por isso, sempre que abrir um notebook novamente, poderá ser necessário fazer um novo upload dos arquivos.

Esse é um comportamento esperado do Colab.

---

# Como enviar um arquivo para o Colab

1. Abra o Google Colab.
2. Clique no ícone de pasta, localizado na barra lateral esquerda.
3. Clique em **Fazer upload**.
4. Selecione o arquivo `alunos.csv`.

Após alguns segundos, o arquivo aparecerá na lista de arquivos disponíveis.

---

# Como verificar os arquivos disponíveis

Antes de tentar abrir um arquivo, é uma boa prática verificar se ele realmente está disponível.

No Google Colab, execute:

```python
import os

os.listdir()
```

## O que esse código faz?

Vamos analisar linha por linha.

### Linha 1

```python
import os
```

Importa a biblioteca `os`, responsável por interagir com o sistema operacional.

### Linha 2

```python
os.listdir()
```

Lista todos os arquivos presentes no diretório atual.

Se o arquivo `alunos.csv` aparecer na lista, significa que ele está disponível para uso.

---

# Lendo um arquivo CSV

Agora sim podemos abrir o arquivo.

```python
import pandas as pd

df = pd.read_csv("alunos.csv")
```

Vamos entender cada parte.

### `pd.read_csv()`

Essa função lê um arquivo CSV e cria automaticamente um DataFrame.

### `"alunos.csv"`

É o nome do arquivo que será aberto.

### `df`

Receberá o DataFrame criado.

---

# Visualizando os dados

Após carregar o arquivo, basta escrever:

```python
df
```

ou

```python
df.head()
```

O Colab exibirá a tabela.

---

# O primeiro erro que quase todo aluno encontra

Imagine executar:

```python
df = pd.read_csv("alunos.csv")
```

sem fazer o upload do arquivo.

O resultado será semelhante a este:

```text
FileNotFoundError:
[Errno 2]
No such file or directory:
'alunos.csv'
```

---

# O que significa esse erro?

Vamos traduzir.

```
File

↓

Arquivo

Not Found

↓

Não encontrado
```

O Python está dizendo:

> "Não consegui localizar o arquivo solicitado."

Isso não significa que o código está errado.

Significa apenas que o arquivo não foi encontrado.

---

# Como resolver?

Sempre siga esta sequência.

✔ Verifique se o arquivo foi enviado.

✔ Confirme o nome do arquivo.

✔ Utilize `os.listdir()` para verificar os arquivos disponíveis.

---

# Outro erro comum

Observe este código.

```python
df = pd.read_csv("Aluno.csv")
```

Mas o arquivo enviado chama-se:

```
alunos.csv
```

Lembre-se.

O nome precisa estar correto.

Mesmo pequenas diferenças podem impedir que o arquivo seja localizado.

---

# Laboratório 02

## Objetivo

Carregar um arquivo CSV utilizando o Google Colab.

---

### Etapa 1

Faça upload do arquivo `alunos.csv`.

---

### Etapa 2

Execute:

```python
import os

os.listdir()
```

Confirme se o arquivo aparece na lista.

---

### Etapa 3

Importe o Pandas.

---

### Etapa 4

Leia o arquivo utilizando:

```python
pd.read_csv()
```

---

### Etapa 5

Exiba a tabela.

---

# Curiosidade

Em projetos profissionais, é comum trabalhar com arquivos que possuem milhões de linhas.

Mesmo nesses casos, o processo de leitura continua sendo praticamente o mesmo.

A diferença está no tamanho do arquivo.

---

# No mercado

Grande parte dos projetos de Ciência de Dados começa com a leitura de arquivos.

Dominar esse processo é essencial antes de aprender algoritmos mais avançados.

---

# Resumo

Neste módulo você aprendeu:

- o que é um arquivo CSV;
- por que ele é tão utilizado;
- como fazer upload de arquivos no Google Colab;
- como verificar os arquivos disponíveis;
- como utilizar `pd.read_csv()`;
- como interpretar o erro `FileNotFoundError`.

---

# Preparando o próximo módulo

Agora que conseguimos carregar um arquivo, precisamos responder uma pergunta.

> O que existe dentro desse DataFrame?

No próximo módulo aprenderemos a explorar os dados utilizando comandos fundamentais do Pandas, como:

- `head()`
- `tail()`
- `shape`
- `columns`
- `dtypes`
- `info()`
- `describe()`

Esses comandos serão utilizados praticamente em todos os projetos de Ciência de Dados.

---

© @karizeviecelli - 2026

<!--
=========================================================
Curso: Machine Learning com Python
Aula 01 – Introdução ao Machine Learning
Módulo 10 – Investigando um Dataset com Pandas

© @karizeviecelli - 2026
=========================================================
-->

# Módulo 10 – Investigando um Dataset com Pandas

---

> **Nível:** 🟢 Iniciante

---

# Onde estamos?

```text
CICLO DE VIDA DE UM PROJETO DE MACHINE LEARNING

Problema de Negócio
        │
        ▼
Coleta de Dados
        │
        ▼
Exploração dos Dados ◄──────── VOCÊ ESTÁ AQUI
        │
        ▼
Limpeza e Preparação
        │
        ▼
Seleção de Features
        │
        ▼
Treinamento
        │
        ▼
Avaliação
        │
        ▼
Uso do Modelo
```

Parabéns!

Você conseguiu carregar seu primeiro dataset.

Agora chegou o momento de fazer aquilo que um Cientista de Dados faz logo nos primeiros minutos de qualquer projeto.

Investigar os dados.

---

# Antes de escrever qualquer algoritmo...

Imagine que uma empresa lhe entregou um arquivo chamado:

```
clientes.csv
```

Você sabe apenas isso.

Será que existem:

- 50 clientes?
- 500 clientes?
- 5 milhões de clientes?

Você não sabe.

Então...

Qual deve ser sua primeira atitude?

Começar um algoritmo?

Não.

Primeiro precisamos conhecer os dados.

---

# O Cientista de Dados é um detetive

Imagine um investigador chegando ao local de um crime.

Ele não sai prendendo pessoas.

Primeiro ele observa.

Depois faz perguntas.

Somente então toma decisões.

Com os dados acontece exatamente a mesma coisa.

```
Receber os dados

↓

Observar

↓

Fazer perguntas

↓

Encontrar respostas

↓

Tomar decisões
```

---

# Pergunta 1

## O arquivo foi carregado corretamente?

A forma mais rápida de responder essa pergunta é:

```python
df.head()
```

---

# O que significa head?

A palavra inglesa:

```
Head

↓

Cabeça
```

No Pandas.

Ela representa o início da tabela.

---

# Executando

```python
df.head()
```

Resultado.

| ID | Nome | Curso | Nota |
|----|-------|--------|------|
|1|Ana|Python|8.5|
|2|João|Java|6.2|
|3|Maria|Python|9.3|
|4|Pedro|Front-end|7.4|
|5|Lucas|Banco de Dados|8.0|

---

# Por que apenas cinco linhas?

Cinco linhas normalmente são suficientes para verificar se:

- o arquivo abriu corretamente;
- os nomes das colunas fazem sentido;
- existem caracteres estranhos;
- os dados parecem consistentes.

---

# Podemos alterar?

Sim.

```python
df.head(10)
```

Agora veremos:

As dez primeiras linhas.

---

# Experimente

Execute:

```python
df.head(15)
```

Depois.

```python
df.head(2)
```

Observe como o resultado muda.

---

# Pergunta 2

## Como termina a tabela?

Imagine um arquivo gigantesco.

Talvez os problemas estejam apenas nas últimas linhas.

Para isso utilizamos.

```python
df.tail()
```

---

# Tail

Em inglês.

```
Tail

↓

Cauda
```

O Pandas exibirá:

As cinco últimas linhas.

---

# Alterando

```python
df.tail(8)
```

Mostra.

As oito últimas linhas.

---

# Quando utilizar?

Sempre que desejar verificar:

- importações incompletas;

- registros finais;

- linhas adicionadas recentemente.

---

# Pergunta 3

## Quantas linhas existem?

Agora queremos descobrir o tamanho do dataset.

Utilizamos.

```python
df.shape
```

Resultado.

```python
(500,6)
```

---

# O que isso significa?

Primeiro número.

```
500

↓

Linhas
```

Segundo número.

```
6

↓

Colunas
```

---

# Dica

Sempre leia assim.

```
(shape)

↓

(Linhas, Colunas)
```

Nunca o contrário.

Esse é um erro muito comum.

---

# Pergunta 4

## Quais colunas existem?

Agora queremos conhecer os nomes das colunas.

```python
df.columns
```

Resultado.

```python
Index([
'ID',

'Nome',

'Curso',

'Nota',

'Frequencia',

'Aprovado'
])
```

---

# Por que isso é importante?

Imagine um dataset com 250 colunas.

É praticamente impossível memorizar todos os nomes.

O comando:

```python
df.columns
```

resolve esse problema rapidamente.

---

# Pergunta 5

## Que tipo de informação existe em cada coluna?

Nem todas as colunas armazenam o mesmo tipo de dado.

Veja.

```python
df.dtypes
```

Resultado.

```text
Nome          object

Idade          int64

Nota         float64

Aprovado      object
```

---

# O que significa?

| Tipo | Significado |
|-------|-------------|
| object | Texto |
| int64 | Número inteiro |
| float64 | Número decimal |
| bool | Verdadeiro ou falso |

Mais adiante conheceremos outros tipos.

---

# Pergunta 6

## Existe algum problema no dataset?

Agora utilizaremos um dos comandos mais importantes do Pandas.

```python
df.info()
```

---

# O que ele mostra?

- quantidade de linhas;

- quantidade de colunas;

- tipos de dados;

- valores não nulos;

- memória utilizada.

É praticamente um "raio X" do DataFrame.

---

# Pergunta 7

## Como os números estão distribuídos?

Agora queremos conhecer os valores numéricos.

```python
df.describe()
```

---

# O que aparece?

O Pandas calcula automaticamente.

- média;

- desvio padrão;

- menor valor;

- maior valor;

- quartis.

Não se preocupe.

Estudaremos cada um desses conceitos no momento adequado.

Hoje queremos apenas aprender a interpretar o resultado.

---

# Resumo dos comandos

| Pergunta | Comando |
|-----------|----------|
| O arquivo abriu corretamente? | `head()` |
| Como termina a tabela? | `tail()` |
| Qual o tamanho? | `shape` |
| Quais colunas existem? | `columns` |
| Quais tipos de dados? | `dtypes` |
| Existe algum problema? | `info()` |
| Como estão distribuídos os números? | `describe()` |

---

# Laboratório 03

Imagine que você acaba de ser contratado por uma empresa.

Recebeu um dataset.

Sua missão é responder às seguintes perguntas.

1. Quantas linhas existem?

2. Quantas colunas?

3. Quais são os nomes das colunas?

4. Quais colunas possuem números?

5. Quais colunas possuem texto?

6. Existe alguma informação ausente?

7. Qual é a média da coluna Nota?

---

# Desafio

Utilizando apenas os comandos aprendidos hoje.

Escreva um pequeno relatório contendo.

- quantidade de registros;

- quantidade de colunas;

- nomes das colunas;

- média da nota;

- menor nota;

- maior nota.

---

# Erro comum

Muitos alunos executam:

```python
df.describe()
```

E tentam decorar todos os números.

Não faça isso.

O objetivo inicial é aprender:

> **Qual pergunta cada comando responde.**

Com o tempo você aprenderá a interpretar cada estatística.

---

# Curiosidade

Na maioria dos projetos profissionais, esses são exatamente os primeiros comandos executados após a leitura de um dataset.

Por isso vale a pena dominá-los muito bem.

---

# No mercado

É comum ver Cientistas de Dados executando:

```python
df.head()

df.info()

df.describe()
```

Nos primeiros minutos de qualquer análise.

Esses comandos ajudam a entender rapidamente a qualidade dos dados antes de qualquer tratamento.

---

# O que aprendemos até aqui?

Observe tudo o que já somos capazes de fazer.

✔ Entender o problema.

✔ Conhecer o dataset.

✔ Criar DataFrames.

✔ Abrir arquivos CSV.

✔ Explorar dados.

Você acaba de concluir a primeira etapa de praticamente todo projeto de Ciência de Dados.

Na próxima aula aprenderemos como identificar e corrigir problemas nos dados antes de treinar nosso primeiro modelo de Machine Learning.

---

© @karizeviecelli - 2026

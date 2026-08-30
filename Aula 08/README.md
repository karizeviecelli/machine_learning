# AULA 08 --- Introdução à IA Generativa e Prompt Engineering

**Trilha de IA Aplicada --- Base Comum dos Projetos**\
**Carga horária sugerida:** 4 horas\
**Pré-requisitos:** noções de Machine Learning, classificação e NLP
básico.


**[Click aqui - Link Colab](https://colab.research.google.com/drive/1leZI7ShHE4jOMDzievMRley5l27SXEey?usp=sharing)**
------------------------------------------------------------------------

## 1. Onde estamos?

Já vimos modelos que recebem dados e produzem previsões.

Exemplos:

``` text
Features
   ↓
Decision Tree
   ↓
classe
```

``` text
Features
   ↓
KNN
   ↓
classe
```

``` text
Texto
   ↓
CountVectorizer
   ↓
classificador
   ↓
categoria
```

Agora vamos conhecer uma abordagem diferente:

> **IA Generativa**

------------------------------------------------------------------------

# 2. O que significa "generativa"?

Até aqui perguntávamos coisas como:

``` text
Qual é a classe?
```

Agora podemos pedir:

``` text
Gere um resumo.
```

``` text
Organize estas informações.
```

``` text
Transforme estes dados em uma descrição.
```

``` text
Crie uma resposta utilizando este contexto.
```

A IA passa a **gerar conteúdo**.

------------------------------------------------------------------------

# 3. IA tradicional × IA generativa

Uma comparação introdutória:

  -----------------------------------------------------------------------
  Machine Learning de classificação   IA Generativa
  ----------------------------------- -----------------------------------
  prevê uma classe                    gera conteúdo

  SIM/NÃO                             texto completo

  categoria A/B/C                     resumo

  prioridade alta/média/baixa         explicação organizada

  usa features para prever target     usa instrução e contexto para
                                      produzir saída
  -----------------------------------------------------------------------

As áreas podem trabalhar juntas.

Não precisamos escolher apenas uma.

------------------------------------------------------------------------

# 4. Exemplos nos projetos

## Cuidadores

Entrada:

``` text
Pressão: 13/8
Medicamentos: administrados
Observação: pouco apetite
```

Saída desejada:

``` text
Resumo organizado do período
```

## NR-1

Entrada:

``` text
Indicadores agregados por setor
```

Saída:

``` text
Resumo organizacional dos principais indicadores
```

Sem diagnóstico individual e sem expor funcionários.

## Fiscalização

Entrada:

``` text
Tipo: móveis
Local: Bairro Centro
Volume: médio
Prioridade: alta
```

Saída:

``` text
Descrição estruturada da ocorrência
```

## Agenda

Entrada:

``` text
09h reunião
15h dentista
19h academia
```

Saída:

``` text
Resumo organizado do dia
```

------------------------------------------------------------------------

# 5. O que é um LLM?

LLM significa:

> **Large Language Model**

Em português:

> **Modelo de Linguagem de Grande Escala**

Para nossa introdução, pense nele como um modelo treinado com grandes
quantidades de texto para trabalhar com padrões da linguagem.

LLMs podem:

-   responder perguntas;
-   resumir;
-   reescrever;
-   classificar;
-   extrair informações;
-   gerar texto;
-   ajudar com código.

Mas possuem limitações importantes.

------------------------------------------------------------------------

# 6. LLM não é banco de dados

Essa ideia precisa ficar muito clara.

Se perguntarmos:

> "Qual foi a última pressão registrada pelo paciente?"

o modelo não deve inventar.

A informação correta precisa vir do sistema.

Fluxo adequado:

``` text
Banco de Dados
      ↓
dados reais
      ↓
LLM
      ↓
resposta organizada
```

Não:

``` text
LLM
 ↓
adivinha
 ↓
resposta
```

------------------------------------------------------------------------

# 7. O que é um prompt?

Prompt é a instrução/contexto enviado ao modelo.

Exemplo:

``` text
Faça um resumo.
```

Isso é um prompt.

Mas é muito vago.

------------------------------------------------------------------------

# 8. Prompt fraco

``` text
Analise os dados.
```

Problemas:

-   analisar o quê?
-   com qual objetivo?
-   qual formato?
-   pode inferir?
-   pode recomendar?
-   qual público receberá a resposta?

Quanto menos clara a instrução, menos controlada tende a ser a saída.

------------------------------------------------------------------------

# 9. Estrutura básica de um bom prompt

Nesta aula usaremos cinco partes:

``` text
PAPEL
+
TAREFA
+
CONTEXTO/DADOS
+
REGRAS
+
FORMATO
```

Não é uma fórmula obrigatória.

É um guia para escrever instruções melhores.

------------------------------------------------------------------------

# 10. PAPEL

Diz qual função o modelo deve assumir dentro da tarefa.

Exemplo:

``` text
Você é um assistente responsável por organizar
registros de uma agenda.
```

Evite atribuir papéis inadequados.

No projeto Cuidadores, por exemplo, não devemos dizer:

``` text
Você é um médico que diagnostica...
```

se essa não é a finalidade e nem a competência do sistema.

------------------------------------------------------------------------

# 11. TAREFA

Diz claramente o que deve ser feito.

Exemplo:

``` text
Gere um resumo dos compromissos do dia.
```

Ou:

``` text
Organize os dados da ocorrência em um parágrafo curto.
```

------------------------------------------------------------------------

# 12. CONTEXTO / DADOS

Fornece as informações que devem sustentar a resposta.

Exemplo:

``` text
Dados:
09:00 - reunião
15:00 - dentista
19:00 - academia
```

Quanto mais a tarefa depende de dados específicos, mais importante é
fornecer contexto correto.

------------------------------------------------------------------------

# 13. REGRAS

Definem limites.

Exemplo:

``` text
Use somente os dados fornecidos.

Não invente horários.

Não crie compromissos inexistentes.
```

Para Cuidadores:

``` text
Não faça diagnóstico.

Não recomende alteração de medicamentos.
```

Para NR-1:

``` text
Não identifique funcionários.

Não faça diagnóstico de saúde mental.

Analise somente os indicadores agregados fornecidos.
```

------------------------------------------------------------------------

# 14. FORMATO

Podemos dizer como queremos receber a resposta.

Exemplo:

``` text
Formato:
- título;
- três tópicos;
- uma conclusão curta.
```

Ou:

``` text
Responda em um único parágrafo de até 80 palavras.
```

------------------------------------------------------------------------

# 15. Montando um prompt completo

``` text
PAPEL:
Você é um assistente de organização de agenda.

TAREFA:
Gere um resumo dos compromissos do dia.

DADOS:
09:00 - reunião de projeto
15:00 - dentista
19:00 - academia

REGRAS:
Use somente os dados fornecidos.
Não invente horários ou compromissos.

FORMATO:
Use uma saudação curta e uma lista cronológica.
```

Agora temos uma instrução muito mais clara.

------------------------------------------------------------------------

# 16. Prompt Engineering

Chamamos de **Prompt Engineering**, de forma introdutória, o processo de
projetar e melhorar instruções para obter respostas mais úteis,
consistentes e adequadas.

Não é simplesmente:

> "escrever uma pergunta bonita".

É pensar em:

-   objetivo;
-   contexto;
-   restrições;
-   formato;
-   exemplos;
-   validação.

------------------------------------------------------------------------

# 17. Iteração

Um prompt raramente precisa nascer perfeito.

Podemos trabalhar assim:

``` text
PROMPT 1
   ↓
testar
   ↓
encontrar problema
   ↓
PROMPT 2
   ↓
testar
   ↓
melhorar
```

Isso é ótimo para os projetos.

Os alunos devem documentar **por que alteraram o prompt**.

------------------------------------------------------------------------

# 18. O que é alucinação?

Uma alucinação acontece quando o modelo gera conteúdo não sustentado
pelo contexto ou apresenta informação incorreta como se fosse válida.

Dados:

``` text
Temperatura: 36,5
Observação: pouco apetite
```

Resposta problemática:

``` text
O paciente está com infecção.
```

Essa conclusão não foi fornecida pelos dados e constitui uma inferência
médica indevida.

------------------------------------------------------------------------

# 19. Outro exemplo de alucinação

Dados da agenda:

``` text
15h → dentista
```

Resposta:

``` text
Seu dentista será na Clínica Central.
```

De onde veio:

``` text
Clínica Central?
```

Não estava no contexto.

O modelo inventou.

------------------------------------------------------------------------

# 20. Como reduzir alucinações?

Não existe uma frase mágica que elimine todos os erros.

Mas podemos reduzir riscos:

-   fornecer contexto;
-   usar dados corretos;
-   limitar a tarefa;
-   pedir uso exclusivo dos dados fornecidos;
-   exigir indicação quando não houver informação;
-   utilizar saída estruturada;
-   validar respostas;
-   manter revisão humana em situações relevantes.

------------------------------------------------------------------------

# 21. Uma regra muito útil

Podemos adicionar:

``` text
Se a informação não estiver nos dados fornecidos,
responda que não há informação suficiente.
```

Exemplo:

Usuário:

``` text
Qual o local da reunião?
```

Dados:

``` text
Reunião às 14h.
```

Resposta desejada:

``` text
O local não foi informado.
```

Muito melhor do que inventar.

------------------------------------------------------------------------

# 22. Temperatura do modelo?

Algumas APIs oferecem parâmetros que influenciam a variabilidade da
geração, frequentemente chamados de `temperature`.

Para esta aula basta compreender:

``` text
menor variabilidade
        ↕
maior variabilidade
```

Não precisamos aprofundar configuração de APIs agora.

Nosso foco é:

> **contexto + instrução + validação.**

------------------------------------------------------------------------

# 23. Dados estruturados

Uma aplicação pode enviar dados em formato organizado.

Exemplo:

``` json
{
  "tipo": "entulho",
  "volume": "medio",
  "bairro": "Centro",
  "prioridade": "alta"
}
```

Prompt:

``` text
Utilize os dados fornecidos para gerar uma descrição
objetiva da ocorrência.

Não acrescente informações que não estejam no objeto.
```

Isso facilita controlar a tarefa.

------------------------------------------------------------------------

# 24. Entrada estruturada × saída estruturada

Também podemos pedir uma resposta estruturada.

Exemplo:

``` text
Retorne:

Resumo:
Prioridade:
Informações ausentes:
```

Ou, em integrações futuras, formatos como JSON podem ser úteis.

Nesta aula basta compreender o conceito.

------------------------------------------------------------------------

# 25. IA generativa não precisa decidir tudo

Um erro comum é entregar toda a lógica do sistema ao LLM.

Exemplo da agenda:

Detectar conflito entre:

``` text
14:00–15:00
```

e:

``` text
14:30–16:00
```

pode ser resolvido de forma determinística pelo próprio sistema.

Não precisamos usar IA para tudo.

Pergunta importante:

> **Esta tarefa realmente precisa de IA?**

------------------------------------------------------------------------

# 26. Regra × ML × IA Generativa

## Regra

``` text
SE horário novo conflita
ENTÃO bloquear
```

## Machine Learning

``` text
features
 ↓
modelo
 ↓
prioridade prevista
```

## IA Generativa

``` text
dados
 ↓
LLM
 ↓
resumo textual
```

Cada ferramenta resolve um tipo de problema.

------------------------------------------------------------------------

# 27. Aplicação: Cuidadores

Uso básico adequado:

``` text
registros estruturados
        ↓
LLM
        ↓
rascunho de resumo
        ↓
revisão humana
```

Restrições:

``` text
não diagnosticar
não alterar medicamentos
não inventar registros
```

------------------------------------------------------------------------

# 28. Aplicação: NR-1

Uso básico adequado:

``` text
indicadores agregados
        ↓
LLM
        ↓
resumo organizacional
```

Restrições:

``` text
não identificar pessoas
não diagnosticar funcionários
não inferir condição clínica
usar somente dados agregados fornecidos
```

------------------------------------------------------------------------

# 29. Aplicação: Fiscalização

Uso básico:

``` text
dados da ocorrência
        ↓
LLM
        ↓
descrição padronizada
```

Exemplo:

``` text
Categoria: móveis
Volume: médio
Localização: informada
Prioridade: alta
```

A classificação visual por imagem pode ser adicionada posteriormente
como especialização.

------------------------------------------------------------------------

# 30. Aplicação: Agenda

Uso básico:

``` text
compromissos
     ↓
LLM
     ↓
resumo diário
```

Exemplo:

``` text
09h reunião
15h dentista
19h academia
```

O LLM organiza o texto.

A lógica de datas, conflitos e persistência continua sendo
responsabilidade da aplicação.

------------------------------------------------------------------------

# 31. Atividade de fixação 1 --- IA ou não?

Para cada caso, indique uma solução possível:

``` text
REGRA
ML
IA GENERATIVA
```

1.  Detectar se dois horários se sobrepõem.
2.  Classificar uma tarefa como alta/média/baixa prioridade usando dados
    históricos.
3.  Gerar um resumo dos compromissos.
4.  Classificar comentário em tema.
5.  Gerar descrição textual de uma ocorrência já estruturada.
6.  Verificar se um campo obrigatório está vazio.

------------------------------------------------------------------------

# 32. Atividade de fixação 2 --- Melhorando prompts

Prompt inicial:

``` text
Faça um resumo.
```

Reescreva incluindo:

-   papel;
-   tarefa;
-   dados;
-   regras;
-   formato.

------------------------------------------------------------------------

# 33. Atividade de fixação 3 --- Encontre o problema

Dados:

``` text
Setor: Produção
Indicador agregado de sobrecarga: elevado
Indicador de apoio: atenção
```

Resposta:

``` text
João provavelmente está sofrendo de ansiedade devido ao excesso de trabalho.
```

Perguntas:

1.  Qual informação foi inventada?
2.  Houve identificação indevida de pessoa?
3.  Houve diagnóstico/inferência clínica?
4.  Como a resposta poderia ser reescrita usando apenas os dados
    agregados?

------------------------------------------------------------------------

# 34. Atividade de fixação 4 --- Agenda

Dados:

``` text
09h reunião
15h dentista
```

Resposta da IA:

``` text
Você tem reunião às 9h no escritório e dentista
às 15h na Clínica Sorriso.
```

Pergunta:

> Quais informações não estavam nos dados?

Resposta:

``` text
"no escritório"
"Clínica Sorriso"
```

------------------------------------------------------------------------

# 35. Prática principal --- Três versões de prompt

Cada grupo escolherá uma funcionalidade do seu projeto.

Deverá criar:

``` text
PROMPT V1
```

Testar.

Registrar o problema.

Depois:

``` text
PROMPT V2
```

Melhorar.

Testar novamente.

Finalmente:

``` text
PROMPT V3
```

Versão final.

------------------------------------------------------------------------

# 36. Modelo de documentação

## Prompt V1

``` text
...
```

### Resultado

``` text
...
```

### Problemas encontrados

``` text
...
```

## Prompt V2

``` text
...
```

### O que melhoramos?

``` text
...
```

## Prompt V3

``` text
...
```

### Por que esta versão é melhor?

``` text
...
```

------------------------------------------------------------------------

# 37. Teste de alucinação

Crie uma informação propositalmente ausente.

Exemplo:

``` text
Compromisso: dentista
Horário: 15h
Local: NÃO INFORMADO
```

Pergunte:

``` text
Onde será o dentista?
```

O prompt deve orientar o modelo a responder:

``` text
O local não foi informado.
```

Registre se isso aconteceu.

------------------------------------------------------------------------

# 38. Validação humana

Em aplicações reais, especialmente quando a saída pode afetar pessoas,
devemos pensar em revisão.

Exemplo:

``` text
DADOS
  ↓
LLM
  ↓
RASCUNHO
  ↓
USUÁRIO REVISA
  ↓
CONFIRMA
  ↓
SALVA
```

Essa arquitetura é mais responsável do que confiar automaticamente em
toda saída.

------------------------------------------------------------------------

# 39. Privacidade

Os projetos lidam com diferentes tipos de informação.

Antes de enviar dados a qualquer serviço externo, a equipe deve
perguntar:

-   precisamos realmente enviar esse dado?
-   há dado pessoal?
-   há dado sensível?
-   podemos anonimizar?
-   quem pode visualizar?
-   por quanto tempo será armazenado?
-   o usuário sabe como seus dados serão utilizados?

IA também é uma discussão de **responsabilidade**.

------------------------------------------------------------------------

# 40. Erros comuns

## Erro 1

Achar que um prompt garante 100% de precisão.

## Erro 2

Permitir que o modelo invente dados ausentes.

## Erro 3

Usar IA para uma regra simples.

## Erro 4

Confundir LLM com banco de dados.

## Erro 5

Enviar dados desnecessários ao modelo.

## Erro 6

Não validar a saída.

## Erro 7

Chamar qualquer uso de API de "modelo treinado pela equipe".

Usar um modelo pronto e treinar um modelo próprio são coisas diferentes.

------------------------------------------------------------------------

# 41. Questões de revisão

1.  O que é IA Generativa?
2.  Qual a diferença básica entre classificação e geração?
3.  O que significa LLM?
4.  O que é um prompt?
5.  O que é Prompt Engineering?
6.  Quais cinco partes usamos como guia para prompts?
7.  Por que fornecer contexto?
8.  O que são regras/restrições?
9.  O que é uma alucinação?
10. Um LLM é um banco de dados?
11. O que o modelo deve responder quando uma informação necessária não
    está no contexto?
12. Por que validar a saída?
13. Toda funcionalidade precisa de IA?
14. Quando uma regra tradicional pode ser melhor?
15. Qual a diferença entre usar um modelo pronto e treinar seu próprio
    modelo?
16. Por que privacidade precisa fazer parte do projeto?

------------------------------------------------------------------------

# 42. Desafio final

Cada grupo deverá demonstrar uma funcionalidade de IA generativa do seu
projeto.

A apresentação deve mostrar:

``` text
1. Problema
2. Dados de entrada
3. Prompt
4. Resposta
5. Limitações
6. Regra contra invenção
7. Como haverá validação
```

A banca/professor deverá fazer uma pergunta cuja resposta **não esteja
nos dados**.

O sistema/prompt deverá evitar inventar.

------------------------------------------------------------------------

# 43. Entrega

Arquivo:

``` text
04_ia_generativa_prompts.md
```

Deve conter:

-   descrição da funcionalidade;
-   Prompt V1;
-   resultado V1;
-   problema encontrado;
-   Prompt V2;
-   melhoria;
-   Prompt V3;
-   teste de informação ausente;
-   análise de alucinação;
-   regras de segurança/privacidade;
-   conclusão.

------------------------------------------------------------------------

# 44. Fechamento da trilha básica

A sequência estudada fica:

``` text
DADOS
  ↓
FEATURES / TARGET
  ↓
DECISION TREE
  ↓
ACURÁCIA
  ↓
KNN
  ↓
MATRIZ DE CONFUSÃO
PRECISÃO / RECALL / F1
  ↓
NLP
  ↓
TEXTO → NÚMEROS
  ↓
CLASSIFICAÇÃO DE TEXTO
  ↓
IA GENERATIVA
  ↓
PROMPTS
  ↓
VALIDAÇÃO
```

A partir daqui, cada projeto pode seguir para uma especialização
diferente:

``` text
Visão computacional
Detecção de anomalias
Extração de entidades
Recomendação
Séries temporais
Mapas de calor
Integração com APIs
```

Mas todos compartilham uma **base comum de Inteligência Artificial**.

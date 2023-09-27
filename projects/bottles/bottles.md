# Projeto -- Bottles

## Introdução

O link em anexo contém 77 imagens de garrafas. Cada garrafa (no centro da imagem) pode possuir um ou mais dos seguintes 8 defeitos:

1. CONTENT_HIGH,
1. CONTENT_LOW,
1. COVER_NONE,
1. BOTTLE_SMASHED,
1. LABEL_WHITE,
1. LABEL_MISPLACED,
1. LABEL_NONE,
1. BOTTLE_NONE.

Cada discente deverá submeter 1 arquivo do tipo 'notebook' (<nome do indivíduo>.ipynb) contendo **uma função** que receba uma imagem (array numpy) e entregue um array (numpy) contendo 8 valores. Cada valor deve ser:
- **0** (para **ausência** do defeito correspondente), ou;
- **1** (para **presença** do defeito correspondente).

### Exemplo:
- A imagem ```train_1.jpg``` não possui defeitos, portanto, o resultado da inspeção deve ser um array numpy contendo 8 zeros:
```
array([0., 0., 0., 0., 0., 0., 0., 0.])
```
- Já a imagem ```train_4.jpg``` possui 2 defeitos; ```CONTENT_HIGH``` e ```COVER_NONE```, portanto, o resultado da inspeção deve ser um array (numpy) como o que segue:
```
array([1., 0., 1., 0., 0., 0., 0., 0.])
```

## Requisitos:
1. Utilizar o Opencv 4.5.5.
1. Em cada imagem, considerar somente a garrafa do meio.
1. Trabalho individual.
1. Utilizar somente técnicas associadas à disciplina de processamento de imagens.

Bons estudos.

# Referências
- [dataset](https://github.com/tfvieira/digital-image-processing/tree/main/img/bottles)
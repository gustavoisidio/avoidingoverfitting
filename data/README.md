# Bases e seus geradores
## Não modifique a base, crie uma variação dela

## Protótipo de bases 
ru: com ruído | ra: com aleatórios | fix: Cruz em posição fixa
* nome.csv (tamanho) - Descrição
* 4x4.csv (4) - As 4 possibilidades de posições para cruz 4x4, sem ruído
* 4x4ra.csv (2888) - 4x4.csv com várias não-cruzes aleatórias
* 4x4ru0.csv - Rascunho de ruído adicionado manualmente
* 4x4ru.csv (72) - 4x4.csv com ruído adicionado manualmente
* 4x4rura.csv (144) - 4x4ru.csv com algumas não-cruz aleatórias, 50% de cada classe
* 4x4rurafix.csv (44) - Cruz em uma posição fixa, com ruído e não-cruzes aleatórias, 50% de cada classe
* data1.csv (65536) - Todas as possibilidades classificadas 
* data2.csv (46902) - data1.csv sem os elementos com quantidade de pontos fora do intervalo [5,8]
* data3.csv (46902) - data2.csv com classes separadas
* data4.csv - data1.csv com classes separadas
* data5.csv - data1.csv com cruzes modificadas para cruzes limpas
* data6.csv - data5.csv com classes separadas

## Bases "Organizadas"
* 0noise.csv (4368) - Cruzes puras, sem ruído + aleatorios 
* 0noise-balanced.csv (8) - 0noise.csv balanceada
* 1noise.csv (12376) - Cruzes com no máximo 1 ruído + aleatorios 
* 1noise-balanced.csv (96) - 1noise.csv balanceada
* 2noises.csv (23816) - Cruzes com no máximo 2 ruídos + aleatorios 
* 2noises-balanced.csv (536) - 2noises.csv balanceada
* 3noises.csv (36686) - Cruzes com no máximo 3 ruídos + aleatorios 
* 3noises-balanced.csv () - Precisa corrigir
* 4noises.csv (48126) - Cruzes com no máximo 4 ruídos + aleatorios 
* 4noises-balanced.csv (4280) - 4noises.csv balanceada

## Geradores / Filtros
* gerador1.c - Gerador de todas as possibilidades de cruzes para 4, 8, 16 e 32
* gerador2.hs - Gerador de aleatórios e tratador de base gerada
* gerador3.hs - Gerador de todas as possibilidades 
* gerador4.hs - Gerador de todas as possibilidades classificadas
* gerador5.hs - Separador de classes
* gerador6.hs - Limpador de cruzes/pontos

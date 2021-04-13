# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 08:08:57 2021

@author: Vieira
"""

#%%
"""
Execute as seguintes tarefas (salve todos os resultados):
(a) Gere uma imagem 640x480 contendo níveis de cinza aleatórios com distribuição de probabilidade normal com média 128 e desvio padrão 20. Utilize semente aleatória padrão igual a 0 (zero).
(b) Leia a imagem "septagon.tif", crie e adicione ruído do tipo sal e pimenta a esta imagem. Salve o resultado.
(c) Carregue a imagem "hsv_disk.png" e a converta para escala de cinza. Salve o resultado.
(d) Carregue a imagem "septagon.tif" e aplique um limiar (threshold) a esta imagem buscando segmentar o objeto do background. Salve o resultado.
"""


#%%
"""
Carregue a imagem "pollen.tif", mostre o seu histograma, aplique uma equalização e apresente a imagem equalizada e o novo histograma. Salve o resultado da imagem com contraste aprimorado.
"""

#%%
"""
Filtragem no domínio espacial:
(a) Carregue a imagem "ckt.tif" e obtenha as imagens gx e gy, contendo os gradientes nas direções x e y, respectivamente, usando filtragem no domínio espacial. Salve os resultados.
(b) Obtenha uma terceira imagem 'g' que é a aproximação do gradiente dada por g = |gx| + |gy|. Salve o resultado.
(c) Obtenha a imagem binária 'b' que contenha o valor '1' nas coordenadas dos pixels de g que são maiores ou iguais a 60 e zero caso contrário. Salve o resultado.
(d) Em que tipo de aplicação essa abordagem poderia ser útil?
"""

#%%
"""
Filtragem no domínio da frequência:
Carregue a imagem "lena_noise.png" e aplique uma filtragem no domínio da frequência para atenuar todos os ruídos periódicos presentes na imagem. Salve o resultado.
"""

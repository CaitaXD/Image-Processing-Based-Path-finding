Proposta da Atividade: Path-Finding de muitas unidades usando operacoes de processamento de imagens sempre que possivel

Utilizando representacoes de grid para unidades e objetivos (imagens)

Experimento 1: Usando OpenCV

1. Máscara (Input)  cv2.threshold (Isso deve tornar possivel utilizar a mesma imagem para unidades e objectivos)

2. Heat Diffusion   cv2.filter2D + cv2.bitwise_and

3. Vector Field     cv2.Sobel(X) e cv2.Sobel(Y)

4. Wall Masking     cv2.bitwise_and

5. Resampling       cv2.remap 

Experimento 2: Gravar um video de cada modo de visualizacao

Experimento 3: Gravar um video lento mostrando as transformacoes das unidades em cada etapa uma transformacao por frame

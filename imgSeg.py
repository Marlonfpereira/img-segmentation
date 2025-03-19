import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import random

def ler_imagem_cinza(file_path):
    with Image.open(file_path) as img:
        img_cinza = img.convert('L')
        array = np.array(img_cinza, dtype=np.uint8)
        min_val = np.min(array)
        max_val = np.max(array)
        
    return array, min_val, max_val

def criar_gaussiana(sigma, tamanho):
    matriz = np.zeros((tamanho, tamanho))
    soma = 0
    centro = tamanho // 2

    for i in range(tamanho):
        for j in range(tamanho):
            valor = math.pow(math.e, -((i - centro) ** 2 + (j - centro) ** 2) / (2 * sigma ** 2))
            matriz[i][j] = valor
            soma += valor

    for i in range(tamanho):
        for j in range(tamanho):
            matriz[i][j] /= soma

    return matriz

def aplicar_filtro(img, filter):
    k_h, k_w = filter.shape
    pad_h, pad_w = (k_h // 2,k_h // 2), (k_w // 2,k_w // 2 )

    if k_h % 2 == 0:
        pad_h = (k_h // 2, k_h // 2 - 1)  
    if k_w % 2 == 0:
        pad_w = (k_w // 2, k_w // 2 - 1)  

    padded_image = np.pad(img, ((pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode='edge')
    result = np.zeros(img.shape)

    for i in range(pad_h[0], padded_image.shape[0] - pad_h[1]):
        for j in range(pad_w[0], padded_image.shape[1] - pad_w[1]):
            result[i - pad_h[0], j - pad_w[0]] = np.sum(padded_image[i - pad_h[0]:i + pad_h[1] + 1, j - pad_w[0]:j + pad_w[1] + 1] * filter)

    return result

"""
Marr-Hildreth
    Gaussiano > Laplaciano > Cruzamento por zero
    -Sensível a ruído
    -Bordas menos precisas em imagens complexas
    +Computacionalmente mais simples
"""
def cruzamentos_por_zero(matriz, t):
    altura, largura = matriz.shape
    resultado = np.zeros(matriz.shape)

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            vizinhos = [(matriz[i-1][j], matriz[i+1][j]), (matriz[i][j-1], matriz[i][j+1]), (matriz[i-1][j-1], matriz[i+1][j+1]), (matriz[i-1][j+1], matriz[i+1][j-1])]
            if any(abs(v[0] - v[1]) > t for v in vizinhos):
                resultado[i][j] = 255
    return resultado


"""
Canny
    Gaussiano > Gradientes de intensidade (Sobel) > Supressão não máxima > Limiar duplo > Alongamento de bordas
    +Melhor para identificar bordas reais e ignorar ruídos
    +Bordas mais nítidas e definidas
    -Computacionalmente mais complexo
"""
def canny(th1, th2, gauss):
    # kernel, kern_size = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3 # Prewitt
    kernel, kern_size = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), 3 # Sobel
    gx, gy = np.zeros_like(gauss, dtype=float), np.zeros_like(gauss, dtype=float)

    for i in range(gauss.shape[0]-(kern_size-1)):
        for j in range(gauss.shape[1]-(kern_size-1)):
            window = gauss[i:i+kern_size, j:j+kern_size]
            gx[i, j], gy[i, j] = np.sum(window * kernel), np.sum(window * kernel.T)

    magnitude = np.sqrt(gx**2 + gy**2)
    theta = np.arctan(gy/gx)
    nms = np.copy(magnitude)

    theta[theta < 0] += 180

    for i in range(theta.shape[0]-(kern_size-1)):
        for j in range(theta.shape[1]-(kern_size-1)):
            if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                if(magnitude[i, j] <= magnitude[i-1, j]) or (magnitude[i, j] <= magnitude[i+1, j]):
                    nms[i, j] = 0
            if (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                if(magnitude[i, j] <= magnitude[i-1, j-1]) or (magnitude[i, j] <= magnitude[i+1, j+1]):
                    nms[i, j] = 0
            if (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                if(magnitude[i, j] <= magnitude[i, j+1]) or (magnitude[i, j] <= magnitude[i, j-1]):
                    nms[i, j] = 0
            if (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                if(magnitude[i, j] <= magnitude[i+1, j-1]) or (magnitude[i, j] <= magnitude[i-1, j+1]):
                    nms[i, j] = 0

    fraca, forte = np.copy(nms), np.copy(nms)

    fraca[fraca < th1] = 0
    fraca[fraca >= th1] = 1

    forte[forte < th2] = 0
    forte[forte >= th2] = 1

    for i in range(1, forte.shape[0] - 1):
        for j in range(1, forte.shape[1] - 1):
            if forte[i, j] == 1: 
                forte[i-1:i+2, j-1:j+2][fraca[i-1:i+2, j-1:j+2] > 0] = 1

    return forte

def otsu_threshold(total_pixels, histograma):
    max_variancia = 0
    T = 0
    sum_total = np.sum(np.arange(len(histograma)) * histograma)
    sum_b, w_b, w_f = 0, 0, 0

    for i in range(len(histograma)):
        w_b += histograma[i]

        if w_b == 0:
            continue

        w_f = total_pixels - w_b
        if w_f == 0:
            break

        sum_b += i * histograma[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        a_var = w_b * w_f * (m_b - m_f) ** 2

        if a_var > max_variancia:
            max_variancia = a_var
            T = i

    return T

def corte_threshold(img, threshold):
    corte = np.copy(img)
    corte[corte < threshold] = 0
    corte[corte >= threshold] = 255
    return corte

def corte_threshold_b(img, threshold):
    corte = np.copy(img)
    corte[corte < threshold] = 0
    corte[corte >= threshold] = 1
    return corte

def contagem_objetos(img):
    kernel = np.zeros((5,5), np.uint8)

    k_h, k_w = kernel.shape
    pad_h, pad_w = (k_h // 2,k_h // 2), (k_w // 2,k_w // 2 )

    if k_h % 2 == 0:
        pad_h = (k_h // 2, k_h // 2 - 1)  
    if k_w % 2 == 0:
        pad_w = (k_w // 2, k_w // 2 - 1)  

    aux = np.pad(img, ((pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode='constant', constant_values=1)

    contagem = 0
    for i in range(pad_h[0], aux.shape[0] - pad_h[1]):
        for j in range(pad_w[0], aux.shape[1] - pad_w[1]):
            if aux[i, j] == 0:
                contagem += 1
                tom = random.randint(1, 100) / 100
                aux[i, j] = tom
                lista = [(i, j)]
                while lista:
                    x, y = lista.pop()

                    region = aux[x - pad_h[0]:x + pad_h[1] + 1, y - pad_w[0]:y + pad_w[1] + 1]
                    for index, value in np.ndenumerate(region):
                        dx, dy = index
                        if aux[x - pad_h[0] + dx, y - pad_w[0] + dy] == 0:
                            aux[x - pad_h[0] + dx, y - pad_w[0] + dy] = tom
                            lista.append((x - pad_h[0] + dx, y - pad_w[0] + dy))

    print(f"numero de objetos: {contagem}")
    return aux, contagem

def dilatation(img, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = (k_h // 2,k_h // 2), (k_w // 2,k_w // 2 )

    if k_h % 2 == 0:
        pad_h = (k_h // 2, k_h // 2 - 1)  
    if k_w % 2 == 0:
        pad_w = (k_w // 2, k_w // 2 - 1)  
    

    padded_image = np.pad(img, ((pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), mode='constant', constant_values=1)
    dilated_image = np.ones_like(img)

    for i in range(pad_h[0], padded_image.shape[0] - pad_h[1]):
        for j in range(pad_w[0], padded_image.shape[1] - pad_w[1]):
            region = padded_image[i - pad_h[0]:i + pad_h[1] + 1, j - pad_w[0]:j + pad_w[1] + 1]
            if np.any(region[kernel == 0] == 0):
                dilated_image[i - pad_h[0], j - pad_w[0]] = 0

    return dilated_image

def freeman(img):
    Aimg, Limg = img.shape
    sector_height = Aimg // 12
    sector_width = Limg // 12
    sectors = np.zeros((12, 12))
    sectors += 10

    for i in range(0, Aimg, sector_height):
        for j in range(0, Limg, sector_width):
            sector = img[i:i+sector_height, j:j+sector_width]
            white_pixels = np.count_nonzero(sector == 1)
            if white_pixels > (sector_height * sector_width) / 4:
                sectors[i//sector_height, j//sector_width] = 11
                img[i:i+sector_height, j:j+sector_width] = 1
            else:
                img[i:i+sector_height, j:j+sector_width] = 0
    
    vizinhos = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    order = [0, 2, 4, 6, 1, 3 ,5, 7]
    direction = 0
    current = (0, 0)
    looped = False
    for i, j in np.ndindex(sectors.shape):
        if sectors[i, j] == 11:
            current = (i, j)
            break

    freemanChain = ""
    while not looped:
        for i in order:
            if sectors[current[0] + vizinhos[i][0], current[1] + vizinhos[i][1]] == 11:
                sectors[current[0], current[1]] = i
                freemanChain += str(i)
                direction = i
                current = (current[0] + vizinhos[i][0], current[1] + vizinhos[i][1])
                break
            if i == 7:
                for i in order:
                    if i == (direction +4) % 8:
                        continue
                    if sectors[current[0] + vizinhos[i][0], current[1] + vizinhos[i][1]] < 10:
                        sectors[current[0], current[1]] = i
                        freemanChain += str(i)
                        break
                looped = True
                break

    for i in range(12):
        for j in range(12):
            print(int(sectors[i][j]) if sectors[i][j] != 10 else "_", end=" ")
        print()

    print("Cadeia de Freeman: ")
    print(freemanChain)
    return img, sectors, freemanChain

def box_filter(size):
    matriz = np.ones((size, size))
    matriz = matriz / (size * size)
    return matriz

def escala_cinza_segmentada(imagem):
    seg = np.copy(imagem)
    seg = np.where(seg <= 50, 25, np.where(seg <= 100, 75, np.where(seg <= 150, 125, np.where(seg <= 200, 175, 255))))
    return seg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operações para segmentação de imagens.")
    parser.add_argument(
        "operation", type=str, choices=["marr", "canny", "otsu", "objects", "freeman", "box", "grey", "q1", "q2"], help="Operação a ser realizada")
    parser.add_argument("image", type=str, help="Caminho para a imagem de entrada.")
    parser.add_argument("-o", type=str, help="Caminho para salvar a imagem de saída.", required=False)
    parser.add_argument("--sigma", type=float, help="Desvio padrão do filtro gaussiano.", required=False)
    parser.add_argument("--threshold", type=int, help="Limiar para cruzamentos por zero.", required=False)
    parser.add_argument("--size", type=int, help="Tamanho do kernel.", required=False)

    args = parser.parse_args() 
    label = "Resultado"
    binary = True
###############################################################

    imagem_exemplo, min, max = ler_imagem_cinza(args.image)
    sigma, threshold = math.ceil(min*0.005), math.ceil(max*0.2)
    threshold = args.threshold if args.threshold else threshold

    sigma = 1 if sigma == 0 else sigma
    sigma = args.sigma if args.sigma else sigma

    tamanho = (6*math.ceil(sigma)) + 1
    tamanho = int(args.size) if args.size else int(tamanho)

    if args.operation == "marr" or args.operation == "canny" or args.operation == "q1":
        print("Parâmetros:")
        print(f"Sigma: {sigma}")
        print(f"Threshold: {threshold}")
        print(f"Tamanho: {tamanho}")

###############################################################

    if args.operation == "marr":
        filtroGauss = criar_gaussiana(sigma, tamanho)
        gauss = aplicar_filtro(imagem_exemplo, filtroGauss)

        mascaraLaplace = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        log = aplicar_filtro(gauss, mascaraLaplace)
        resultado = cruzamentos_por_zero(log, threshold)
        label = "Marr-Hildreth"

    if args.operation == "canny":
        filtroGauss = criar_gaussiana(sigma, tamanho)
        gauss = aplicar_filtro(imagem_exemplo, filtroGauss)
        resultado = canny(threshold, 2*threshold, gauss)
        label = "Canny"

    if args.operation == "otsu":
        hist, bins = np.histogram(imagem_exemplo.flatten(), 256, [0, 256])
        otsu_t = otsu_threshold(imagem_exemplo.size, hist)
        resultado = corte_threshold_b(imagem_exemplo, otsu_t)
        label = "Otsu"
        print(f"Threshold Otsu: {otsu_t}")

    if args.operation == "q1" or args.operation == "q2":
        filtroGauss = criar_gaussiana(sigma, tamanho)
        gauss = aplicar_filtro(imagem_exemplo, filtroGauss)

        mascaraLaplace = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        marr = aplicar_filtro(gauss, mascaraLaplace)
        marr = cruzamentos_por_zero(marr, threshold)

        cannyImg = canny(threshold, 2*threshold, gauss)

        hist, bins = np.histogram(imagem_exemplo.flatten(), 256, [0, 256])
        otsu_t = otsu_threshold(imagem_exemplo.size, hist)
        otsu = corte_threshold_b(imagem_exemplo, otsu_t)
        print(f"Threshold Otsu: {otsu_t}")

    if args.operation == "objects":
        filtroGauss = criar_gaussiana(.84, 20)
        imagem_exemplo = aplicar_filtro(imagem_exemplo, filtroGauss)
        hist, bins = np.histogram(imagem_exemplo.flatten(), 256, [0, 256])
        otsu_t = otsu_threshold(imagem_exemplo.size, hist)
        resultado = corte_threshold_b(imagem_exemplo, otsu_t)
        resultado = dilatation(resultado, np.array([[0],[0],[0],[0],[0]]))
        resultado, numero = contagem_objetos(resultado)
        label = f"Foram encontrados {numero} objetos"

    if args.operation == "freeman":
        resultado = aplicar_filtro(imagem_exemplo, box_filter(9))
        hist, bins = np.histogram(resultado.flatten(), 256, [0, 256])
        otsu_t = otsu_threshold(resultado.size, hist)
        resultado = corte_threshold_b(resultado, otsu_t)
        x, y = math.ceil(resultado.shape[0] * 0.01), math.ceil(resultado.shape[1] * 0.01)
        resultado = dilatation(resultado, np.zeros((x,y)))
        resultado, heat, cadeia = freeman(resultado)

    if args.operation == "box":
        kernel = box_filter(tamanho)
        resultado1 = aplicar_filtro(imagem_exemplo, kernel)
        kernel = box_filter(tamanho+1)
        resultado2 = aplicar_filtro(imagem_exemplo, kernel)
        kernel = box_filter(tamanho+3)
        resultado3 = aplicar_filtro(imagem_exemplo, kernel)
        kernel = box_filter(tamanho+5)
        resultado = aplicar_filtro(imagem_exemplo, kernel)
        binary = False

    if args.operation == "grey":
        resultado = escala_cinza_segmentada(imagem_exemplo)
        label = "Escala de cinza segmentada"
        binary = False

####################### output

    if args.operation == "box":
        fig, ax = plt.subplots(1, 5, figsize=(10, 10))
        for a in ax:
            a.axis('off')
        ax[0].imshow(imagem_exemplo, cmap='gray')
        ax[0].set_title('Imagem original')
        ax[1].imshow(resultado1, cmap='gray')
        ax[1].set_title(f"Box ({tamanho}x{tamanho})")
        ax[2].imshow(resultado2, cmap='gray')
        ax[2].set_title(f"Box ({tamanho+1}x{tamanho+1})")
        ax[3].imshow(resultado3, cmap='gray')
        ax[3].set_title(f"Box ({tamanho+3}x{tamanho+3})")
        ax[4].imshow(resultado, cmap='gray')
        ax[4].set_title(f"Box ({tamanho+5}x{tamanho+5})")
        plt.show()
    elif args.operation == "q1":
        fig, ax = plt.subplots(1, 4, figsize=(10, 10))
        for a in ax:
            a.axis('off')
        ax[0].imshow(imagem_exemplo, cmap='gray')
        ax[0].set_title('Imagem original')
        ax[1].imshow(marr, cmap='gray')
        ax[1].set_title("Marr-Hildreth")
        ax[2].imshow(cannyImg, cmap='gray')
        ax[2].set_title("Canny")
        ax[3].imshow(otsu, cmap='gray')
        ax[3].set_title(f"Otsu (t={otsu_t})")
        fig.suptitle(f"Sigma: {sigma}, Tamanho: {tamanho}, Threshold: {threshold}\nTamanho da imagem: {imagem_exemplo.shape[0]}x{imagem_exemplo.shape[1]}", fontsize=12)
        plt.show()
    elif args.operation == "q2":
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        for a in ax:
            a.axis('off')
        ax[0].imshow(imagem_exemplo, cmap='gray')
        ax[0].set_title('Imagem original')
        ax[1].imshow(marr, cmap='gray')
        ax[1].set_title("Marr-Hildreth")
        ax[2].imshow(cannyImg, cmap='gray')
        ax[2].set_title("Canny")
        fig.suptitle(f"Sigma: {sigma}, Tamanho: {tamanho}, Threshold: {threshold}\nTamanho da imagem: {imagem_exemplo.shape[0]}x{imagem_exemplo.shape[1]}", fontsize=12)
        plt.show()
    elif args.operation == "freeman":
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        for a in ax:
            a.axis('off')
        ax[0].imshow(imagem_exemplo, cmap='gray')
        ax[0].set_title('Imagem original')
        ax[1].imshow(resultado, cmap='gray')
        ax[1].set_title("Imagem em setores")
        ax[2].imshow(heat, cmap='inferno', interpolation='nearest')
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax[2].text(j, i, str(int(heat[i, j])), ha="center", va="center", color="white", size=8)
        ax[2].set_title("Cadeia Freeman")
        fig.suptitle(f"Cadeia Freeman: {cadeia}", fontsize=12)
        plt.show()

    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        for a in ax:
            a.axis('off')
        ax[0].imshow(imagem_exemplo, cmap='gray')
        ax[0].set_title('Imagem original')
        ax[1].imshow(resultado, cmap='gray')
        ax[1].set_title(label)
        plt.show()

    if 'resultado' in locals() and args.o:
        cv2.imwrite(args.o, resultado*255 if binary else resultado)


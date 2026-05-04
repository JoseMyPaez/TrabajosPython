
import cv2  
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#imprimir todo
def printAll(matriz1, matriz2, matriz3, matriz4, matriz5):
    plt.figure(figsize=(10, 10))

    for i in range(1, 101):
        imagen = matriz1[i-1]
        plt.subplot(10, 10, i)
        plt.imshow(imagen)
        plt.axis("off")

    plt.figure(figsize=(10, 10))

    for i in range(1, 101):
        imagen = matriz2[i-1]
        plt.subplot(10, 10, i)
        plt.imshow(imagen)
        plt.axis("off")

    plt.figure(figsize=(10, 10))

    for i in range(1, 101):
        imagen = matriz3[i-1]
        plt.subplot(10, 10, i)
        plt.imshow(imagen)
        plt.axis("off")

    plt.figure(figsize=(10, 10))

    for i in range(1, 101):
        imagen = matriz4[i-1]
        plt.subplot(10, 10, i)
        plt.imshow(imagen)
        plt.axis("off")

    plt.figure(figsize=(10, 10))

    for i in range(1, 101):
        imagen = matriz5[i-1]
        plt.subplot(10, 10, i)
        plt.imshow(imagen)
        plt.axis("off")

def printkernel(listaA, listaB, listaC, listaD, listaE):
    # Definimos una cuadrícula de 5 filas y 6 columnas
    plt.figure(figsize=(18, 12))
    
    # Agrupamos las listas que enviaste
    todas = [listaA, listaB, listaC, listaD, listaE]
    nombres_filas = ["Obj A", "Obj B", "Obj C", "Obj D", "Obj E"]
    nombres_columnas = ["Original (Gris)", "Sobel", "Laplaciano", "Emboss", "Prewitt H", "Identidad"]

    for fila in range(5):
        for col in range(6):
            # El índice para una cuadrícula de 5x6 va de 1 a 30
            indice = (fila * 6) + col + 1
            plt.subplot(5, 6, indice)
            
            # Mostramos la imagen correspondiente
            plt.imshow(todas[fila][col], cmap='gray')
            plt.axis("off")
            
            # Títulos de las columnas (solo en la primera fila)
            if fila == 0:
                plt.title(nombres_columnas[col])
            
            # Nombres de las filas (solo en la primera columna)
            if col == 0:
                # Usamos text en lugar de ylabel para que no se encime
                plt.text(-20, todas[fila][col].shape[0]//2, nombres_filas[fila], 
                         va='center', ha='right', fontsize=12, fontweight='bold')

    plt.tight_layout()

def rgb_a_gris(imagen):
    
    filas, columnas = imagen.shape[:2]
    imagen_gris = np.zeros((filas, columnas), dtype=np.uint8)
# Convertir a escala de grises
    for i in range(0, filas-1):             
        for j in range(0, columnas-1):
        
            rojo  = float(imagen[i, j, 0]) 
            verde = float(imagen[i, j, 1])
            azul  = float(imagen[i, j, 2])
            
            gris = (rojo + verde + azul) / 3  
            # ... después de calcular la suma ...
            if gris > 255:
                gris = 255
            elif gris < 0:
                gris = 0

            imagen_gris[i, j] = np.uint8(gris)
    
    imagen_gris = np.array(imagen_gris)
            
    return imagen_gris        

def convolucion(imagen, kernel):    
    filas, columnas = imagen.shape
    
    imagen_filtrada = np.zeros((filas, columnas), dtype=np.uint8)
    
    for i in range(0, filas-2):
                
        for j in range(0, columnas-2):
                
            suma = 0
                
            H1 = imagen[i+0, j+0] * kernel[0, 0]
            H2 = imagen[i+0, j+1] * kernel[0, 1]
            H3 = imagen[i+0, j+2] * kernel[0, 2]
                
            H4 = imagen[i+1, j+0] * kernel[1, 0]
            H5 = imagen[i+1, j+1] * kernel[1, 1]
            H6 = imagen[i+1, j+2] * kernel[1, 2]
                
            H7 = imagen[i+2, j+0] * kernel[2, 0]
            H8 = imagen[i+2, j+1] * kernel[2, 1]
            H9 = imagen[i+2, j+2] * kernel[2, 2]
            
            suma = H1 + H2 + H3 + H4 + H5 + H6 + H7 + H8 + H9
            
            if suma > 255:
                suma = 255
            elif suma < 0:
                suma = 0     

            imagen_filtrada[i, j] = np.uint8(suma)
    return imagen_filtrada

def conversinaFloat(imagen):
    filas, columnas = imagen.shape
    imagen_float = np.zeros((filas, columnas), dtype=np.float32)
    
    for i in range(filas):
        for j in range(columnas):
            imagen_float[i, j] = float(imagen[i, j])
    return imagen_float

def promedioDimensional(matriz):
    num_filas = len(matriz)
    num_columnas = len(matriz[0])
    medias = [0] * num_columnas
    #fila por fila 
    for fila in matriz:
        for i in range(num_columnas):
            medias[i] += fila[i]
    #numcol
    for i in range(num_columnas):
        medias[i] = medias[i] / num_filas
    return medias

def promedio(imagen):
    filas, columnas = imagen.shape
    suma = 0.0
    for i in range(filas):
        for j in range(columnas):
            suma += imagen[i, j]
    promedio = suma / (filas * columnas)
    return promedio

def desviacion_estandar(matrizFloat, promedio):
    filas, columnas = matrizFloat.shape
    suma = 0.0
    for i in range(filas):
        for j in range(columnas):
            suma += (matrizFloat[i, j] - promedio) ** 2
    varianza = suma / (filas * columnas)
    desviacion_estandar = np.sqrt(varianza)
    return desviacion_estandar

def asimetria(matrizFloat):
    filas, columnas = matrizFloat.shape
    suma = 0.0
    for i in range(filas):
        for j in range(columnas):
            suma += matrizFloat[i, j] ** 3
    asimetria = suma / (filas * columnas)
    return asimetria
    
def kurtosis(matrizFloat):
    filas, columnas = matrizFloat.shape
    suma = 0.0
    for i in range(filas):
        for j in range(columnas):
            suma += matrizFloat[i, j] ** 4
    kurtosis = suma / (filas * columnas)
    return kurtosis

def calcular_descriptores(imagen):
    imagen_float = conversinaFloat(imagen)
    promedio_valor = promedio(imagen_float)
    desviacion_estandar_valor = desviacion_estandar(imagen_float, promedio_valor)
    asimetria_valor = asimetria(imagen_float)
    kurtosis_valor = kurtosis(imagen_float)
    
    return [promedio_valor, desviacion_estandar_valor, asimetria_valor, kurtosis_valor]
# Cargar las imágenes de la base de datos

def normalizarDescriptores(descriptores):
    descriptores = np.array(descriptores)
    #media = promedioDimensional(descriptores)
    media = np.mean(descriptores)
    desviacion_estandar = np.std(descriptores, axis=0)
    descriptores_normalizados = (descriptores - media) / desviacion_estandar
    return descriptores_normalizados

imagenesA = []

for i in range(1, 101):
    carpeta = Path("Base_datos_cereal") / "A"
    ruta = carpeta / f"A ({i}).jpg"
    imagen = cv2.imread(ruta)
    imagenesA.append(imagen)

imagenesB = []

for i in range(1, 101):
    carpeta = Path("Base_datos_cereal") / "B"
    ruta = carpeta / f"B ({i}).jpg"
    imagen = cv2.imread(ruta)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagenesB.append(imagen_rgb)
    
imagenesC  = []

for i in range(1, 101):
    carpeta = Path("Base_datos_cereal") / "C"
    ruta = carpeta / f"C ({i}).jpg"
    imagen = cv2.imread(ruta)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagenesC.append(imagen_rgb)

imagenesD = []

for i in range(1, 101):
    carpeta = Path("Base_datos_cereal") / "D"
    ruta = carpeta / f"D ({i}).jpg"
    imagen = cv2.imread(ruta)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagenesD.append(imagen_rgb)

imagenesE  = []

for i in range(1, 101):
    carpeta = Path("Base_datos_cereal") / "E"
    ruta = carpeta / f"E ({i}).jpg"
    imagen = cv2.imread(ruta)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagenesE.append(imagen_rgb)

print("Número de imágenes de A:", len(imagenesA))
print("Número de imágenes de B:", len(imagenesB))
print("Número de imágenes de C:", len(imagenesC))
print("Número de imágenes de D:", len(imagenesD))
print("Número de imágenes de E:", len(imagenesE))

printAll(imagenesA,imagenesB,imagenesC,imagenesD,imagenesE)


# 1. Kernel Sobel (Bordes principales y silueta) - En este caso Sobel en X
kernel_sobel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

# 2. Kernel Laplaciano (Detalles finos y el hueco central)
kernel_laplaciano = np.array([[ 0,  1,  0],
                              [ 1, -4,  1],
                              [ 0,  1,  0]])

# 3. Kernel Emboss (Relieve de inscripciones y textura)
kernel_emboss = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])

# 4. Kernel Prewitt Horizontal (Diferencias en geometría/bordes horizontales)
kernel_prewitt_h = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]])

# 5. Kernel Identidad (Reflectividad del metal - Imagen original en gris)
kernel_identidad = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]])

centroideA = []
centroideB = []
centroideC = []
centroideD = []
centroideE = []

avance = 0

for i in range(1, 71):
    
    avance += 1
    print(f"Avance: {avance}/70")

    imagenA = imagenesA[i-1]
    imagenB = imagenesB[i-1]
    imagenC = imagenesC[i-1]
    imagenD = imagenesD[i-1]
    imagenE = imagenesE[i-1]

    # Convertir a escala de grisesgggggggggggg
    imagenA_gris = rgb_a_gris(imagenA)
    imagenB_gris = rgb_a_gris(imagenB)
    imagenC_gris = rgb_a_gris(imagenC)
    imagenD_gris = rgb_a_gris(imagenD)
    imagenE_gris = rgb_a_gris(imagenE)

    # A Filtros
    imagenA_sobel = convolucion(imagenA_gris, kernel_sobel)
    imagenA_laplaciano = convolucion(imagenA_gris, kernel_laplaciano)
    imagenA_emboss = convolucion(imagenA_gris, kernel_emboss)
    imagenA_prewitt_h = convolucion(imagenA_gris, kernel_prewitt_h)
    imagenA_identidad = convolucion(imagenA_gris, kernel_identidad)

    #B Filtros
    imagenB_sobel = convolucion(imagenB_gris, kernel_sobel)
    imagenB_laplaciano = convolucion(imagenB_gris, kernel_laplaciano)
    imagenB_emboss = convolucion(imagenB_gris, kernel_emboss)
    imagenB_prewitt_h = convolucion(imagenB_gris, kernel_prewitt_h)
    imagenB_identidad = convolucion(imagenB_gris, kernel_identidad)

    #C Filtros
    imagenC_sobel = convolucion(imagenC_gris, kernel_sobel)
    imagenC_laplaciano = convolucion(imagenC_gris, kernel_laplaciano)
    imagenC_emboss = convolucion(imagenC_gris, kernel_emboss)
    imagenC_prewitt_h = convolucion(imagenC_gris, kernel_prewitt_h)
    imagenC_identidad = convolucion(imagenC_gris, kernel_identidad)

    #D Filtros
    imagenD_sobel = convolucion(imagenD_gris, kernel_sobel)
    imagenD_laplaciano = convolucion(imagenD_gris, kernel_laplaciano)
    imagenD_emboss = convolucion(imagenD_gris, kernel_emboss)
    imagenD_prewitt_h = convolucion(imagenD_gris, kernel_prewitt_h)
    imagen_D_identidad = convolucion(imagenD_gris, kernel_identidad)
    
    #E Filtros
    imagenE_sobel = convolucion(imagenE_gris, kernel_sobel)
    imagenE_laplaciano = convolucion(imagenE_gris, kernel_laplaciano)
    imagenE_emboss = convolucion(imagenE_gris, kernel_emboss)
    imagenE_prewitt_h = convolucion(imagenE_gris, kernel_prewitt_h)
    imagenE_identidad = convolucion(imagenE_gris, kernel_identidad)

    centroideA.append([calcular_descriptores(imagenA_sobel)+
                       calcular_descriptores(imagenA_laplaciano)+
                       calcular_descriptores(imagenA_emboss)+
                       calcular_descriptores(imagenA_prewitt_h)+
                       calcular_descriptores(imagenA_identidad)])
    
    centroideB.append([calcular_descriptores(imagenB_sobel)+
                       calcular_descriptores(imagenB_laplaciano)+
                       calcular_descriptores(imagenB_emboss)+
                       calcular_descriptores(imagenB_prewitt_h)+
                       calcular_descriptores(imagenB_identidad)])
    
    centroideC.append([calcular_descriptores(imagenC_sobel)+
                       calcular_descriptores(imagenC_laplaciano)+
                       calcular_descriptores(imagenC_emboss)+
                       calcular_descriptores(imagenC_prewitt_h)+
                       calcular_descriptores(imagenC_identidad)])
    
    centroideD.append([calcular_descriptores(imagenD_sobel)+
                       calcular_descriptores(imagenD_laplaciano)+
                       calcular_descriptores(imagenD_emboss)+
                       calcular_descriptores(imagenD_prewitt_h)+
                       calcular_descriptores(imagen_D_identidad)])

    centroideE.append([calcular_descriptores(imagenE_sobel)+
                       calcular_descriptores(imagenE_laplaciano)+
                       calcular_descriptores(imagenE_emboss)+
                       calcular_descriptores(imagenE_prewitt_h)+
                       calcular_descriptores(imagenE_identidad)])

    if i<=5:
        printkernel([imagenA_gris, imagenA_sobel, imagenA_laplaciano, imagenA_emboss, imagenA_prewitt_h, imagenA_identidad],
                    [imagenB_gris, imagenB_sobel, imagenB_laplaciano, imagenB_emboss, imagenB_prewitt_h, imagenB_identidad],
                    [imagenC_gris, imagenC_sobel, imagenC_laplaciano, imagenC_emboss, imagenC_prewitt_h, imagenC_identidad],
                    [imagenD_gris, imagenD_sobel, imagenD_laplaciano, imagenD_emboss, imagenD_prewitt_h, imagen_D_identidad],
                    [imagenE_gris, imagenE_sobel, imagenE_laplaciano, imagenE_emboss, imagenE_prewitt_h, imagenE_identidad])
    #calculo de descriptores 

centroideA = np.array(centroideA)
centroideB = np.array(centroideB)
centroideC = np.array(centroideC)
centroideD = np.array(centroideD)
centroideE = np.array(centroideE)

print("Forma de los centroides:")
print("Centroide A:", centroideA.shape)
print("Centroide B:", centroideB.shape)
print("Centroide C:", centroideC.shape)
print("Centroide D:", centroideD.shape)
print("Centroide E:", centroideE.shape)

'''
promCentroideA = np.mean(centroideA, axis=0)
promCentroideB = np.mean(centroideB, axis=0)
promCentroideC = np.mean(centroideC, axis=0)
promCentroideD = np.mean(centroideD, axis=0)
promCentroideE = np.mean(centroideE, axis=0)

'''

promCentroideA = promedioDimensional(centroideA)
promCentroideB = promedioDimensional(centroideB)
promCentroideC = promedioDimensional(centroideC)
promCentroideD = promedioDimensional(centroideD)
promCentroideE = promedioDimensional(centroideE)


print("Promedio de los centroides:")
print("Centroide A:", promCentroideA)
print("Centroide B:", promCentroideB)
print("Centroide C:", promCentroideC)
print("Centroide D:", promCentroideD)
print("Centroide E:", promCentroideE)

plt.show()

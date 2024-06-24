import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
ruta_archivo_pickle = 'C:\\Users\\luisf\\Documents\\GitHub\\keypoints-transformer\\datasetMascaras.pkl'
df = pd.read_pickle(ruta_archivo_pickle)
# Supongamos que df es tu DataFrame inicial
# columns son las columnas que contienen los keypoints en formato de cadena
columns = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
           'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
           'right_knee', 'left_ankle', 'right_ankle', 'mid_shoulder', 'mid_hip']

# Convertir las cadenas de texto de las columnas en numpy arrays
for column in columns:
    df[column] = df[column].apply(lambda x: np.array(ast.literal_eval(x)))

# Normalizar los keypoints
scaler = StandardScaler()
for column in columns:
    keypoints = np.stack(df[column].values)  # Convertir la columna en un array 2D
    keypoints_normalized = scaler.fit_transform(keypoints)  # Normalizar
    df[column] = [arr for arr in keypoints_normalized]  # Convertir de nuevo en una lista de arrays

# Eliminar la coordenada z de las primeras 13 columnas que tienen (x, y, z)
for column in columns[:13]:
    df[column] = df[column].apply(lambda x: x[:2])

# Verificar que todas las filas tienen el mismo número de dimensiones
for col in columns:
    lengths = df[col].apply(len).unique()
    if len(lengths) != 1:
        df = df[df[col].apply(len) == lengths[0]]

# Obtener los keypoints como un array numpy de forma (n_samples, num_keypoints, num_dimensions)
X_keypoints = np.array([np.concatenate(df[columns].iloc[i].values).ravel() for i in range(len(df))])

# Obtener las etiquetas
etiquetas = df['etiqueta'].values

# Codificar las etiquetas a valores numéricos
label_encoder = LabelEncoder()
etiquetas_codificadas = label_encoder.fit_transform(etiquetas)

# Convertir las etiquetas codificadas a float32
etiquetas_float = etiquetas_codificadas.astype(np.float32)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_keypoints, etiquetas_float, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

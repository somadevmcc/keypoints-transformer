import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Cargar el nuevo archivo CSV
ruta_nuevo_csv = 'C:\\Users\\luisf\\Documents\\GitHub\\keypoints-transformer\\datasetSujetos.csv'
df_nuevo = pd.read_csv(ruta_nuevo_csv)

# Convertir las cadenas de texto de las columnas en numpy arrays
columns = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
           'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
           'right_knee', 'left_ankle', 'right_ankle', 'mid_shoulder', 'mid_hip']

for column in columns:
    df_nuevo[column] = df_nuevo[column].apply(lambda x: np.array(ast.literal_eval(x)))

# Normalizar los keypoints usando el mismo scaler ajustado
scaler = StandardScaler()
for column in columns:
    keypoints = np.stack(df_nuevo[column].values)
    if keypoints.shape[1] == 3:  # Si tiene 3 características (x, y, z), eliminar z
        keypoints = keypoints[:, :2]
    keypoints_normalized = scaler.transform(keypoints)
    df_nuevo[column] = [arr for arr in keypoints_normalized]

# Eliminar la coordenada z de las primeras 13 columnas que tienen (x, y, z)
for column in columns[:13]:
    df_nuevo[column] = df_nuevo[column].apply(lambda x: x[:2])

# Verificar que todas las filas tienen el mismo número de dimensiones
for col in columns:
    lengths = df_nuevo[col].apply(len).unique()
    if len(lengths) != 1:
        df_nuevo = df_nuevo[df_nuevo[col].apply(len) == lengths[0]]

# Obtener los keypoints como un array numpy de forma (n_samples, num_keypoints, num_dimensions)
X_nuevo_keypoints = np.array([np.concatenate(df_nuevo[columns].iloc[i].values).ravel() for i in range(len(df_nuevo))])

# Convertir a tensores de PyTorch
X_nuevo = torch.tensor(X_nuevo_keypoints, dtype=torch.float32)


# Definir la arquitectura del modelo Transformer
class TransformerModel(nn.Module):
    def __init__(self, num_keypoints, num_dimensions, embed_dim, num_heads, ff_dim, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_dimensions = num_dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_classes = num_classes

        # Verificar si embed_dim es divisible por num_heads
        assert embed_dim % num_heads == 0, "embed_dim debe ser divisible por num_heads"

        # Capa de embeddings para los keypoints
        self.keypoints_embedding = nn.Linear(num_keypoints * num_dimensions, embed_dim)

        # Codificador Transformer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=1
        )

        # Capa de salida
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Aplanar los keypoints
        x = x.view(-1, self.num_keypoints * self.num_dimensions)

        # Capa de embeddings lineales
        x = self.keypoints_embedding(x)

        # Ajustar la forma para el Transformer (seq_len, batch, input_size)
        x = x.unsqueeze(1).transpose(0, 1)

        # Pasar por el codificador Transformer
        x = self.transformer_encoder(x)

        # Aplanar la salida y pasar por la capa de salida
        x = x.squeeze(0)
        x = self.output_layer(x)

        return x

# Inicializar el modelo
num_heads = 4
embed_dim = 32
ff_dim = 64
num_classes = 10  # Debes ajustar esto según tu número de clases

model = TransformerModel(num_keypoints=len(columns), num_dimensions=2, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_classes=num_classes)

# Cargar el estado del modelo preentrenado
model.load_state_dict(torch.load('ruta_al_archivo.pth'))
model.eval()  # Establecer el modelo en modo de evaluación


# Evaluar el modelo en los nuevos datos
with torch.no_grad():
    outputs_nuevo = model(X_nuevo)
    _, predicted_nuevo = torch.max(outputs_nuevo, 1)

# Codificar etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(df['etiqueta'].values)

# Obtener las etiquetas predichas
etiquetas_predichas = label_encoder.inverse_transform(predicted_nuevo.numpy())

# Mostrar las predicciones
print("Predicciones del modelo en el nuevo conjunto de datos:")
for i, etiqueta in enumerate(etiquetas_predichas):
    print(f"Muestra {i+1}: {etiqueta}")

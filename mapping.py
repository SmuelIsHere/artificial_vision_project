import numpy as np
import cv2
import math
import json

# File di input/output
output_txt_file = r"line_specs.txt"
config_file = r"config.txt"


with open(config_file, "r") as file:
    config = json.load(file)

thyaw = config["thyaw"]  
throll = config["throll"] 
thpitch = config["thpitch"]   
XC = config["xc"] 
YC = config["yc"] 
ZC = config["zc"]  
f = config["f"] 
U = config["U"]//1.7  
V = config["V"]//1.7 

# Estrazione delle coordinate reali x, z
x_real = []
z_real = []
id = []
for line in config["lines"]:
    id.append(line["id"])
    x_real.append(line["x1"])
    x_real.append(line["x2"])
    z_real.append(line["y1"])
    z_real.append(line["y2"])

x_real = np.array(x_real)
z_real = np.array(z_real)
y_real = -np.zeros_like(x_real)  # asse Z invertito (verso l'alto)

assert len(x_real) == len(y_real), "Le coordinate X e Y devono avere la stessa lunghezza."

# Parametri del sensore (pixel pitch)
SW = 0.00498  # Larghezza del sensore in metri
SH = 0.00374  # Altezza del sensore in metri

# Fattori di scala (pixel per metro sul sensore)
px_per_meter_x = U / SW
px_per_meter_y = V / SH

# Matrice intrinseca della fotocamera
K = np.array([
    [f * px_per_meter_x, 0, U / 2],
    [0, f * px_per_meter_y, V / 2],
    [0, 0, 1]
])

# Matrici di rotazione
Rx = np.array([
    [-1, 0, 0],
    [0, math.cos(thpitch), -math.sin(thpitch)],
    [0, math.sin(thpitch), math.cos(thpitch)]
])

Ry = np.array([
    [math.cos(throll), 0, math.sin(throll)],
    [0, 1, 0],
    [-math.sin(throll), 0, math.cos(throll)]
])

Rz = np.array([
    [math.cos(thyaw), -math.sin(thyaw), 0],
    [math.sin(thyaw), math.cos(thyaw), 0],
    [0, 0, 1]
])

R_reflect = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

# Combinazione della matrice di riflessione con la rotazione
R = R_reflect @ Rx @ Ry @ Rz
# Traslazione della fotocamera
T = np.array([XC, ZC, -YC]).reshape(3, 1)  #

# Matrice di proiezione (combinazione rotazione e traslazione)
RT = np.hstack((R, -R @ T))

# Punti 3D omogenei
points_3d = np.vstack((x_real, y_real, z_real, np.ones_like(x_real)))

# Matrice di proiezione completa
P = K @ RT
points_2d_homogeneous = P @ points_3d

# Normalizzazione per ottenere coordinate 2D (u, v)
# La normalizzazione deve avvenire per ogni punto (x', y', z')
u = points_2d_homogeneous[0] / points_2d_homogeneous[2]
v = points_2d_homogeneous[1] / points_2d_homogeneous[2]

# Salvataggio delle linee in formato line_specs
line_specs = []
for i, l in zip(range(0, len(u) - 1, 2), id):
    if(u[i]<u[i+1]):
        upwards=True #true vuol dire freccia su
    else:
        upwards=False #false vuol dire freccia giù
    
    line_specs.append({
        "id": int(l),
        "x1_start": int(round(u[i])),
        "x1_finish": int(round(u[i + 1])),
        "y1_start": int(round(v[i])),
        "y1_finish": int(round(v[i + 1])),
        "upwards": upwards
    })

# Scrittura delle specifiche delle linee in un file
with open(output_txt_file, "w") as f:
    for line in line_specs:
        f.write(f"{line}\n")

print(f"Dati salvati in {output_txt_file}")
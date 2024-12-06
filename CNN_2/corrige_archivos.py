import os

# Ruta al directorio que contiene los archivos
dataset_dir = 'C:/Users/marce/OneDrive/Imágenes/FotosDocentes/'  # Cambia esta ruta por la ruta correcta

# Función para renombrar los archivos en un directorio
def rename_files_in_directory(directory):
    # Iterar a través de todos los archivos en el directorio
    for filename in os.listdir(directory):
        # Crear la ruta completa del archivo
        file_path = os.path.join(directory, filename)

        # Verificar si es un archivo y no una carpeta
        if os.path.isfile(file_path):
            # Reemplazar los espacios por guiones bajos
            new_filename = filename.replace(" ", "_")

            # Crear la nueva ruta con el nombre modificado
            new_file_path = os.path.join(directory, new_filename)

            # Renombrar el archivo
            os.rename(file_path, new_file_path)

            print(f"Archivo renombrado: {filename} -> {new_filename}")

# Llamar a la función para renombrar los archivos en los directorios 'Hombres' y 'Mujeres'
hombres_dir = os.path.join(dataset_dir, 'Hombres')
mujeres_dir = os.path.join(dataset_dir, 'Mujeres')

print("Renombrando archivos en Hombres...")
rename_files_in_directory(hombres_dir)

print("Renombrando archivos en Mujeres...")
rename_files_in_directory(mujeres_dir)

print("Renombramiento completado.")

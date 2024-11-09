import subprocess


glew_include = "C:/C_LIBRERIAS/glew-2.1.0/include"
sdl2_include = "C:/C_LIBRERIAS/SDL2-2.30.9/include"
cuda_include = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
glew_lib = "C:/C_LIBRERIAS/glew-2.1.0/lib/Release/x64"
sdl2_lib = "C:/C_LIBRERIAS/SDL2-2.30.9/lib/x64"
cuda_lib = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"


cu_file = "effect.cu"
cpp_file = "main.cpp"
output_exe = "sdl_opengl_cuda_example.exe"


nvcc_command = [
    "nvcc", cpp_file, cu_file, "-o", output_exe,
    f"-I{glew_include}", f"-I{sdl2_include}", f"-I{cuda_include}",
    f"-L{glew_lib}", "-lglew32",
    f"-L{sdl2_lib}", "-lSDL2main", "-lSDL2",
    f"-L{cuda_lib}", "-lcuda", "-lcudart",
    "-lopengl32", "-lgdi32", "-lshell32", "-Xlinker", "/SUBSYSTEM:CONSOLE"
]

try:
    print("Compilando y enlazando archivos con nvcc...")
    subprocess.check_call(nvcc_command)

    print(f"Compilación exitosa. Ejecutable generado: {output_exe}")

except subprocess.CalledProcessError as e:
    print("Error durante la compilación:", e)

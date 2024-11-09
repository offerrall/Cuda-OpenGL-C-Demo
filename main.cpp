#include <GL/glew.h>
#include <SDL.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void launch_cuda_effect(uchar4* d_output, int width, int height, int* d_current, int* d_next);

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Error al inicializar SDL: %s\n", SDL_GetError());
        return -1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

    int width = 1920;
    int height = 1080;

    SDL_Window* window = SDL_CreateWindow("Juego de la Vida con CUDA",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Error al crear la ventana SDL: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        fprintf(stderr, "Error al crear el contexto OpenGL: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Error al inicializar GLEW: %s\n", glewGetErrorString(err));
        SDL_GL_DeleteContext(glContext);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    glViewport(0, 0, width, height);

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsResource* cudaPboResource;
    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int* d_current;
    int* d_next;
    cudaMalloc(&d_current, width * height * sizeof(int));
    cudaMalloc(&d_next, width * height * sizeof(int));

    int* h_initial = (int*)malloc(width * height * sizeof(int));
    for (int i = 0; i < width * height; ++i) {
        h_initial[i] = rand() % 2;
    }
    cudaMemcpy(d_current, h_initial, width * height * sizeof(int), cudaMemcpyHostToDevice);
    free(h_initial);

    int running = 1;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = 0;
        }

        uchar4* d_output;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cudaPboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cudaPboResource);

        launch_cuda_effect(d_output, width, height, d_current, d_next);

        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture);

        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();

        glDisable(GL_TEXTURE_2D);
        SDL_GL_SwapWindow(window);

        int* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    cudaGraphicsUnregisterResource(cudaPboResource);
    cudaFree(d_current);
    cudaFree(d_next);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

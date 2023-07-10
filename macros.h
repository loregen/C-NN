#ifndef MACROS_H
#define MACROS_H

#define ALLOCA(arr, size, type, msg)                        \
    do {                                                    \
        arr = malloc((size) * sizeof(type));                \
        if (arr == NULL) {                                  \
            printf(msg);                                    \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

#define CALLOCA(arr, size, type, msg)                       \
    do {                                                    \
        arr = calloc(size, sizeof(type));                   \
        if (arr == NULL) {                                  \
            printf(msg);                                    \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

#define REALLOCA(arr, newsize, type, tmp, msg)              \
    do {                                                    \
        tmp = realloc(arr, (newsize) * sizeof(type));       \
        if (tmp == NULL) {                                  \
            printf(msg);                                    \
            exit(EXIT_FAILURE);                             \
        }                                                   \
        arr = tmp;                                          \
    } while (0)




#define SQUARE(x) ((x)*(x))

#endif
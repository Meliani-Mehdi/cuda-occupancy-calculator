#ifndef QUEUE_H
#define QUEUE_H

#include <stdio.h>
#include <stdlib.h>

/*
 * Macro to define a queue for a given type
 *
 * Example:
 *   QUEUE_DEFINE(Kernel_t, kernel)
 *   -> defines struct queue_kernel, and functions:
 *        void queue_kernel_init(...)
 *        void queue_kernel_free(...)
 *        int  queue_kernel_empty(...)
 *        int  queue_kernel_full(...)
 *        void queue_kernel_enqueue(...)
 *        void queue_kernel_dequeue(...)
 */

#define QUEUE_DEFINE(TYPE, NAME)                                           \
typedef struct {                                                           \
    TYPE *data;                                                            \
    int head, tail;                                                        \
    int capacity;                                                          \
} queue_##NAME##_t;                                                        \
                                                                           \
static inline void queue_##NAME##_init(queue_##NAME##_t *q, int capacity) {\
    q->data = (TYPE *) malloc(sizeof(TYPE) * capacity);                    \
    if (!q->data) {                                                        \
        fprintf(stderr, "Memory allocation failed\n");                     \
        exit(1);                                                           \
    }                                                                      \
    q->head = q->tail = 0;                                                 \
    q->capacity = capacity;                                                \
}                                                                          \
                                                                           \
static inline void queue_##NAME##_free(queue_##NAME##_t *q) {              \
    free(q->data);                                                         \
    q->data = NULL;                                                        \
    q->head = q->tail = q->capacity = 0;                                   \
}                                                                          \
                                                                           \
static inline int queue_##NAME##_empty(const queue_##NAME##_t *q) {        \
    return q->head == q->tail;                                             \
}                                                                          \
                                                                           \
static inline int queue_##NAME##_full(const queue_##NAME##_t *q) {         \
    return ((q->tail + 1) % q->capacity) == q->head;                       \
}                                                                          \
                                                                           \
static inline void queue_##NAME##_enqueue(queue_##NAME##_t *q, TYPE value) {\
    if (queue_##NAME##_full(q)) {                                          \
        fprintf(stderr, "Queue overflow\n");                               \
        return;                                                            \
    }                                                                      \
    q->data[q->tail] = value;                                              \
    q->tail = (q->tail + 1) % q->capacity;                                 \
}                                                                          \
                                                                           \
static inline TYPE queue_##NAME##_dequeue(queue_##NAME##_t *q) {           \
    if (queue_##NAME##_empty(q)) {                                         \
        fprintf(stderr, "Queue underflow\n");                              \
        exit(1);                                                           \
    }                                                                      \
    TYPE val = q->data[q->head];                                           \
    q->head = (q->head + 1) % q->capacity;                                 \
    return val;                                                            \
}

#endif // QUEUE_H

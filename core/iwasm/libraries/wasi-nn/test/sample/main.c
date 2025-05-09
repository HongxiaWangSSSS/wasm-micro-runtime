/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "nn.h"
#include "logger.h"

#undef EPSILON
#define EPSILON 1e-2

void
parse_detection_results(float *out_tensor, uint32_t output_size)
{
    uint32_t offset = 0;

    uint32_t num = (uint32_t)out_tensor[output_size - 1]; // num 是最后一个元素
    printf("Total number of detections: %d\n", num);

    printf("Boxes (ymin, xmin, ymax, xmax):\n");
    float *boxes = &out_tensor[offset];
    for (int i = 0; i < num; i++) {
        printf("Box %d: [%f, %f, %f, %f]\n", i, boxes[i * 4] * 576,
               boxes[i * 4 + 1] * 768, boxes[i * 4 + 2] * 576,
               boxes[i * 4 + 3] * 768);
    }
    offset += num * 4;

    printf("Classes (category IDs):\n");
    float *classes = &out_tensor[offset];
    for (int i = 0; i < num; i++) {
        printf("Class %d: %f\n", i, classes[i]);
    }
    offset += num;

    printf("Scores (confidence):\n");
    float *scores = &out_tensor[offset];
    for (int i = 0; i < num; i++) {
        printf("Score %d: %f\n", i, scores[i]);
    }
    offset += num;

    printf("Parsing complete.\n");
}

static double
time_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
}

void
test_detection(execution_target target)
{
    FILE *file = fopen("input_tensor.bin", "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    size_t tensor_size = 1 * 300 * 300 * 3; // shape (1, 300, 300, 3)
    uint32_t dim[] = { 1, 300, 300, 3 };
    // size_t tensor_size = 1 * 320 * 320 * 3; 
    // uint32_t dim[] = { 1, 320, 320, 3 };


    uint8_t *input_tensor = (uint8_t *)malloc(tensor_size * sizeof(uint8_t));
    if (input_tensor == NULL) {
        perror("Memory allocation failed");
    }

    fread(input_tensor, sizeof(uint8_t), tensor_size, file);
    fclose(file);
    assert(input_tensor[0] == 58);
    assert(input_tensor[1] == 59);

    uint32_t output_size = 0;
    double start_time_ = time_ms();
    float *output = run_inference(target, input_tensor, dim, &output_size,
                                  "./detection.tflite", 4, up8);
    double end_time_ = time_ms();
    printf("Detection size = 4MB, Inference time: %f ms\n", end_time_ - start_time_);
    parse_detection_results(output, output_size);
    free(input_tensor);
    free(output);
}

// Function to read the labels from the file and store them in an array
int
load_labels(const char *filename, char ***labels)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening label file.\n");
        return -1;
    }

    // Calculate the number of lines in the file
    int num_labels = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        num_labels++;
    }

    // Allocate memory for the labels
    *labels = (char **)malloc(num_labels * sizeof(char *));
    fseek(file, 0, SEEK_SET); // Reset the file pointer

    // Read the labels into the array
    int i = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        buffer[strcspn(buffer, "\n")] = 0; // Remove newline character
        (*labels)[i] = strdup(buffer);     // Allocate and copy label
        i++;
    }

    fclose(file);
    return num_labels;
}

// Function to find the index of the maximum value in the output array
int
find_max_index(float *output, int num_classes)
{
    int max_index = 0;
    float max_value = output[0];

    for (int i = 1; i < num_classes; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            max_index = i;
        }
    }

    return max_index;
}

// Function to parse the inference result and print the predicted label
void
parse_inference_result(float *output, int num_classes)
{
    // Load the labels from the file
    char **labels;
    int num_labels = load_labels("label_mobile.txt", &labels);
    if (num_labels == -1) {
        printf("Error loading labels.\n");
        return;
    }

    int max_index = find_max_index(output, num_classes);
    printf("Predicted label: %s\n", labels[max_index]);

    // Free memory
    for (int i = 0; i < num_labels; i++) {
        free(labels[i]);
    }
    free(labels);
}

void
test_classification(execution_target target)
{
    uint32_t dim[] = { 1, 224, 224, 3 };
    FILE *file = fopen("input_tensor_224.bin", "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    size_t tensor_size = 1 * 224 * 224 * 3; //

    float *input_tensor = (float *)malloc(tensor_size * sizeof(float));
    if (input_tensor == NULL) {
        perror("Memory allocation failed");
    }

    fread(input_tensor, sizeof(float), tensor_size, file);
    fclose(file);
    assert(fabs(input_tensor[0] - 0.039215688) < EPSILON);
    assert(fabs(input_tensor[1] - 0.0627451) < EPSILON);

    uint32_t output_size = 0;
    double start_time_ = time_ms();
    float *output = run_inference(target, input_tensor, dim, &output_size,
                                  "./mobilenet_v2_1.0_224.tflite", 1, fp32);
    double end_time_ = time_ms();
    printf("Classification size = 8MB, Inference time: %f ms\n", end_time_ - start_time_);

    // Parse and print the result
    parse_inference_result(output, output_size);

    free(input_tensor);
    free(output);
}
int
main()
{
    char *env = getenv("TARGET");
    if (env == NULL) {
        NN_INFO_PRINTF("Usage:\n--env=\"TARGET=[cpu|gpu|tpu]\"");
        return 1;
    }
    execution_target target;
    if (strcmp(env, "cpu") == 0)
        target = cpu;
    else if (strcmp(env, "gpu") == 0)
        target = gpu;
    else if (strcmp(env, "tpu") == 0)
        target = tpu;
    else {
        NN_ERR_PRINTF("Wrong target!");
        return 1;
    }

    // NN_INFO_PRINTF("################### Testing detection model...");
    // test_detection(target);
    NN_INFO_PRINTF("################### Testing classification model...");
    test_classification(target);

    NN_INFO_PRINTF("Tests: passed!");
    return 0;
}

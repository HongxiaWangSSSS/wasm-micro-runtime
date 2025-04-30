/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "logger.h"

#undef EPSILON
#define EPSILON 1e-2

void
test_average_quantized(execution_target target)
{
    int dims[] = { 1, 5, 5, 1 };
    input_info input = create_input(dims);

    uint32_t output_size = 0;
    float *output =
        run_inference(target, input.input_tensor, input.dim, &output_size,
                      "./models/quantized_model.tflite", 1);

    NN_INFO_PRINTF("Output size: %d", output_size);
    NN_INFO_PRINTF("Result: average is %f", output[0]);
    // NOTE: 11.95 instead of 12 because of errors due quantization
    assert(fabs(output[0] - 11.95) < EPSILON);

    free(input.dim);
    free(input.input_tensor);
    free(output);
}

void
parse_detection_results(float *out_tensor, uint32_t output_size)
{
    uint32_t offset = 0;

    uint32_t num = (uint32_t)out_tensor[output_size - 1];
    printf("Total number of detections: %d\n", num);

    printf("Boxes (ymin, xmin, ymax, xmax):\n");
    float *boxes = &out_tensor[offset];
    for (int i = 0; i < num; i++) {
        printf("Box %d: [%f, %f, %f, %f]\n", i, boxes[i * 4] * 576,
               boxes[i * 4 + 1] * 768, boxes[i * 4 + 2] * 576,
               boxes[i * 4 + 3] * 768); //img (w,h)
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

void
test_detection(execution_target target)
{
    uint32_t dim[] = { 1, 300, 300, 3 };
    FILE *file = fopen("input_tensor.bin", "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    size_t tensor_size = 1 * 300 * 300 * 3; 

    uint8_t *input_tensor = (uint8_t *)malloc(tensor_size * sizeof(uint8_t));
    if (input_tensor == NULL) {
        perror("Memory allocation failed");
    }

    fread(input_tensor, sizeof(uint8_t), tensor_size, file);
    fclose(file);
    printf("input_tensor[0][0][0] = %d\n", input_tensor[0]);
    printf("input_tensor[0][0][0] = %d\n", input_tensor[1]);
    assert(input_tensor[0] == 58);
    assert(input_tensor[1] == 59);

    uint32_t output_size = 0;
    float *output = run_inference(target, input_tensor, dim, &output_size,
                                  "./detection.tflite", 4);

    NN_INFO_PRINTF("Output size: %d", output_size);
    parse_detection_results(output, output_size);
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
    // NN_INFO_PRINTF("################### Testing quantized model...");
    // test_average_quantized(target);
    NN_INFO_PRINTF("################### Testing detection model...");
    test_detection(target);

    NN_INFO_PRINTF("Tests: passed!");
    return 0;
}

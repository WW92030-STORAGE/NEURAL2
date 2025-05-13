#ifndef LAYER_NEURAL2
#define LAYER_NEURAL2


#include <vector>
#include <string>

#include "NN_IMPORTANT.h"


#include <iostream>

// Base Layer

class Layer {
    public:
    bool verbose = false;
    // Dimension of inputs and outputs
    int N_INPUTS; // + 1 for bias
    int N_OUTPUTS;

    // Learning rate
    NN_NUMERIC_T LR = 0.1;
    // most recent outputs/inputs
    NN_NUMERIC_T* mro;
    NN_NUMERIC_T* mri;

    void initMR() {
        mro = new NN_NUMERIC_T[N_OUTPUTS];
        mri = new NN_NUMERIC_T[N_INPUTS + 1];

        for (int i = 0; i <= N_INPUTS; i++) mri[i] = 1;
        for (int i = 0; i < N_OUTPUTS; i++) mro[i] = 0;
    }

    Layer() {
        N_INPUTS = 2;
        N_OUTPUTS = 2;
        initMR();
    }

    Layer(int NI, int NO) {
        N_INPUTS = NI;
        N_OUTPUTS = NO;
        initMR();
    }

    Layer(const Layer& other) {
        N_INPUTS = other.N_INPUTS;
        N_OUTPUTS = other.N_OUTPUTS;
        LR = other.LR;
        initMR();
        for (int i = 0; i <= N_INPUTS; i++) mri[i] = other.mri[i];
        for (int i = 0; i < N_OUTPUTS; i++) mro[i] = other.mro[i];
    }

    virtual ~Layer() {
        delete[] mro;
        delete[] mri;
    }

    virtual std::string to_string() {
        return "Layer[]";
    }

    void setMR(std::vector<NN_NUMERIC_T> input, std::vector<NN_NUMERIC_T> output) {
        for (int i = 0; i < input.size() && i < N_INPUTS; i++) mri[i] = input[i];
        for (int j = 0; j < output.size() && j < N_OUTPUTS; j++) mro[j] = output[j];
    }


    virtual std::vector<NN_NUMERIC_T> forward(std::vector<NN_NUMERIC_T> input) {
        std::vector<NN_NUMERIC_T> output(N_OUTPUTS, 0);
        setMR(input, output);
        return output;
        
    }

    std::vector<NN_NUMERIC_T> operator()(std::vector<NN_NUMERIC_T> input) {
        return forward(input);
    }

    // backprop using the most recent forward pass. returns gradients with respect to inputs.
    virtual std::vector<NN_NUMERIC_T> backward(std::vector<NN_NUMERIC_T> gradients) {
        return std::vector<NN_NUMERIC_T>(N_INPUTS, 0);
    }
};

#endif
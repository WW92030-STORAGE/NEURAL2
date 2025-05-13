#ifndef STACK_NEURAL2
#define STACK_NEURAL2

/*

A stack of NN layers! You put in an input, get an output, it can also backprop using a single method!

*/

#include <vector>
#include <string>

#include "Layer.h"
#include "LayerTypes.h"
#include "NN_IMPORTANT.h"

class LayerStack {
    public:
    std::vector<Layer*> layers;
    bool verbose = false;
    // Dimension of inputs and outputs
    int N_INPUTS; // Does not include any constants or biases
    int N_OUTPUTS;

    LayerStack() {
        LinearLayer* L = new LinearLayer(2, 2);
        SigmoidLayer* S = new SigmoidLayer(2);
        layers = std::vector<Layer*>({L, S});
        N_INPUTS = 2;
        N_OUTPUTS = 2;
    }

    LayerStack(int NI, int NO) {
        N_INPUTS = NI;
        N_OUTPUTS = NO;
        LinearLayer* L = new LinearLayer(NI, NO);
        SigmoidLayer* S = new SigmoidLayer(NO);
        layers = std::vector<Layer*>({L, S});
    }

    LayerStack(std::vector<Layer*> L) {
        layers = std::vector<Layer*>(L);
    }

    ~LayerStack() {
    }

    void setLR(NN_NUMERIC_T lr) {
        for (int i = 0; i < layers.size(); i++) layers[i]->LR = lr;
    }

    std::string to_string() {
        std::string res = "LayerStack[";

        for (int i = 0; i < layers.size(); i++) {
            if (i) res = res + " ";
            res = res + layers[i]->to_string() + ",";
        }
        return res + "]";
    }

    std::vector<NN_NUMERIC_T> forward(std::vector<NN_NUMERIC_T> input) {
        for (int i = 0; i < layers.size(); i++) input = layers[i]->forward(input);
        return input;
    }

    std::vector<NN_NUMERIC_T> operator()(std::vector<NN_NUMERIC_T> input) {
        return forward(input);
    }

    std::vector<NN_NUMERIC_T> backward(std::vector<NN_NUMERIC_T> gradients) {
        for (int i = layers.size() - 1; i >= 0; i--) gradients = layers[i]->backward(gradients);
        return gradients;
    }
};

#endif
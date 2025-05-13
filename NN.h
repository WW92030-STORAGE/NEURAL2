#ifndef NN_NEURAL2
#define NN_NEURAL2

#include "NN_IMPORTANT.h"
#include "Layer.h"

#include <vector>
#include <string>


#include <iostream>

// Simple dense linear layer

class LinearLayer : public Layer {
    public:
    
    NN_NUMERIC_T** weights;
    

    // weights[i][j] represents that output j will have input i * weights[i][j] added to it.
    // inputs start at 0, input no. N_INPUTS is the "bias" which is 1 and is controlled by weights[N_INPUTS][j].
    inline void init() {
        weights = new NN_NUMERIC_T*[N_INPUTS + 1];

        for (int i = 0; i <= N_INPUTS; i++) {
            
            weights[i] = new NN_NUMERIC_T[N_OUTPUTS];
            for (int j = 0; j < N_OUTPUTS; j++) weights[i][j] = randradius();
        }
        
    }

    LinearLayer() : Layer() {
        init();
    }

    LinearLayer(int NI, int NO) : Layer(NI, NO) {
        init();
    }

    LinearLayer(const LinearLayer& other) : Layer(other) {
        init();
        for (int i = 0; i <= N_INPUTS; i++) {
            for (int j = 0; j < N_OUTPUTS; j++) weights[i][j] = other.weights[i][j];
        } 
    }

    ~LinearLayer() {
        for (int i = 0; i <= N_INPUTS; i++) delete[] weights[i];
        delete[] weights;
    }

    virtual std::string to_string() {
        std::string res = "LinearLayer[";
        for (int i = 0; i <= N_INPUTS; i++) {
            res += toString(weights[i], N_OUTPUTS);
        }
        res = res + "]";
        return res;
    }

    // Forward propagation

    std::vector<NN_NUMERIC_T> forward(std::vector<NN_NUMERIC_T> inputs) {
        if (inputs.size() != N_INPUTS) {
            auto oo = std::vector<NN_NUMERIC_T>(N_OUTPUTS, 0);
            return oo;
        }

        std::vector<NN_NUMERIC_T> outputs(N_OUTPUTS, 0.0);

        for (int j = 0; j < N_OUTPUTS; j++) {
            for (int i = 0; i < N_INPUTS; i++) outputs[j] += weights[i][j] * inputs[i];
            outputs[j] += weights[N_INPUTS][j];
        }
        setMR(inputs, outputs);
        return outputs;
    }

    /*
    
    Linear layer has inputs I[0 ... NI] where I[NI] = 1 (bias)
    and outputs O[0 ... NO - 1]. The total error at the end of the neural network is E.

    Suppose that the gradients upon reaching the outputs of this layer are G[0 ... NO - 1] where G[j] = dE / dO[j]
    We need to compute two things: dE / dW[i][j] and dE / dI[i]. 
    The values dE/dI[i] will then be fed into the preceding layers and will be returned.
    And the dE / dW[i][j] will be used to update the weights.

    The weights are simple enough:
    dE/dW[i][j] = dE/dO[j] * dO[j]/dW[i][j] = dE/dO[j] * I[i]

    The values are harder. Remember you can add the gradients of individual outputs
    dE/dI[i] = sum dE/dO[j] * dO[j]/dI[i] = sum dE/dO[j] * W[i][j]
    
    */
    std::vector<NN_NUMERIC_T> backward(std::vector<NN_NUMERIC_T> gradients) {
        if (verbose) std::cout << "Linear Rev " << toString(gradients) << "\n";
        std::vector<std::vector<NN_NUMERIC_T>> wgrads(N_INPUTS + 1, std::vector<NN_NUMERIC_T>(N_OUTPUTS, 0));
        for (int i = 0; i <= N_INPUTS; i++) {
            for (int j = 0; j < N_OUTPUTS; j++) wgrads[i][j] = gradients[j] * mri[i];
        }

        std::vector<NN_NUMERIC_T> vgrads(N_INPUTS, 0);

        for (int i = 0; i < N_INPUTS; i++) {
            for (int j = 0; j < N_OUTPUTS; j++) vgrads[i] += gradients[j] * weights[i][j];
        }

        if (verbose)  {
            std::cout << "Linear Grads " << toString(vgrads) << "\n";
            std::cout << "Begin wgrads\n";
            for (int i = 0; i <= N_INPUTS; i++) {
                for (int j = 0; j < N_OUTPUTS; j++) std::cout << wgrads[i][j] << " ";
                std::cout << "\n";
            }
            std::cout << "End wgrads\n";
        }

        // update the weights

        for (int i = 0; i <= N_INPUTS; i++) {
            for (int j = 0; j < N_OUTPUTS; j++) weights[i][j] -= LR * wgrads[i][j];
        }
        if (verbose) {
            std::cout << "Begin weights\n";
            for (int i = 0; i <= N_INPUTS; i++) {
                for (int j = 0; j < N_OUTPUTS; j++) std::cout << weights[i][j] << " ";
                std::cout << "\n";
            }
            std::cout << "End weights\n";
        }

        return vgrads;
    }
};

class SigmoidLayer : public Layer {
    public:

    SigmoidLayer() : Layer() {
        N_INPUTS = 2;
        N_OUTPUTS = 2;
    }

    SigmoidLayer(int N) : Layer(N, N) {
    }

    SigmoidLayer(const SigmoidLayer& other) : Layer(other) {

    }

    ~SigmoidLayer() {

    }

    std::vector<NN_NUMERIC_T> forward(std::vector<NN_NUMERIC_T> input) {
        std::vector<NN_NUMERIC_T> output(N_OUTPUTS, 0);
        for (int i = 0; i < N_INPUTS; i++) output[i] = sigmoid(input[i]);
        setMR(input, output);
        return output;
    }

    // backprop using the most recent forward pass. returns gradients with respect to inputs.
    // dE/dI = dE/dS * dS/dI
    std::vector<NN_NUMERIC_T> backward(std::vector<NN_NUMERIC_T> gradients) {
        std::vector<NN_NUMERIC_T> vgrads(N_INPUTS, 0);
        if (verbose) std::cout << "Sigmoid Rev " << toString(mro, N_OUTPUTS) << " " << toString(gradients) << "\n";

        for (int i = 0; i < N_INPUTS; i++) vgrads[i] = sigd(mri[i], mro[i]) * gradients[i];
        if (verbose) std::cout << "Sigmoid Grads " << toString(vgrads) << "\n";

        return vgrads;
    }
};

#endif
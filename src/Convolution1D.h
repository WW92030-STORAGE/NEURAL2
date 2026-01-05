#ifndef CONV1D_NEURAL2  
#define CONV1D_NEURAL2

#include "NN_IMPORTANT.h"
#include "Layer.h"
#include "NNLayer.h"

#include <vector>
#include <string>
#include <algorithm>

/*

Convolutional layer with stride 1 (scans across all windows) (extremely crude implementation)

*/

class Convolution1D : public Layer {
    public:
    int WIDTH = 1;

    NN_NUMERIC_T* filter = 0;

    void init() {
                // first window is from 0 ... W - 1
        // second from 1 ... W
        // all the way to (NI - W) ... (NI - 1)
        // NI - W + 1 = NO --> W = NI + 1 - NO
        WIDTH = N_INPUTS + 1 - N_OUTPUTS;
        if (filter) delete[] filter;
        filter = new NN_NUMERIC_T[WIDTH];
        for (int i = 0; i < WIDTH; i++) filter[i] = randradius();
    }

    Convolution1D() : Layer() {
        init();
    }

    Convolution1D(int NI, int NO) : Layer(NI, NO) {
        init();
    }

    Convolution1D(const Convolution1D& other) : Layer(other) {
        // do stuff here
        init();

        for (int i = 0; i < WIDTH; i++) filter[i] = other.filter[i];
    }

    ~Convolution1D() {
        if (filter) delete[] filter;
    }
    
    virtual std::string to_string() {
        std::string res = Layer::to_string() + ": Convolution1D[";
        for (int i = 0; i < WIDTH; i++) {
            if (i) res = res + ", ";
            res = res + std::to_string(filter[i]);
        }
        return res + "]";
    }

    // Forward propagation

    std::vector<NN_NUMERIC_T> forward(std::vector<NN_NUMERIC_T> input) {
        std::vector<NN_NUMERIC_T> output(N_OUTPUTS, 0);

        for (int i = 0; i < N_OUTPUTS; i++) {
            NN_NUMERIC_T res = 0;
            for (int j = 0; j < WIDTH; j++) res += filter[WIDTH - 1 - j] * input[i + j]; // Flip the kernel in convolution
            output[i] = res;
        }

        setMR(input, output);
        return output;
        
    }

    // backprop using the most recent forward pass. returns gradients with respect to inputs.
    std::vector<NN_NUMERIC_T> backward(std::vector<NN_NUMERIC_T> gradients) {
        std::vector<NN_NUMERIC_T> vgrads(N_INPUTS, 0);
        std::vector<NN_NUMERIC_T> wgrads(WIDTH, 0);

        /*
        
        simple way to do this is to treat this as multiple inputs to input into the same small neural network "filter" and then produce each output
        to do backpropagation on each output:

        OUT[i] = IN[i] * W[0] + IN[i + 1] * W[1] ... + IN[i + w - 1] * W[w - 1]

        dE/dIN[i] = sum over all applicable j, k, m (one k, m per j): dE/dOUT[j] * dOUT[j]/dIN[k] = gradients[j] * W[m] <-- W[m] is the coefficient of IN[k] for OUT[j]
        dE/dW[i] = sum over all applicable j, k (one k per j): dE/dOUT[j] * dOUT[j] / dW[i] = gradients[j] * IN[k] <-- W[i] is the coefficient of IN[k] for OUT[j]

        But if k, m are uniquely determined by j, i, then how do we determine these?

        For dE/dIN[i]: OUT[j] = IN[i] * W[m] + ... satisfies (i - m) = j. Thus m = i - j
        For dE/dW[i]: OUT[j] = IN[k] * W[i] + ... satisfies (k - i) = j. Thus k = j + i.

        Reassigning variables to keep things consistent:
        OUT[i] = IN[j] * W[k] sum:
        OUT[i] = IN[j] * W[k] + ... satisfies (j - k) = i. Thus k = j - i, i = j - k, and j = i + k.
        Remember that k ranges from 0 to (WIDTH - 1)

        dE/dIN[j] = sum over all applicable: gradients[i] * W[k]
        dE/dW[k] = sum over all applicable: gradients[i] * IN[j] <-- That's a convolution! (the j's, i's that are influenced/influence the W[k] are a sliding window.)

        Of course if we flip the kernel in a convolution then the corresponding indices are also flipped:

        dE/dW[k'] = sum gradients[i] * in[j] where k' = WIDTH - 1 - k...

        */

        for (int i = 0; i < N_OUTPUTS; i++) {
            for (int k = 0; k < WIDTH; k++) {
                int j = i + k;
                if (j < 0 || j >= N_INPUTS) continue;
                vgrads[j] += gradients[i] * filter[WIDTH - 1 - k];
                wgrads[WIDTH - 1 - k] += gradients[i] * mri[j];
            }
        }

        // Propagate the gradients

        for (int k = 0; k < WIDTH; k++) filter[k] -= wgrads[k] * LR;


        return vgrads;
    }
};

#endif
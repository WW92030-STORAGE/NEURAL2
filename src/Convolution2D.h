#ifndef CONV2D_NEURAL2_H
#define CONV2D_NEURAL2_H

#include "NN_IMPORTANT.h"
#include "Layer.h"

#include <vector>
#include <string>
#include <array>

/*

2-dimensional convolutional layer. I trust that you don't need additional striding or padding for now.
UNFINISHED!!!!!!!!

*/

// Padding mode
enum Conv2dPaddingMode {
    PAD_ZERO = 0
};

class Convolution2D : public Layer {
    public:

    int filter_dims[4] = {1, 1, 1, 1};

    Tensor4 filters;
    Tensor3 mro;
    Tensor3 mri;

    void init() {
        // filters[i][o] is how much input i influences output o
        filters = std::vector<std::vector<std::vector<std::vector<NN_NUMERIC_T>>>>(filter_dims[0], std::vector<std::vector<std::vector<NN_NUMERIC_T>>>(filter_dims[1], std::vector<std::vector<NN_NUMERIC_T>>(filter_dims[2], std::vector<NN_NUMERIC_T>(filter_dims[3], randradius()))));
    }

    Convolution2D() : Layer() {
        init();
    }

    
    Convolution2D(int KR, int KC, int KI = 1, int KO = 1) : Layer(KI, KO) {
        filter_dims[0] = KI;
        filter_dims[1] = KO;
        filter_dims[2] = KR;
        filter_dims[3] = KC;
        init();
    }

    Convolution2D(const Convolution2D& other) : Layer(other) {
        for (int i = 0; i < 4; i++) filter_dims[i] = other.filter_dims[i];
        // do stuff here
        init();
    }

    ~Convolution2D() {
    }
    
    virtual std::string to_string() {
        // return "Layer[" + std::to_string(N_INPUTS) + ", " + std::to_string(N_OUTPUTS) + "]";
        
        std::string res = Layer::to_string() + ": NNLayer[]";
        return res;
    }

    // Forward propagation

    Tensor3 forward(Tensor3 input) {
        int R = input[0].size() - filter_dims[2] + 1;
        int C = input[0][0].size() - filter_dims[3] + 1;
        int NO = filter_dims[1];
        int NI = filter_dims[0];
        int KR = filter_dims[2];
        int KC = filter_dims[3];
        Tensor3 output = Tensor3D(NO, R, C, false);

        // do stuff here

        for (int ro = 0; ro < R; ro++) { // row in the output stack (2nd dimension)
            for (int co = 0; co < C; co++) { // column in the output stack (3rd dimension)
                for (int no = 0; no < NO; no++) { // number of the output (1st dimension output, 2nd dimension kernel)
                    for (int ni = 0; ni < NI; ni++) { // number of the input (1st dimension kernel)
                        for (int kr = 0; kr < KR; kr++) {
                            for (int kc = 0; kc < KC; kc++) output[no][ro][co] += filters[ni][no][kr][kc] * input[ni][kr + ro][kc + co];
                        }
                    }
                }
            }
        }

        // Update

        mri = input;
        mro = output;
        return output;
    }

    Tensor3 operator()(Tensor3 input) {
        return forward(input);
    }

    // backprop using the most recent forward pass. returns gradients with respect to inputs.
    Tensor3 backward(Tensor3 gradients) {
        int NI = filter_dims[0];
        int NO = filter_dims[1];
        int KR = filter_dims[2];
        int KC = filter_dims[3];

        Tensor3 vgrads = Tensor3D(NI, mri[0].size(), mri[0][0].size());

        int R = gradients[0].size();
        int C = gradients[0][0].size();

        /*

        (Article written by Mersenne)
        
        Oh boy. Here we go again. Once again this is figuring out where each portion of the output gradient stems from.
        Using the most recent inputs and outputs (mri, mro) --> (input, output):

        Each output output[no][r][c] is the sum: input[ni][r + kr][c + kc] * filters[ni][no][kr][kc], on the appropriate values of (ni, kr, kc)
        Therefore each gradient[no][r][c] contributes filters[ni][no][kr][kc] (times the gradient itself) to the dE/dinput[ni][r + kr][c + kc], over the same appropriate (ni, kr, kc)
        
        But what about the weights? the contribute to dE/dfilters[ni][no][kr][kc] is add input[ni][r + kr][c + kc] * gradients[no][r][c]

        */

        Tensor4 wgrads = Tensor4D(NI, NO, KR, KC);

        for (int no = 0; no < NO; no++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    for (int ni = 0; ni < NI; ni++) {
                        for (int kr = 0; kr < KR; kr++) {
                            for (int kc = 0; kc < KC; kc++) {
                                wgrads[ni][no][kr][kc] += mri[ni][r + kr][c + kc] * gradients[no][r][c];
                                vgrads[ni][r + kr][c + kc] += filters[ni][no][kr][kc] * gradients[no][r][c];
                            }
                        }
                    }
                }
            }
        }

        for (int ni = 0; ni < NI; ni++) {
            for (int no = 0; no < NO; no++) {
                for (int kr = 0; kr < KR; kr++) {
                    for (int kc = 0; kc < KC; kc++) filters[ni][no][kr][kc] -= LR * wgrads[ni][no][kr][kc];
                }
            }
        }

        return vgrads;
    }
};

class ReLU3D : public Layer {
    public:
    int NI = 1;
    int R = 1;
    int C = 1;
    NN_NUMERIC_T LEAK = 0.01;
    Tensor3 mri;
    Tensor3 mro;

    ReLU3D() : Layer() {

    }

    ReLU3D(int ni, int r, int c) : Layer(ni * r * c, ni * r * c) {
        NI = ni;
        R = r;
        C = c;
    }

    Tensor3 forward(Tensor3 tensor) {
        Tensor3 res = Tensor3D(NI, R, C);
        for (int ni = 0; ni < NI; ni++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) res[ni][r][c] = (tensor[ni][r][c] > 0 ? 1 : LEAK) * tensor[ni][r][c];
            }
        }
        mri = tensor;
        mro = res;
        return res;
    }
    Tensor3 operator()(Tensor3 input) {
        return forward(input);
    }

    Tensor3 backward(Tensor3 gradients) {
        Tensor3 res = Tensor3D(NI, R, C);
        for (int ni = 0; ni < NI; ni++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) res[ni][r][c] = (mro[ni][r][c] > 0 ? 1 : LEAK) * gradients[ni][r][c];
            }
        }
        return res;
    }
};

class Sigmoid3D : public Layer {
    public:
    int NI = 1;
    int R = 1;
    int C = 1;
    NN_NUMERIC_T LEAK = 0.01;
    Tensor3 mri;
    Tensor3 mro;

    Sigmoid3D() : Layer() {

    }

    Sigmoid3D(int ni, int r, int c) : Layer(ni * r * c, ni * r * c) {
        NI = ni;
        R = r;
        C = c;
    }

    Tensor3 forward(Tensor3 tensor) {
        Tensor3 res = Tensor3D(NI, R, C);
        for (int ni = 0; ni < NI; ni++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) res[ni][r][c] = sigmoid(tensor[ni][r][c]);
            }
        }
        mri = tensor;
        mro = res;
        return res;
    }
    Tensor3 operator()(Tensor3 input) {
        return forward(input);
    }

    Tensor3 backward(Tensor3 gradients) {
        Tensor3 res = Tensor3D(NI, R, C);
        for (int ni = 0; ni < NI; ni++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) res[ni][r][c] = sigd(mri[ni][r][c], mro[ni][r][c]) * gradients[ni][r][c];
            }
        }
        return res;
    }
};

class Flatten3D : public Layer {
    public:

    int NI = 1;
    int R = 1;
    int C = 1;
    
    Flatten3D() : Layer() {

    }

    Flatten3D(int ni, int r, int c) : Layer(ni * r * c, ni * r * c) {
        NI = ni;
        R = r;
        C = c;
    }

    std::vector<NN_NUMERIC_T> forward(Tensor3 tensor) {
        std::vector<NN_NUMERIC_T> res(NI * R * C);

        int counter = 0;
        for (int ni = 0; ni < NI; ni++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) res[counter++] = tensor[ni][r][c];
            }
        }
        return res;
    }
    std::vector<NN_NUMERIC_T> operator()(Tensor3 input) {
        return forward(input);
    }

    Tensor3 backwards(std::vector<NN_NUMERIC_T> gradients) {
        Tensor3 res = Tensor3D(NI, R, C);
        int counter = 0;
        for (int ni = 0; ni < NI; ni++) {
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) res[ni][r][c] = gradients[counter++];
            }
        }
        return res;
    }
};

#endif
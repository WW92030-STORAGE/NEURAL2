#include <bits/stdc++.h>


#include "Layer.h"
#include "NN.h"
#include "NN_IMPORTANT.h"
#include "LayerStack.h"

using namespace std;


// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
void mazurr() {
    LinearLayer L1(2, 2);
    SigmoidLayer S1(2);
    LinearLayer L2(2, 2);
    SigmoidLayer S2(2);

    L1.weights[0][0] = 0.15;
    L1.weights[1][0] = 0.2;
    L1.weights[0][1] = 0.25;
    L1.weights[1][1] = 0.3;
    L1.weights[2][0] = 0.35;
    L1.weights[2][1] = 0.35;


    L2.weights[0][0] = 0.4;
    L2.weights[1][0] = 0.45;
    L2.weights[0][1] = 0.5;
    L2.weights[1][1] = 0.55;
    L2.weights[2][0] = 0.6;
    L2.weights[2][1] = 0.6;

    /*
    L1.verbose = true;
    L2.verbose = true;
    S1.verbose = true;
    S2.verbose = true;
    */

    for (int iter = 0; iter < 100000; iter++) {


    std::vector<NN_NUMERIC_T> input({0.05, 0.1});

    auto output = S2(L2(S1(L1(input))));
    std::cout << toString(output) << "\n";

    std::vector<NN_NUMERIC_T> expected({0.01, 0.99});

    auto err = sub(output, expected);

    L1.backward(S1.backward(L2.backward(S2.backward(err))));

    }
    
}

// XOR gate: sum of the two numbers is > 0.5 and < 1.5 or something
void simpletest() {
    int N_TRAIN = 100000;

    LinearLayer L(2, 3);
    SigmoidLayer S(3);
    LinearLayer L2(3, 2);
    SigmoidLayer S2(2);

    int confusion[2][2] = {{0, 0}, {0 ,0}};

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        std::vector<NN_NUMERIC_T> input = randivec(2, 2);


        auto expected = std::vector<NN_NUMERIC_T>({(input[0] + input[1] == 1) ? 0.99 : 0.01});
        expected.push_back(1.0 - expected[0]);
        auto actual = S2(L2(S(L(input))));

        std::cout << "ITER " << TRAIN << " ";
        std::cout << "INPUTS " << toString(input) << " ";
        std::cout << "EXPECTED " << toString(expected) << " ";
        std::cout << "ACTUAL " << toString(actual) + "\n";
        // std::cout << "WEIGHTS " << L.to_string() << "\n";

        auto err = sub(actual, expected);
        L.backward(S.backward(L2.backward(S2.backward(err))));

        confusion[actual[0] > actual[1]][expected[0] > expected[1]]++;
    }

    std::cout << "CONFUSION (Rows actual, columns expected)\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << confusion[i][j] << " ";
        std::cout << std::endl;
    }

    std::cout << L.to_string() << "\n";
    std::cout << L2.to_string() << "\n";
}

// Is the 3-dimensional point at most 1 away from the origin but more than 0.5? (Points are chosen in a box roughly of half-side length 1.01549129756 = 0.5 * cuberoot of 4pi/3 * 2)
void spheretest() {
    int N_TRAIN = 100000;
    int THRESHOLD = 0.9 * N_TRAIN;

    LinearLayer L(3, 4);
    SigmoidLayer S(4);
    LinearLayer L2(4, 2);
    SigmoidLayer S2(2);

    int confusion[2][2] = {{0, 0}, {0 ,0}};

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        std::vector<NN_NUMERIC_T> input = randradvec(3, 1.01549129756);

        NN_NUMERIC_T distsq = input[0] * input[0] + input[1] * input[1] + input[2] * input[2];

        auto expected = std::vector<NN_NUMERIC_T>({(distsq <= 1) ? 0.99 : 0.01});
        expected.push_back(1.0 - expected[0]);
        auto actual = S2(L2(S(L(input))));

        std::cout << "ITER " << TRAIN << " ";
        std::cout << "INPUTS " << toString(input) << " ";
        std::cout << "EXPECTED " << toString(expected) << " ";
        std::cout << "ACTUAL " << toString(actual) + "\n";
        // std::cout << "WEIGHTS " << L.to_string() << "\n";

        auto err = sub(actual, expected);
        L.backward(S.backward(L2.backward(S2.backward(err))));

        if (TRAIN >= THRESHOLD) confusion[actual[0] > actual[1]][expected[0] > expected[1]]++;
    }

    std::cout << "CONFUSION (Rows actual, columns expected)\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << confusion[i][j] << " ";
        std::cout << std::endl;
    }
}

// Same test except with two hidden layers
void spheretest3() {
    int N_TRAIN = 100000;
    int THRESHOLD = 0.9 * N_TRAIN;

    LinearLayer L(3, 4);
    SigmoidLayer S(4);
    LinearLayer L2(4, 4);
    SigmoidLayer S2(4);
    LinearLayer L3(4, 2);
    SigmoidLayer S3(2);

    int confusion[2][2] = {{0, 0}, {0 ,0}};

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        std::vector<NN_NUMERIC_T> input = randradvec(3, 1.01549129756);

        NN_NUMERIC_T distsq = input[0] * input[0] + input[1] * input[1] + input[2] * input[2];

        auto expected = std::vector<NN_NUMERIC_T>({(distsq <= 1) ? 0.99 : 0.01});
        expected.push_back(1.0 - expected[0]);
        auto actual = S3(L3(S2(L2(S(L(input))))));

        std::cout << "ITER " << TRAIN << " ";
        std::cout << "INPUTS " << toString(input) << " ";
        std::cout << "EXPECTED " << toString(expected) << " ";
        std::cout << "ACTUAL " << toString(actual) + "\n";
        // std::cout << "WEIGHTS " << L.to_string() << "\n";

        auto err = sub(actual, expected);
        L.backward(S.backward(L2.backward(S2.backward(L3.backward(S3.backward(err))))));

        if (TRAIN >= THRESHOLD) confusion[actual[0] > actual[1]][expected[0] > expected[1]]++;
    }

    std::cout << "CONFUSION (Rows actual, columns expected)\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << confusion[i][j] << " ";
        std::cout << std::endl;
    }
}

// Same test except with a stack instead of individual layers
void stacktest() {
    int N_TRAIN = 100000;
    int THRESHOLD = 0.9 * N_TRAIN;
    int confusion[2][2] = {{0, 0}, {0 ,0}};

    // Dimensions
    int N_IN = 3;
    int N_OUT = 1;

    // Declare layers here
    LinearLayer L(3, 4);
    SigmoidLayer S(4);
    LinearLayer L2(4, 4);
    SigmoidLayer S2(4);
    LinearLayer L3(4, 2);
    SigmoidLayer S3(2);
    LayerStack stack(std::vector<Layer*>({&L, &S, &L2, &S2, &L3, &S3}));

    // Training loop on random data
    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        // Declare the inputs
        std::vector<NN_NUMERIC_T> input = randradvec(N_IN, 1.01549129756);

        // Calculate the expected outputs
        NN_NUMERIC_T distsq = input[0] * input[0] + input[1] * input[1] + input[2] * input[2];
        auto expected = std::vector<NN_NUMERIC_T>({(distsq <= 1) ? 0.99 : 0.01});
        expected.push_back(1.0 - expected[0]);

        // Foward propagate the model
        auto actual = stack(input);

        std::cout << "ITER " << TRAIN << " ";
        std::cout << "INPUTS " << toString(input) << " ";
        std::cout << "EXPECTED " << toString(expected) << " ";
        std::cout << "ACTUAL " << toString(actual) + "\n";

        // Backward propagaate the model
        auto err = sub(actual, expected);
        stack.backward(err);

        if (TRAIN >= THRESHOLD) confusion[actual[0] > actual[1]][expected[0] > expected[1]]++;
    }

    std::cout << "CONFUSION (Rows actual, columns expected)\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << confusion[i][j] << " ";
        std::cout << std::endl;
    }
}

// Detect the presence of any run of 1s in the array using a standard neural network
void runstest() {
    int N_TRAIN = 200000;
    int THRESHOLD = 0.9 * N_TRAIN;
    int confusion[2][2] = {{0, 0}, {0 ,0}};

    int N_IN = 16;
    int N_H = 12;
    int N_OUT = 2;

    int CONS = 3;

    // Declare layers here
    LinearLayer L1(N_IN, N_H);
    SigmoidLayer S1(N_H);
    LinearLayer L2(N_H, N_OUT);
    SigmoidLayer S2(N_OUT);

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        // Declare the inputs
        std::vector<NN_NUMERIC_T> input(N_IN, 0);
        for (int i = 0; i < N_IN; i++) input[i] = (randi(2) == 0) ? 1 : 0;

        // calculate the expected
        std::vector<NN_NUMERIC_T> expected(2, 0.01);
        bool hasrun = false;
        for (int i = 0; i < N_IN + 1 - CONS; i++) {
            bool x = true;
            for (int j = 0; j < CONS; j++) {
                if (input[j + i] == 0) x = false; 
            }
            hasrun |= x;
        }

        if (hasrun) expected[0] = 0.99;
        else expected[1] = 0.99;

        // Foward propagate the model
        auto actual = S2(L2(S1(L1(input))));

        std::cout << "ITER " << TRAIN << " ";
        std::cout << "EXPECTED " << toString(expected) << " ";
        std::cout << "ACTUAL " << toString(actual) + "\n";

        // Backward propagaate the model
        auto err = sub(actual, expected);
        L1.backward(S1.backward(L2.backward(S2.backward(err))));

        if (TRAIN >= THRESHOLD) {
            confusion[actual[0] > actual[1]][expected[0] > expected[1]]++;
        }
    }

    std::cout << "CONFUSION (Rows actual, columns expected)\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << confusion[i][j] << " ";
        std::cout << std::endl;
    }
}

// Convolution 1D. This model will detect the presence of any run of 1s in the array. (Same test as the last one except using a CNN)
void conv1d() {
    int N_TRAIN = 200000;
    int THRESHOLD = 0.9 * N_TRAIN;
    int confusion[2][2] = {{0, 0}, {0 ,0}};

    int N_IN = 16;
    int N_OUT = 13;
    int WIDTH = N_IN - N_OUT + 1;

    int CONS = 4;

    // Declare layers here
    Convolution1D conv(N_IN, N_OUT);
    SigmoidLayer T(N_OUT);
    LinearLayer L(N_OUT, 2);
    SigmoidLayer S(2);

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        // Declare the inputs
        std::vector<NN_NUMERIC_T> input(N_IN, 0);
        for (int i = 0; i < N_IN; i++) input[i] = (randi(2) == 0) ? 1 : 0;

        // calculate the expected
        std::vector<NN_NUMERIC_T> expected(2, 0.01);
        bool hasrun = false;
        for (int i = 0; i < N_IN + 1 - CONS; i++) {
            bool x = true;
            for (int j = 0; j < CONS; j++) {
                if (input[j + i] == 0) x = false; 
            }
            hasrun |= x;
        }

        if (hasrun) expected[0] = 0.99;
        else expected[1] = 0.99;

        // Foward propagate the model
        auto actual = S(L(T(conv(input))));

        std::cout << "ITER " << TRAIN << " ";
        std::cout << "XXX " << toString(conv.filter, conv.WIDTH) << " ";
        std::cout << "EXPECTED " << toString(expected) << " ";
        std::cout << "ACTUAL " << toString(actual) + "\n";

        // Backward propagaate the model
        auto err = sub(actual, expected);
        conv.backward(T.backward(L.backward(S.backward(err))));

        if (TRAIN >= THRESHOLD) {
            confusion[actual[0] > actual[1]][expected[0] > expected[1]]++;
        }
    }

    std::cout << "CONFUSION (Rows actual, columns expected)\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << confusion[i][j] << " ";
        std::cout << std::endl;
    }
}

int main() {
    srand(4);
    runstest();
    return 0;
}
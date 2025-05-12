#include <bits/stdc++.h>


#include "Layer.h"
#include "NN.h"
#include "NN_IMPORTANT.h"



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
}

// Is the 3-dimensional point at most 1 away from the origin but more than 0.5? (Points are chosen in a box roughly of half-side length 1.612 = cuberoot of 4pi/3)
void spheretest() {
    int N_TRAIN = 1000000;
    int THRESHOLD = 0.9 * N_TRAIN;

    LinearLayer L(3, 4);
    SigmoidLayer S(4);
    LinearLayer L2(4, 2);
    SigmoidLayer S2(2);

    int confusion[2][2] = {{0, 0}, {0 ,0}};

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        std::vector<NN_NUMERIC_T> input = randradvec(3, 1.612);

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
    int N_TRAIN = 1000000;
    int THRESHOLD = 0.9 * N_TRAIN;

    LinearLayer L(3, 4);
    SigmoidLayer S(4);
    LinearLayer L2(4, 4);
    SigmoidLayer S2(4);
    LinearLayer L3(4, 2);
    SigmoidLayer S3(2);

    int confusion[2][2] = {{0, 0}, {0 ,0}};

    for (int TRAIN = 0; TRAIN < N_TRAIN; TRAIN++) {
        std::vector<NN_NUMERIC_T> input = randradvec(3, 1.612);

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

int main() {
    srand(4);
    spheretest();
    return 0;
}
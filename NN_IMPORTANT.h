#ifndef NN_IMPORTANT_NEURAL2
#define NN_IMPORTANT_NEURAL2

#include <climits>
#include <cstdint>
#include <vector>
#include <string>

#define NN_NUMERIC_T double

// Random

inline NN_NUMERIC_T rand01() {
    return (NN_NUMERIC_T)((double)(rand()) / RAND_MAX);
}

inline NN_NUMERIC_T randradius(NN_NUMERIC_T R = 1.0) {
    return (1.0 - 2.0 * rand01()) * R;
}

inline NN_NUMERIC_T deriv(NN_NUMERIC_T (*f)(NN_NUMERIC_T), NN_NUMERIC_T x) {
    const NN_NUMERIC_T d = 0.0001;
    return (f(x + d) - f(x - d)) / (d + d);
}

inline NN_NUMERIC_T sigmoid(NN_NUMERIC_T x) {
    return 1.0 / (1.0 + exp(-x));
}

inline NN_NUMERIC_T sigd(NN_NUMERIC_T x, NN_NUMERIC_T y) {
    return y * (1 - y);
}

int randi(int x) {
    return (int)(rand01() * x);
}

std::vector<NN_NUMERIC_T> randvec(int n, NN_NUMERIC_T range = 1) {
    std::vector<NN_NUMERIC_T> thing(n);
    for (int i = 0; i < n; i++) thing[i] = rand01() * range;
    return thing;
}


std::vector<NN_NUMERIC_T> randradvec(int n, NN_NUMERIC_T range = 1) {
    std::vector<NN_NUMERIC_T> thing(n);
    for (int i = 0; i < n; i++) thing[i] = randradius() * range;
    return thing;
}

std::vector<NN_NUMERIC_T> randivec(int n, NN_NUMERIC_T range = 2) {
    std::vector<NN_NUMERIC_T> thing(n);
    for (int i = 0; i < n; i++) thing[i] = randi(range);
    return thing;
}

// Vector elementwise difference

inline std::vector<NN_NUMERIC_T> sub(std::vector<NN_NUMERIC_T> a, std::vector<NN_NUMERIC_T> b) {
    std::vector<NN_NUMERIC_T> res;
    for (int i = 0; i < a.size() && i < b.size(); i++) res.push_back(a[i] - b[i]);
    return res;
}

// Printouts

inline std::string toString(std::vector<NN_NUMERIC_T> input) {
    std::string res = "[";
    for (int i = 0; i < input.size(); i++) {
        if (i) res = res + ", ";
        res = res + std::to_string(input[i]);
    }
    return res + "]";
}

inline std::string toString(NN_NUMERIC_T* input, int size) {
    std::string res = "[";
    for (int i = 0; i < size; i++) {
        if (i) res = res + ", ";
        res = res + std::to_string(input[i]);
    }
    return res + "]";
}

#endif
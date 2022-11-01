#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace py = pybind11;

float *matmul(const float *a, const float *b, size_t x, size_t y, size_t z) {
    float *c = new float[x * z];
    for (size_t i = 0; i < x; i++)
        for (size_t k = 0; k < z; k++) {
            float s = 0;
            for (size_t j = 0; j < y; j++) s += a[i * y + j] * b[j * z + k];
            c[i * z + k] = s;
        }
    return c;
}

float *softmax(const float *a, size_t x, size_t y) {
    float *c = new float[x * y];
    for (size_t i = 0; i < x; i++) {
        float sum = 0;
        for (size_t j = 0; j < y; j++) sum += c[i * y + j] = exp(a[i * y + j]);
        for (size_t j = 0; j < y; j++) c[i * y + j] /= sum;
    }
    return c;
}

float *eye(const unsigned char *a, size_t x, size_t y) {
    float *c = new float[x * y];
    memset(c, 0, sizeof(float) * x * y);
    for (size_t i = 0; i < x; i++) c[i * y + a[i]] = 1;
    return c;
}

float *transpose(const float *a, size_t x, size_t y) {
    float *c = new float[x * y];
    for (size_t i = 0; i < x; i++)
        for (size_t j = 0; j < y; j++) c[j * x + i] = a[i * y + j];
    return c;
}

float *matsub(float *a, const float *b, size_t x, size_t y) {
    for (size_t i = 0; i < x; i++)
        for (size_t j = 0; j < y; j++) a[i * y + j] -= b[i * y + j];
    return a;
}

float *matscale(float *a, size_t x, size_t y, float v) {
    for (size_t i = 0; i < x; i++)
        for (size_t j = 0; j < y; j++) a[i * y + j] *= v;
    return a;
}

void softmax_regression_epoch_cpp(const float         *X,
                                  const unsigned char *y,
                                  float               *theta,
                                  size_t               m,
                                  size_t               n,
                                  size_t               k,
                                  float                lr,
                                  size_t               batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    for (size_t i = 0; i < m; i += batch) {
        float *Z = softmax(matmul(X + i * n, theta, batch, n, k), batch, k);
        float *I = eye(y + i, batch, k);
        float *G = matmul(transpose(X + i * n, batch, n),
                          matsub(Z, I, batch, k),
                          n,
                          batch,
                          k);
        matsub(theta, matscale(G, n, k, lr / batch), n, k);
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style>         X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style>         theta,
           float                                          lr,
           int                                            batch) {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("theta"),
        py::arg("lr"),
        py::arg("batch"));
}

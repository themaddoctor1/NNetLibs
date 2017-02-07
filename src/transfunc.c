#include "neuralnet.h"

#include <math.h>

Matrix unitStepTransfer(Matrix m) {
    Matrix u = makeMatrix(m->ROWS, 1);

    int r = m->ROWS;
    while (r--) {
        double d = getMtrxVal(m, r, 0);
        setMtrxVal(u, r, 0, d >= 0 ? 1 : 0);
    }

    return u;
}

Matrix competeTransfer(Matrix m) {
    //A small, negative constant.

    Matrix y = cloneMatrix(m);
    int max = m->ROWS - 1;
    double maxVal = getMtrxVal(m, max, 0);

    int i = max;
    while (i--) {
        double a = getMtrxVal(y, i, 0);
        if (a >= maxVal) {
            setMtrxVal(y, max, 0, 0);
            max = i;
            maxVal = a;
        } else {
            setMtrxVal(y, i, 0, 0);
        }
    }

    setMtrxVal(y, max, 0, 0);
    
    return y;
}

Matrix zeroMatrix(Matrix m) {
    return makeMatrix(m->ROWS, m->ROWS);
}

Matrix linearTransfer(Matrix m) {
    return mulMtrxC(m, 1);
}

Matrix linearTransferGradient(Matrix m) {
    Matrix g = makeMatrix(m->ROWS, m->ROWS);
    int r = m->ROWS;
    while (r--) {
        setMtrxVal(g, r, r, 1);
    }

    return g;
}

Matrix sigmoidTransfer(Matrix m) {
    Matrix sig = makeMatrix(m->ROWS, m->COLS);
    int r = 0;
    while(r < m->ROWS) {
        int c = 0;
        while(c < m->COLS) {
            double d = getMtrxVal(m, r, c);
            setMtrxVal(sig, r, c, 1.0 / (1 + exp(-d)));
            c++;
        }
        r++;
    }

    return sig;
}

Matrix sigmoidTransferGradient(Matrix m) {
    Matrix g = makeMatrix(m->ROWS, m->ROWS);
    int r = 0;
    while(r < m->ROWS) {
        double d = getMtrxVal(m, r, r);
        d = 1.0 / (1 + exp(-d));
        setMtrxVal(g, r, 0, d * (1 - d));
        r++;
    }

    return g;
}


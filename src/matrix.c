#include "matrix.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

Matrix makeMatrix(int r, int c) {
    Matrix m = (Matrix) malloc(sizeof(struct matrix));

    m->ROWS = r;
    m->COLS = c;

    m->vals = (double*) malloc(r * c * sizeof(double));

    int i = r * c;
    while(i--)
        m->vals[i] = 0;

    return m;

}

Matrix cloneMatrix(Matrix A) {
    Matrix m = (Matrix) makeMatrix(A->ROWS, A->COLS);
    int i = A->ROWS * A->COLS;
    while(i--)
        m->vals[i] = A->vals[i];
    
    return m;
    
}

Matrix identityMatrix(int n) {
    Matrix I = makeMatrix(n, n);
    int i = n * (n + 1);
    while(i) {
        i -= n+1;
        printf("%i\n", i);
        I->vals[i] = 1;
    }
    return I;
}

void freeMatrix(Matrix m) {
    int i = m->ROWS * m->COLS;
    while(i--)
            m->vals[i] = 0;
    free(m->vals);
    m->vals = NULL;

    m->ROWS = 0;
    m->COLS = 0;
    free(m);

}

Matrix addMtrx(Matrix a, Matrix b) {
    Matrix m = (Matrix) malloc(sizeof(struct matrix));

    m->ROWS = a->ROWS;
    m->COLS = a->COLS;

    m->vals = (double*) malloc(m->ROWS * m->COLS * sizeof(double*));

    int i = m->ROWS * m->COLS;
    while(i--)
        m->vals[i] = a->vals[i] + b->vals[i];

    return m;
}

Matrix subMtrx(Matrix a, Matrix b) {
    Matrix m = (Matrix) malloc(sizeof(struct matrix));
    
    m->ROWS = a->ROWS;
    m->COLS = a->COLS;

    m->vals = (double*) malloc(m->ROWS * m->COLS * sizeof(double*));

    int i = m->ROWS * m->COLS;
    while(i--)
        m->vals[i] = a->vals[i] - b->vals[i];

    return m;
}

Matrix mulMtrxC(Matrix a, double d) {
    Matrix m = (Matrix) malloc(sizeof(struct matrix));

    m->ROWS = a->ROWS;
    m->COLS = a->COLS;

    m->vals = (double*) malloc(m->ROWS * m->COLS * sizeof(double*));

    int i = m->ROWS * m->COLS;
    while(i--) {
        m->vals[i] = a->vals[i] * d;
    }

    return m;
}

Matrix mulMtrxM(Matrix a, Matrix b) {
    if(!a || !b) //Error check
        return NULL;
    
    if (a->COLS != b->ROWS) {
        printf("Dangerous mult. btwn %i x %i and %i by %i matrices.\n", a->ROWS, a->COLS, b->ROWS, b->COLS);
    }

    Matrix m = (Matrix) malloc(sizeof(struct matrix));

    m->ROWS = a->ROWS;
    m->COLS = b->COLS;

    m->vals = (double*) malloc(m->ROWS * m->COLS * sizeof(double*));

    int i = m->ROWS;
    while(i--) {
        int j = m->COLS;
        while(j--) {
            double d = 0;
            int k = a->COLS;
            while(k--)
                d += getMtrxVal(a, i, k) * getMtrxVal(b, k, j);
            setMtrxVal(m, i, j, d);
        }
    }

    return m;
}

Matrix hadamardProduct(Matrix a, Matrix b) {
    Matrix m = (Matrix) malloc(sizeof(struct matrix));
    
    m->ROWS = a->ROWS;
    m->COLS = a->COLS;

    m->vals = (double*) malloc(m->ROWS * m->COLS * sizeof(double*));

    int i = m->ROWS * m->COLS;
    while(i--)
        m->vals[i] = a->vals[i] * b->vals[i];

    return m;
}

Matrix transpose(Matrix a) {
    Matrix m = (Matrix) malloc(sizeof(struct matrix));

    m->ROWS = a->COLS;
    m->COLS = a->ROWS;

    m->vals = (double*) malloc(m->ROWS * m->COLS * sizeof(double*));
    
    int i = m->ROWS;
    while(i--) {
        int j = m->COLS;
        while(j--)
            setMtrxVal(m, i, j, getMtrxVal(a, j, i));
    }

    return m;
   
}

int gaussian(Matrix A) {
    int rank = rowEchelon(A) - 1;
    int i = rank;
    while(i > 0) {
        int j = 0;
        while(A->vals[i * A->COLS + j] == 0) j++;
        
        int k = i;
        while(k--) {
            Matrix row = getRowVector(A, i);
            mulMtrxRow(row, 0, -getMtrxVal(A, k, j));
            addMtrxRow(A, k, row);
            freeMatrix(row);
        }

        i--;

    }
    return rank;
}

int rowEchelon(Matrix A) {
    int pivots = 0;
    
    int j = 0;
    while(j < A->COLS && pivots < A->ROWS) {
        //Attempt to reduce column j
        
        int i = pivots;
        if(getMtrxVal(A, i, j) != 0) {
            mulMtrxRow(A, i, 1.0 / getMtrxVal(A, i, j));
            int k = i + 1;
            while(k < A->ROWS) {
                Matrix row = getRowVector(A, i);
                mulMtrxRow(row, 0, -getMtrxVal(A, k, j));
                addMtrxRow(A, k, row);
                freeMatrix(row);
                k++;
            }
            pivots++;
            j++;
        } else {

            while(i < A->ROWS) {
                if(getMtrxVal(A, i, j) != 0) {
                    swapMtrxRows(A, i, pivots);
                    break;
                }
                i++;
            }

            if(i >= A->ROWS) {
                j++;
            }
        }
    }

    return pivots;
}

double vecNorm(Matrix m) {
    double d = 0;
    int i = 0;
    while(i < m->ROWS) {
        int j = 0;
        while(j < m->COLS) {
            d += getMtrxVal(m, i, j) * getMtrxVal(m, i, j);
            j++;
        }
        i++;
    }

    return sqrt(d);
}

double dotProd(Matrix a, Matrix b) {
    double d = 0;
    int i = a->ROWS;
    while (i--)
        d += a->vals[i] * b->vals[i];
    return d;
}

double getMtrxVal(Matrix m, int r, int c) {
    return m->vals[r * m->COLS + c];
}

void setMtrxVal(Matrix m, int r, int c, double val) {
    m->vals[r * m->COLS + c] = val;
}

Matrix getRowVector(Matrix A, int r) {
    Matrix row = makeMatrix(1, A->COLS);
    int i = A->COLS;
    while(i--) {
        row->vals[i] = A->vals[r * A->COLS + i];
    }
    return row;
}

Matrix getColVector(Matrix A, int c) {
    Matrix col = makeMatrix(A->ROWS, 1);
    int i = A->ROWS;
    while(i--) {
        col->vals[i] = A->vals[i * A->COLS + c];
    }
    return col;
}

void swapMtrxRows(Matrix A, int i, int j) {
    double d;

    int x = 0;
    while(x < A->COLS) {
        d = A->vals[i * A->COLS + x];
        A->vals[i * A->COLS + x] = A->vals[j * A->COLS + x];
        A->vals[j * A->COLS + x] = d;
        x++;
    }

}

void addMtrxRow(Matrix A, int r, Matrix row) {
    int i = A->COLS;
    while(i--) {
        A->vals[r * A->COLS + i] += row->vals[i];
    }
}

void mulMtrxRow(Matrix A, int r, double c) {
    int i = A->COLS;
    while(i--) {
        A->vals[r * A->COLS + i] *= c;
    }
}

void printVector(Matrix v) {
    printf("<");
    int r = 0;
    while (r < v->ROWS) {
        if (r) printf(", ");
        printf("%lf", v->vals[r]);
        r++;
    }
    printf(">\n");
}

void printMatrix(Matrix m) {
    printf("%i by %i matrix:\n", m->ROWS, m->COLS);

    int r = 0;
    while (r < m->ROWS) {
        printf("|");

        int c = 0;
        while (c < m->COLS) {
            if (c) printf(" ");
            if(m->vals[r * m->COLS + c] > 0) printf(" ");
            printf("%lf", m->vals[r * m->COLS + c]);
            c++;
        }
        r++;

        printf("|\n");
    }

}



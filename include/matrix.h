#ifndef _MATRIX_H_
#define _MATRIX_H_

struct matrix {
    int ROWS;
    int COLS;
    double* vals;
};

typedef struct matrix* Matrix;

Matrix makeMatrix(int r, int c);
Matrix cloneMatrix(Matrix A);
void freeMatrix(Matrix m);

Matrix identityMatrix(int n);

Matrix addMtrx(Matrix a, Matrix b);
Matrix subMtrx(Matrix a, Matrix b);
Matrix mulMtrxM(Matrix a, Matrix b);
Matrix mulMtrxC(Matrix m, double d);

Matrix hadamardProduct(Matrix a, Matrix b);

Matrix transpose(Matrix a);
int gaussian(Matrix A);
int rowEchelon(Matrix A);

double vecNorm(Matrix m);
double dotProd(Matrix a, Matrix b);

double getMtrxVal(Matrix m, int r, int c);
void setMtrxVal(Matrix m, int r, int c, double val);

Matrix getRowVector(Matrix A, int r);
Matrix getColVector(Matrix A, int r);
void addMtrxRow(Matrix A, int r, Matrix row);
void mulMtrxRow(Matrix A, int r, double c);
void swapMtrxRows(Matrix A, int i, int j);

void printMatrix(Matrix m);
void printVector(Matrix v);

#endif


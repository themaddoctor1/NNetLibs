#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <math.h>

#include "neuralnet.h"
#include "nettrain.h"
#include "test.h"

#include <time.h>

NetTrainKit kit = NULL;

int conwaySizes[] = {1, 2, 3, 1, 0};

NeuralNet makeConwayNet(NeuralNet filter, int r, int c) {
    
    int depth = getNetDepth(filter);
    int sizes[depth + 3];
    sizes[depth+2] = 0;

    int i = depth;
    while (i--) {
        sizes[i+2] = r * c * getNetWeights(filter, i)->ROWS;
    }
    sizes[1] = r * c * getNetWeights(filter, 0)->COLS;
    sizes[0] = r * c;

    NeuralNet network = makeNeuralNet(sizes);
    
    setLayerFunc(getNetLayer(network, 0), linearTransfer);

    i = 2;
    while (sizes[i]) {
        setLayerFunc(getNetLayer(network, i-1), getLayerFunc(getNetLayer(filter, i-2)));
        i++;
    }

    int x = r;
    while (x--) {
        int y = c;
        while (y--) {
            //Apply input at (x, y)
            int i = x * c + y; //Index of (x,y)
            
            //The first layer adds the neighbors and livelihood for each cell
            printf("Building custom filter for (%i, %i)...\n", x, y);
            int j = -1;
            while (j <= 1) {
                int k = -1;
                while (k <= 1) {
                    int l = (j+x) * c + (k+y); //Index of other cell
                    if (j+x < 0 || j+x >= r || k+y < 0 || k+y >= c) {
                        k++;
                        continue;
                    }

                    if (j != 0 || k != 0)
                        setMtrxVal(getNetWeights(network, 0), 2*i+1, l, 1); //Adding neighbors
                    else
                        setMtrxVal(getNetWeights(network, 0), 2*i, l, 1); //Is cell living
                    k++;
                }
                j++;
            }
            
            printf("Building layer filters...\n");

            j = getNetDepth(filter);
            while (j--) {
                Matrix M = getNetWeights(filter, j);
                int k = M->ROWS;
                while (k--) {
                    int m = M->COLS;
                    while (m--) {
                        setMtrxVal(getNetWeights(network, j+1),
                                    i * M->ROWS + k,
                                    i * M->COLS + m,
                                    getMtrxVal(M, k, m));
                    }
                }
            }

        }
    }

    return network;
}

NeuralNet makeConwayFilter(NeuralNet filter) {
    //To train game of life, we will create a network to compute an n * m grid.
    //The net will take n * m inputs and return n * m outputs

    //Normally, I could train a network with backpropagation, but I'd have to
    //worry about deriving really large matrices for bigger boards.

    //So, I will create a filter that will train a 9-1 network to compute
    //individual cells, and then use that to generate a filter. This will
    //significantly reeduce the training time.

    int *sizes = &conwaySizes[1];
    
    printf("Making network...\n");
    NeuralNet network = filter ? filter : makeNeuralNet(sizes);
    
    srand(time(NULL));
    
    printf("Randomizing network...\n");

    int i = 0;
    while (sizes[i+1]) {
        Matrix M = getNetWeights(network, i);
        int r = M->ROWS;
        while(r--) {
            int c = M->COLS;
            while (c--) {
                double d = ((double)rand()) / ((double) RAND_MAX);
                setMtrxVal(M, r, c, 2 * d - 1);
            }
        }
        i++;
    }
    
    
    if (!kit) {

        printf("Building training kit...\n");
        
        kit = (NetTrainKit) malloc(sizeof(struct nettrainkit));

        kit->functions = (TransFunc*) malloc(3 * sizeof(TransFunc));
        //kit->functions[0] = linearTransfer;
        kit->functions[0] = sigmoidTransfer;
        kit->functions[1] = unitStepTransfer;
        kit->derivatives = (TransFunc*) malloc(3 * sizeof(TransFunc));
        //kit->derivatives[0] = linearTransferGradient;
        kit->derivatives[0] = sigmoidTransferGradient;
        kit->derivatives[1] = linearTransferGradient;
        
        //We also need a learning rate, as well as a maximum number of cycles.
        kit->learnRate = 1.0 / 256;
        kit->momentum = 0.05;
        kit->decay = 0; // Decay rate is not needed.
        kit->maxCycles = 65536;
        
        kit->data = (Matrix**) malloc(19 * sizeof(Matrix*));
        kit->data[18] = NULL;

        i = 18;
        while (i--) {
            int live = i / 9;
            int neighbors = i % 9;
            
            kit->data[i] = (Matrix*) malloc(2 * sizeof(Matrix));

            kit->data[i][0] = makeMatrix(2, 1);
            setMtrxVal(kit->data[i][0], 0, 0, live);
            setMtrxVal(kit->data[i][0], 1, 0, neighbors);

            int res = 0;
            if (live) {
                if (neighbors == 2 || neighbors == 3)
                    res = 1;
            } else if(neighbors == 3)
                res = 1;

            kit->data[i][1] = makeMatrix(1, 1);
            setMtrxVal(kit->data[i][1], 0, 0, res);

        }
    }
    printf("Applying transfer functions...\n");
    setLayerFunc(getNetLayer(network, 0), kit->functions[0]);
    setLayerFunc(getNetLayer(network, 1), kit->functions[1]);
    //setLayerFunc(getNetLayer(network, 2), kit->functions[2]);

    printf("Training network...\n");

    backpropagationTrain(network, kit);
    
    printf("Complete!\n");

    return network;
}

NeuralNet makeConway(NeuralNet f, int r, int c) {
    
    NeuralNet filter = makeConwayFilter(f);

    double error = 0;
    int i = 18;
    while (i--) {
        //printf("%p\n", (void*) kit);

        Matrix x = kit->data[i][0];
        Matrix y = netFunction(filter, x);

        int live = getMtrxVal(x, 0, 0) > 0;
        int neighbors = getMtrxVal(x, 1, 0);
/*
        while (j < 9) {
            if (j != 4 && getMtrxVal(x, j, 0) > 0)
                living++;
            j++;
        }
*/
        printf("Input: {Live: %i, Nghbr: %i}\n", live, neighbors);
        printf("Output: %i\n", getMtrxVal(y, 0, 0) > 0);
        if (live) {
            if ((neighbors == 2 || neighbors == 3) == (getMtrxVal(y, 0, 0) > 0))
                printf("Right\n\n");
            else
                printf("Wrong (expected %i, got %lf)\n\n", (neighbors == 2 || neighbors == 3), getMtrxVal(y, 0, 0));
        } else if ((neighbors == 3) == (getMtrxVal(y, 0, 0) > 0))
            printf("Right\n\n");
        else
            printf("Wrong (expected %i, got %lf)\n\n", (neighbors == 3), getMtrxVal(y, 0, 0));

        Matrix err = subMtrx(y, kit->data[i][1]);
        double dE = vecNorm(err);
        error += dE;
        freeMatrix(y);
        freeMatrix(err);
    }

    //printf("The filter's error is %lf\n", error/* / 512*/);
    
    //Do not allow error.
    if (error < 0.5) {
        /*i = 0;
        while (sizes[i+1]) {
            printMatrix(getNetWeights(network, i));
            printf("\n");
            i++;
        }*/
        
        printf("Creating ConvoNet...\n");
        
        NeuralNet cwNet = makeConwayNet(filter, r, c);

        printf("Made ConvoNet successfully.\n");
        /*
        i = 0;
        while (i < 3) {
            printMatrix(getNetWeights(cwNet, i));
            printf("\n");
            i++;
        }*/

        return cwNet;
        
    } else {
        //Free the filter
        printf("Freeing filter.\n");
        freeNeuralNet(filter);
        printf("Done.\n");
    }

    return NULL;

}

void playConway(NeuralNet cwNet, int cycles, int r, int c, int *board) {
    Matrix grid = makeMatrix(r * c, 1);

    int i = r * c;
    while (i--) {
        printf("%i, %i = %i\n", i/r, i%r, board[i]);
        setMtrxVal(grid, i, 0, board[i]);
    }

    i = 0;
    while (i < r * c) {
        if (getMtrxVal(grid, i, 0) > 0)
            printf("X");
        else
            printf("O");
        if (i % c == c - 1)
            printf("\n");
        i++;
    }
     
    int iter = cycles;

    while (iter) {

        printf("\nGenning result...\n");
        Matrix newGrid = netFunction(cwNet, grid);
        printf("Done.\n");

        i = 0;
        while (i < r * c) {
            if (getMtrxVal(newGrid, i, 0) > 0)
                printf("X");
            else
                printf("O");
            if (i % c == c-1)
                printf("\n");
            i++;
        }

        freeMatrix(grid);
        grid = newGrid;

        iter--;

    }


}


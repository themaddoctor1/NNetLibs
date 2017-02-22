#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <math.h>

#include "neuralnet.h"
#include "nettrain.h"
#include "test.h"

#include <time.h>

void printMove(char *name, int move) {
    printf("%s played ", name);

    switch (move) {
        case 0:
            printf("rock\n");
            return;
        case 1:
            printf("paper\n");
            return;
        case 2:
            printf("scissors\n");
            return;
        default:
            printf("an illegal move\n");
            return;
    }

}

Matrix* rpsPair(int pPrev, int bPrev, int next) {
    Matrix *pair = (Matrix*) malloc(2 * sizeof(Matrix));

    pair[0] = makeMatrix(9, 1);
    setMtrxVal(pair[0], 3 * pPrev + bPrev, 0, 1);

    pair[1] = makeMatrix(3, 1);
    setMtrxVal(pair[1], next, 0, 1);

    return pair;

}

void trainMoveSequence(NeuralNet net, int pPrev, int bPrev, int next) {
    struct nettrainkit kit;

    Matrix *data[2];
    data[0] = rpsPair(pPrev, bPrev, next);
    data[1] = NULL;

    kit.data = &data[0];

    kit.learnRate = 1;
    kit.maxCycles = 1;
    kit.decay = 0.1875;
    
    supervisedHebbRuleTrain(net, &kit);

    //freeMatrix(data[0][0]);
    //freeMatrix(data[0][1]);
    free(data[0]);
}

int chooseMove(NeuralNet net, int pPrev, int bPrev) {
    
    //Make an input vector based on the previous move.
    Matrix x = makeMatrix(9, 1);
    if (pPrev >= 0 && bPrev >= 0) //Activate the state of the prev move.
        setMtrxVal(x, 3 * pPrev + bPrev, 0, 1);
    else {
        //If no data available, 
        int k = 9;
        while (k--)
            setMtrxVal(x, k, 0, 1);
    }

    Matrix y = netFunction(net, x);
    freeMatrix(x);
    
    //Determine which move had the largest activation.
    int m = 0;
    int i = 3;
    while (--i)
        if (getMtrxVal(y, m, 0) < getMtrxVal(y, i, 0)) {
            m = i;
        }
   
    return (m+1) % 3;

}

/*
 Creates a Neural Net that takes 3 inputs and returns 3 outputs.
*/
NeuralNet rpsNet() {
    
    //Neural net input takes a choice among nine possible
    //result states from the previous round. Output is
    //the activation for each move (largest is chosen).
    int sizes[] = {9, 3, 0};

    NeuralNet bot = makeNeuralNet(sizes);

    setLayerFunc(getNetLayer(bot, 0), linearTransfer);

    return bot;

}

int runRound(NeuralNet net, int playerMove, int botMove) {
    
    //Prompt
    printMove("You", playerMove);
    printMove("Bot", botMove);
    
    int winner = 0;

    //Decide who wom.
    if ((playerMove+1) % 3 == botMove) {
        printf("Bot wins\n");
        winner = -1;
    } else if ((botMove+1) % 3 == playerMove) {
        printf("Player wins\n");
        winner = 1;
    } else
        printf("Tie\n");


    return winner;
}

void rockPaperScissors() {

    //Get a Neural Net to play me in RPS
    //rpsBot should be f: R^3 -> R^3
    NeuralNet rpsBot = rpsNet();
    
    srand(time(NULL));

    for (int i = 0; i < 9; i++)
        setMtrxVal(getNetWeights(rpsBot, 0), i/3, i%3, (rand() % 256) / 256.0);

    int pPrev = 0;
    int bPrev = 0;

    int pScore = 0;
    int bScore = 0;

    while (1) {
        int pMove;

        //Choose a player move. 
        printf("\nMake your move (1 for rock, 2 for paper, 3 for scissors):\n");
        scanf("%d", &pMove);
        pMove--;

        //Error check
        if (pMove < 0 || pMove >= 3) {
            printf("That move is not valid.\n");
            continue;
        }
        
        //Choose a bot move.
        int bMove = chooseMove(rpsBot, pPrev, bPrev);
        
        //Decide who the winner is by running the round.
        int winner = runRound(rpsBot, pMove, bMove);

        //Mod score.
        if (winner > 0)
            pScore++;
        else if (winner < 0)
            bScore++;

        //Extra training.
        trainMoveSequence(rpsBot, pPrev, bPrev, pMove);
        
        //UI update.
        printf("Scores:\nPlayer: %i\nBot   : %i\n", pScore, bScore);

        pPrev = pMove;
        bPrev = bMove;
    }

}



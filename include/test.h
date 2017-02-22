#ifndef _TEST_H_
#define _TEST_H_

void backpropXorDemo();
void hebbianXODemo();
void deltaOrGateDemo();
void hebbianBananaDemo();

/**
 * Runs Conway's Game of Life
 * 
 * filter - An optional parameter that allows a custom filter to be applied.
 *          Can be used for custom rules.
 */
NeuralNet makeConway(NeuralNet filter, int r, int c);
void playConway(NeuralNet cwNet, int cycles, int r, int c, int *board);

/**
 * Runs a game of rock paper scissors between a user and a Neural Net.
 */
void rockPaperScissors();

#endif



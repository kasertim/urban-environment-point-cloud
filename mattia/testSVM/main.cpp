#include <iostream>
#include <bitset>
#include "svm.h"

#include "svm-train.h"

int main(int argc, char **argv) {
    SvmTrain train;

    train.nFeatures=13; // n of features for a point
    train.prob.l = 270; // n of elements/points
    train.prob.y = Malloc(double,train.prob.l);
    train.prob.x = Malloc(struct svm_node *,train.prob.l);
    //train.x_space = Malloc(struct svm_node,train.prob.l * (train.nFeatures+1));
    //train.cross_validation=1;
    train.nr_fold = 4; // is how many sets to split your input data

    // Fill the training
    for (int i=0;i<train.prob.l;i++)
    {
      train.prob.y[i] = ( (double)(rand()%2) ); // label
      train.prob.x[i] = Malloc(struct svm_node,train.nFeatures+1);
      
      int j=0;
      for (j=0; j < train.nFeatures;j++)
      {
	train.prob.x[i][j].index = j;
	train.prob.x[i][j].value = ( (double)( rand()%1000 )/1000 );
      }
      train.prob.x[i][j].index = -1; // set last element of a sample
    }
    
    train.read_problem("heart_scale");
    train.execute();
    
    ///////////////////////////////////////////////
    
    SvmPredict pred;
    pred.model = train.model;
//    pred.loadModel("output.model");
    
    pred.input.l = 10000;
    pred.input.x = Malloc(struct svm_node *,pred.input.l);
    pred.input.y =  Malloc(double,pred.input.l);
    
    for (int i=0;i<pred.input.l;i++)
    {
      pred.input.y[i] = ( (double)(rand()%2) ); // label
      pred.input.x[i] = Malloc(struct svm_node,train.nFeatures+1);
      
      int j=0;
      for (j=0; j < train.nFeatures;j++)
      {
	pred.input.x[i][j].index = j;
	pred.input.x[i][j].value = ( (double)( rand()%1000 )/1000 );
      }
      pred.input.x[i][j].index = -1; // set last element of a sample
    }
    pred.loadTest("heart_scale");
    pred.predict();
}

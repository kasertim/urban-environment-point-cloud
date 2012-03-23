#include <iostream>
#include <bitset>
#include "svm.h"

#include "svm-train.h"

int main(int argc, char **argv) {
    SvmTrain train;

    int nFeatures=13; // n of features for a point
    train.prob.l = 270; // n of elements/points
    train.prob.y = Malloc(double,train.prob.l);
    train.prob.x = Malloc(struct svm_node *,train.prob.l);
    train.x_space = Malloc(struct svm_node,train.prob.l * (nFeatures+1));
    //train.cross_validation=1;
    train.nr_fold = 4;

    // Fill the training
    int j=0;
//     for (int i=0;i<train.prob.l;i++)
//     {
//         train.prob.y[i] = ( ( rand()%2 ) ); // label
//         train.prob.x[i] = &train.x_space[j];
//         while (1)
//         {
//             if (j >= i*(nFeatures+1) + nFeatures )
//                 break;
//             //train.prob.x[i] = Malloc(struct svm_node,nFeatures);
//             train.x_space[j].index = j - i*(nFeatures+1);
//             train.x_space[j].value = ( (double)( rand()%1000 )/1000 );
//             ++j;
//         }
//         train.x_space[j++].index = -1;
//     }
    for (int i=0;i<train.prob.l;i++)
    {
      train.prob.y[i] = ( (double)(rand()%11)/10 ); // label
      train.prob.x[i] = Malloc(struct svm_node,nFeatures+1);
      
      int j=0;
      for (j=0; j < nFeatures;j++)
      {
	train.prob.x[i][j].index = j;
	train.prob.x[i][j].value = ( (double)( rand()%1000 )/1000 );
      }
      train.prob.x[i][j].index = -1; // set last element of a sample
    }
    
    
    // initialize gamma parameter
    if (train.param.gamma == 0 && nFeatures > 0)
            train.param.gamma = 1.0/nFeatures;
    
//    train.read_problem("heart_scale");
    
    train.execute();
}

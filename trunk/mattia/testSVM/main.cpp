#include <iostream>
#include "svm.h"

#include "svm-train.h"

int main(int argc, char **argv) {
    SvmTrain train;
    
    int nFeatures=5;
    train.prob.l = 100; // n of elements
    train.prob.y = Malloc(double,train.prob.l);
    train.prob.x = Malloc(struct svm_node *,train.prob.l);
    train.x_space = Malloc(struct svm_node,nFeatures);
  //setParam;
}

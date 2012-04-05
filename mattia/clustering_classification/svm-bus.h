#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <Eigen/Core>

#include "svm.h"
#include <vector>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Realloc(var,type,n) (type *)realloc(var,(n)*sizeof(type))

namespace pcl {

class SvmTrain {

protected:
    void print_null(const char *s) {}

    void exit_input_error(int line_num)
    {
        fprintf(stderr,"Wrong input format at line %d\n", line_num);
        exit(1);
    }
    char* readline(FILE *input);
    void do_cross_validation();

public:
    SvmTrain();

    ~SvmTrain() {
        svm_destroy_param(&param);
    }

    svm_parameter param;	// set by parse_command_line
    svm_problem prob;		// set by read_problem
    //Eigen::matrixXd set;
    svm_model *model;
    svm_node *x_space;

    // Stores the scaling factors
    svm_scaling scaling;
    int cross_validation;
    int nFeatures;
    int nr_fold;
    char model_file_name[1024];

    char *line;
    int max_line_len;


    int execute();

// read in a problem (in svmlight format)

    void read_problem(const char *filename, svm_problem prob);

    /*
     * Save problem in specified file
     */
    void saveProblem(char *filename);

};

class SvmPredict : protected SvmTrain {
public:
    //struct svm_node **input;
    svm_problem input;
    //int max_nr_attr;
    svm_scaling scaling;

    struct svm_model* model;
    bool predict_probability;

    int nFeatures;

    std::vector<double> prediction;

    FILE *output;

    //char *line;
    //int max_line_len;

    SvmPredict () {
        line = NULL;
        max_line_len = 10000;
        predict_probability=0;
        //max_nr_attr = 64;
        //input = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
    }

    void loadModel(const char *filename);

    void read_problem(const char *filename, svm_problem input) {
        SvmTrain::read_problem(filename, input);
    };

    /*
     * Predicts using the SVM machine on labelled input set.
     * It outputs the prediciton accuracy
     * */
    void prediction_test();

    /*
    * Predicts using the SVM machine.
    * */
    void predict();

    /*
     * Save problem in specified file
     */
    void saveProblem(const char *filename);
};
}
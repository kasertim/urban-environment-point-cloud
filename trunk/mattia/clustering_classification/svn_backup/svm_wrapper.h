#ifndef SVM_WRAPPER_header_libSVM_
#define SVM_WRAPPER_header_libSVM_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <Eigen/Core>


#include <vector>

#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Realloc(var,type,n) (type *)realloc(var,(n)*sizeof(type))

static void (*print_func)(const char*);

namespace pcl {

class svmDataPoint {
public:
    int idx;
    float value;
};

class svmData {
public:
    double *label;
    std::vector<pcl::svmDataPoint> SV;
    
    svmData() : label(NULL) {};
};

class SVM {

protected:
    std::vector<svmData> trainingSet_;
    static void print_null(const char *s) {}
    svm_problem prob_;
    svm_model *model_;
    svm_node *x_space_;
    svm_scaling scaling_;
    
    bool labelledTrainingSet_;

    char model_file_name[1024];

    char *line;
    int max_line_len;

    char* readline(FILE *input);

    void exit_input_error(int line_num)
    {
        fprintf(stderr,"Wrong input format at line %d\n", line_num);
        exit(1);
    }

    void adaptInputToLibSVM(std::vector<svmData> trainingSet, svm_problem *prob);
    void adaptLibSVMToInput(std::vector<svmData> *trainingSet, svm_problem prob);


    bool loadProblem(const char *filename, svm_problem *prob);

    bool saveProblem(const char *filename, svm_problem prob_, bool labelled);
    bool saveProblemNorm(const char *filename, svm_problem prob_, bool labelled);

//public:
    svm_parameter param;
    
public:

    void saveModel(const char *filename) {
        if (model_->l == 0)
            return;
	
        if (svm_save_model(filename,model_))
        {
            fprintf(stderr, "can't save model to file %s\n", model_file_name);
            exit(1);
        }
    };

};

class SvmTrain : public SVM {

protected:

  void do_cross_validation();
    void scaleFactors(std::vector<svmData> trainingSet, svm_scaling *scaling);

    int cross_validation;
    int nr_fold;

    bool debug_;

public:
    SvmTrain();

    ~SvmTrain() {
        svm_destroy_param(&param);
    }

    using SVM::param;
    
    svm_model *getOutputModel() {
        return model_;
    }

    void setInputTrainingSet(std::vector<svmData> trainingSet) {
        trainingSet_.insert(trainingSet_.end(), trainingSet.begin(), trainingSet.end());
    }

    void resetTrainingSet() {
        trainingSet_.clear();
    }

    // set by parse_command_line

    int train();

    // read in a problem (in svmlight format)
    bool loadProblem(const char *filename) {
       return SVM::loadProblem(filename, &prob_);
    };

    /*
     * Save problem in specified file
     */

    void setDebugMode(bool in) {
        debug_ = in;
        if (in)
            print_func =NULL;
        else
            print_func =&print_null;

        svm_set_print_string_function(print_func);
    };

    bool saveProblem(const char *filename) {
        return SVM::saveProblem(filename, prob_, 1);
    };

    bool saveProblemNorm(const char *filename) {
        return SVM::saveProblemNorm(filename, prob_, 1);
    };
};



class SvmPredict : public SVM {

protected:
    void scaleProblem(svm_problem *input, svm_scaling scaling);
    bool predict_probability;
    using SVM::labelledTrainingSet_;
    
    std::vector< std::vector<double> > prediction_;


public:

    void setInputTrainingSet(std::vector<svmData> trainingSet) {
        trainingSet_.insert(trainingSet_.end(), trainingSet.begin(), trainingSet.end());
        SVM::adaptInputToLibSVM(trainingSet_, &prob_);
    }

    void resetTrainingSet() {
        trainingSet_.clear();
    }

    SvmPredict () {
        line = NULL;
        max_line_len = 10000;
        predict_probability=0;
        model_ = new svm_model;
        model_->l = 0;
	labelledTrainingSet_=1;
	prediction_.clear();
    }

    bool loadModel(const char *filename);
    
    std::vector< std::vector<double> > getPrediction(){
      return prediction_;
    }
    
    void savePrediction(const char *filename);

    void setInputModel(svm_model* model) {
        model_ = model;
	
	int i=0;
	while(model_->scaling[i].index != -1)
	  i++;
	
	scaling_.max = i;
	scaling_.obj = model_->scaling;
    };

    bool loadProblem(const char *filename) {
        assert (model_->l != 0);

        bool out = SVM::loadProblem(filename, &prob_);
        SVM::adaptLibSVMToInput(&trainingSet_, prob_);
        scaleProblem(&prob_, scaling_);
	return out;
    };

    bool loadProblemNorm(const char *filename) {
        bool out = SVM::loadProblem(filename, &prob_);
        SVM::adaptLibSVMToInput(&trainingSet_, prob_);
	return out;
    };

    /*
     * Predicts using the SVM machine on labelled input set.
     * It outputs the prediciton accuracy
     * */

    void setProbabilityEstimates(bool set) {
        predict_probability = set;
    };
    
    void prediction_test();

    void predict();
    
    std::vector<double> predict(svmData in);

    /*
     * Save problem in specified file
     */
    bool saveProblem(const char *filename) {
       return SVM::saveProblem(filename, prob_, 0);
    };

    bool saveProblemNorm(const char *filename) {
       return SVM::saveProblemNorm(filename, prob_, 0);
    };
};
}

#endif
#ifndef SVM_WRAPPER_impl_libSVM_
#define SVM_WRAPPER_impl_libSVM_

#include "svm_wrapper.h"
#include <assert.h>
#include <fstream>

inline float module(float a) {
    if (a>0)
        return a;
    else
        return -a;
}

inline double module(double a) {
    if (a>0)
        return a;
    else
        return -a;
}

char* pcl::SVM::readline(FILE *input)
{
    int len;
    //char *line2 = NULL;

    if (fgets(line,max_line_len,input) == NULL)
        return NULL;

    while (strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if (fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

void pcl::SvmTrain::do_cross_validation()
{
    int i;
    int total_correct = 0;
    double total_error = 0;
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double *target = Malloc(double,prob_.l);
    if (nr_fold < 2)
    {
        fprintf(stderr,"n-fold cross validation: n must >= 2\n");
        return;
    }

    svm_cross_validation(&prob_,&param,nr_fold,target);
    if (param.svm_type == EPSILON_SVR ||
            param.svm_type == NU_SVR)
    {
        for (i=0;i<prob_.l;i++)
        {
            double y = prob_.y[i];
            double v = target[i];
            total_error += (v-y)*(v-y);
            sumv += v;
            sumy += y;
            sumvv += v*v;
            sumyy += y*y;
            sumvy += v*y;
        }
        printf("Cross Validation Mean squared error = %g\n",total_error/prob_.l);
        printf("Cross Validation Squared correlation coefficient = %g\n",
               ((prob_.l*sumvy-sumv*sumy)*(prob_.l*sumvy-sumv*sumy))/
               ((prob_.l*sumvv-sumv*sumv)*(prob_.l*sumyy-sumy*sumy))
              );
    }
    else
    {
        for (i=0;i<prob_.l;i++)
            if (target[i] == prob_.y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob_.l);
    }
    free(target);
}

pcl::SvmTrain::SvmTrain() {
    prob_.l = 0;
    max_line_len = 10000;
    line = Malloc(char,max_line_len);

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;	// 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    cross_validation = 0;
    nr_fold=0;
    strcpy(model_file_name,"output.model");
    model_ = Malloc(svm_model,1);
    model_->probA = NULL;
    model_->probB = NULL;

    print_func =&print_null;
    debug_ = 0;
    svm_set_print_string_function(print_func);

}

void pcl::SvmTrain::scaleFactors(std::vector<svmData> trainingSet, svm_scaling *scaling) {

    int max=0;

    for (int i=0; i<trainingSet.size() ; i++)
        for (int j=0; j<trainingSet[i].SV.size() ; j++)
            if (trainingSet[i].SV[j].idx > max )
                max = trainingSet[i].SV[j].idx;

	    max+=1;
    scaling->obj = Malloc(struct svm_node,max+1);

    scaling->obj[max].index = -1;

    for (int i=0; i < max; i++) {
        scaling->obj[i].index = 0;
        scaling->obj[i].value = 0;
    }

    for (int i=0; i < trainingSet.size(); i++)

        for (int j=0; j < trainingSet[i].SV.size(); j++)

            if ( module(trainingSet[i].SV[j].value) > scaling->obj[ trainingSet[i].SV[j].idx ].value ){
	      scaling->obj[ trainingSet[i].SV[j].idx ].index = 1;
              scaling->obj[ trainingSet[i].SV[j].idx ].value = module(trainingSet[i].SV[j].value);
	    }
};

void pcl::SVM::adaptLibSVMToInput(std::vector<svmData> *trainingSet, svm_problem prob) {

    trainingSet->clear();

    for (int i=0; i < prob.l; i++)
    {
        svmData parent;
        int j=0;
	
	if(labelledTrainingSet_)
	  parent.label=&prob.y[i];

        while (prob.x[i][j].index != -1) {
            svmDataPoint seed;
            if ( std::isfinite( prob.x[i][j].value ) ) {
                seed.idx = prob.x[i][j].index;
                seed.value = prob.x[i][j].value;
                parent.SV.push_back(seed);
            }
            j++;
        }
        trainingSet->push_back(parent);
    }
};

void pcl::SVM::adaptInputToLibSVM(std::vector<svmData> trainingSet, svm_problem *prob) {
    prob->l = trainingSet.size(); // n of elements/points
    prob->y = Malloc(double,prob->l);
    prob->x = Malloc(struct svm_node *,prob->l);

    for (int i=0; i < prob->l; i++)
    {
      
      if(trainingSet[i].label != NULL){
        prob->y[i] = *trainingSet[i].label; // label 0 for noise, 1 for good
        labelledTrainingSet_ = 1;
      } else
	labelledTrainingSet_ = 0;
        
        prob->x[i] = Malloc(struct svm_node,trainingSet[i].SV.size()+1);
        int k=0;

        for (int j=0; j < trainingSet[i].SV.size(); j++)
            if ( std::isfinite( trainingSet[i].SV[j].value ) )
            {
                prob->x[i][k].index = trainingSet[i].SV[j].idx;

                if (trainingSet[i].SV[j].idx < scaling_.max && scaling_.obj[ trainingSet[i].SV[j].idx ].index == 1)
                    prob->x[i][k].value = trainingSet[i].SV[j].value / scaling_.obj[ trainingSet[i].SV[j].idx ].value;
                else
                    prob->x[i][k].value = trainingSet[i].SV[j].value;

                k++;
            }

        prob->x[i][k].index = -1;

    }
};

int pcl::SvmTrain::train()
{
    assert(trainingSet_.size()>0);
    scaleFactors(trainingSet_, &scaling_);
    adaptInputToLibSVM(trainingSet_, &prob_);

    const char *error_msg;
    error_msg = svm_check_parameter(&prob_,&param);

    // initialize gamma parameter
    if (param.gamma == 0 && scaling_.max > 0)
        param.gamma = 1.0/scaling_.max;

    if (error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }

    if (cross_validation)
    {
        do_cross_validation();
    }
    else
    {
        model_ = svm_train(&prob_,&param);
        model_->scaling = scaling_.obj;
    }

    return 0;
}

bool pcl::SVM::loadProblem(const char *filename, svm_problem *prob)
{
    int elements, max_index, inst_max_index, i, j;
    FILE *fp = fopen(filename,"r");
    
    if(fp==NULL) return 0;
    
    char *endptr;
    char *idx, *val, *label;

    if (fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    elements = 0;
    prob->l = 0;

    max_line_len = 10000;
    line = Malloc(char,max_line_len);
    // readline function writes one line in var. "line"
    while (readline(fp)!=NULL)
    {
        // "\t" cuts the tab or space.
        // strtok splits the string into tokens
        char *p = strtok(line," \t"); // label
        ++elements;
        // features
        while (1)
        {
            // split the next element
            p = strtok(NULL," \t");
            if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
        }
        ++elements; // contains the number of elements in the string
        ++prob->l; // number op
    }
    rewind(fp); // returns to the top pos of fp


    prob->y = Malloc(double,prob->l);
    prob->x = Malloc(struct svm_node *,prob->l);
    x_space_ = Malloc(struct svm_node,elements);

    max_index = 0;
    j=0;
    bool isUnlabelled = 0;

    for (i=0;i<prob->l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        // read one line in the file
        readline(fp);
        prob->x[i] = &x_space_[j];

        if (!isUnlabelled) {
            label = strtok(line," \t\n"); //save first element as label
            char * pch;
            pch = strpbrk(label,":");
            // std::cout << label << std::endl;

            // check if the first element is really a label
            if (pch == NULL) {
                if (label == NULL) // empty line
                    exit_input_error(i+1);
                labelledTrainingSet_=1;
                prob->y[i] = strtod(label,&endptr);
                if (endptr == label || *endptr != '\0')
                    exit_input_error(i+1);

                //idx = strtok(NULL,":"); // indice
            } else {
                isUnlabelled = 1;
                labelledTrainingSet_=0;
                i = -1;
                rewind(fp);
                continue;
            }
        } //else
// 	  idx=strtok(line,": \t\n");
        int k=0;
        while (1)
        {
            if (k++==0 && isUnlabelled)
                idx=strtok(line,": \t\n");
            else
                idx = strtok(NULL,":"); // indice

            val = strtok(NULL," \t"); // valore

            if (val == NULL) {

                break; // exit with the last element

            }
            //std::cout << idx << ":" << val<< " ";
            errno = 0;
            x_space_[j].index = (int) strtol(idx,&endptr,10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || x_space_[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space_[j].index;

            errno = 0;
            x_space_[j].value = strtod(val,&endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;

        }
        //std::cout <<"\n";
        if (inst_max_index > max_index)
            max_index = inst_max_index;
        x_space_[j++].index = -1;
    }



    if (param.gamma == 0 && max_index > 0)
        param.gamma = 1.0/max_index;

    if (param.kernel_type == PRECOMPUTED)
        for (i=0;i<prob->l;i++)
        {
            if (prob->x[i][0].index != 0)
            {
                fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                exit(1);
            }
            if ((int)prob->x[i][0].value <= 0 || (int)prob->x[i][0].value > max_index)
            {
                fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                exit(1);
            }
        }

    fclose(fp);
    //std::cout << "size: " << elements << " and " << max_index << std::endl;
    return 1;
}


bool pcl::SVM::saveProblem(const char *filename, svm_problem prob_, bool labelled = 0) {
    std::ofstream myfile;
    myfile.open (filename);
    
    if (!myfile.is_open()) return 0;
    
    for (int j=0; j < trainingSet_.size() ; j++)
    {
        if (labelled)
            myfile << *trainingSet_[j].label << " ";

        for (int i=0; i < trainingSet_[j].SV.size(); i++)
            if (std::isfinite( trainingSet_[j].SV[i].value) )
                myfile << trainingSet_[j].SV[i].idx << ":"<< trainingSet_[j].SV[i].value<< " ";

        myfile << "\n";
    }
    myfile.close();
    std::cout << " * " << filename << " saved" << std::endl;
    return 1;
}

bool pcl::SVM::saveProblemNorm(const char *filename, svm_problem prob_, bool labelled = 0) {
    if (prob_.l == 0) {
        std::cout << "Can't save " << filename << " whitout creating the problem before." << std::endl;
        return 0;
    }
    std::ofstream myfile;
    myfile.open (filename);
    
    if (!myfile.is_open()) return 0;
    
    for (int j=0; j < prob_.l ; j++)
    {
        if (labelled)
            myfile << prob_.y[j] << " ";

        // for (int i=0; i < nFeatures+2; i++)
        int i=0;
        while (prob_.x[j][i].index != -1) {
            myfile << prob_.x[j][i].index << ":"<< prob_.x[j][i].value<< " ";
            i++;
        }
        myfile << "\n";
    }
    myfile.close();
    std::cout << " * " << filename << " saved" << std::endl;
    return 1;
}

bool pcl::SvmPredict::loadModel(const char *filename) {
    if ((model_=svm_load_model(filename))==0)
    {
        fprintf(stderr,"can't open model file %s\n",filename);
        return 0;
    }
    scaling_.obj = model_->scaling;
    
    int i=0;
    while (model_->scaling[i].index != -1)
        i++;

    scaling_.max = i;
    return 1;
}


void pcl::SvmPredict::prediction_test()
{
    assert(model_->l != 0);
    assert(prob_.l != 0);
    assert(labelledTrainingSet_!=0);

    if (predict_probability)
    {
        if (svm_check_probability_model(model_)==0)
        {
            fprintf(stderr,"\nModel does not support probabiliy estimates\n");
            exit(1);
        }
    }
    else
    {
        if (svm_check_probability_model(model_)!=0)
            printf("\nModel supports probability estimates, but disabled in prediction.\n");
    }

    ////////////////////////////////////////////////////
    ////////////////////////////////////////////////
    ////////////////////////////////////////////////
    // controllare che model ha stesso numero di feature di input

    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type=svm_get_svm_type(model_);
    int nr_class=svm_get_nr_class(model_);
    double *prob_estimates=NULL;
    int j;

    prediction_.clear();

    if (predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model_));
        else
        {
            prob_estimates = (double *) malloc(nr_class*sizeof(double));
        }
    }

    int ii=0;
    prediction_.resize(prob_.l);
    while ( ii < prob_.l  )
    {
        //int i = 0;
        double target_label, predict_label;
        char *idx, *val, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        target_label = prob_.y[ii]; //takes the first label

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model_,prob_.x[ii],prob_estimates);
            prediction_[ii].push_back(predict_label);
            for (j=0;j<nr_class;j++) {
                prediction_[ii].push_back(prob_estimates[j]);
            }
        }
        else
        {
            predict_label = svm_predict(model_,prob_.x[ii]);
            prediction_[ii].push_back(predict_label);
        }


        if (predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump  += predict_label;
        sumt  += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
        ii++;
    }
    if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
    {
        printf("Mean squared error = %g (regression)\n",error/total);
        printf("Squared correlation coefficient = %g (regression)\n",
               ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
               ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
              );
    }
    else
        printf("Accuracy = %g%% (%d/%d) (classification)\n",
               (double)correct/total*100,correct,total);
    if (predict_probability)
        free(prob_estimates);
}

void pcl::SvmPredict::predict()
{
    assert(model_->l != 0);
    assert(prob_.l != 0);

    if (predict_probability && !svm_check_probability_model(model_))
      fprintf(stderr,"\nModel does not support probabiliy estimates\n");
    
    if (!predict_probability && svm_check_probability_model(model_))
      printf("\nModel supports probability estimates, but disabled in prediction.\n");

    ////////////////////////////////////////////////////
    ////////////////////////////////////////////////
    ////////////////////////////////////////////////
    // controllare che model ha stesso numero di feature di input

    int correct = 0;
    int total = 0;

    int svm_type=svm_get_svm_type(model_);
    int nr_class=svm_get_nr_class(model_);
    double *prob_estimates=NULL;
    int j;

    prediction_.clear();

    if (predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model_));
        else
        {
            prob_estimates = (double *) malloc(nr_class*sizeof(double));
        }
    }
    int ii=0;
    prediction_.resize(prob_.l);
    while ( ii < prob_.l  )
    {
        double predict_label;
        char *idx, *val, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model_,prob_.x[ii],prob_estimates);
            prediction_[ii].push_back(predict_label);
            for (j=0;j<nr_class;j++) {
                prediction_[ii].push_back(prob_estimates[j]);
            }
        }
        else
        {
            predict_label = svm_predict(model_,prob_.x[ii]);
            prediction_[ii].push_back(predict_label);
        }

        ++total;
        ii++;
    }

    if (predict_probability)
        free(prob_estimates);
}

std::vector<double> pcl::SvmPredict::predict(pcl::svmData in){
     
  assert(model_->l != 0);

    if (predict_probability && !svm_check_probability_model(model_))
      fprintf(stderr,"\nModel does not support probabiliy estimates\n");
    
    if (!predict_probability && svm_check_probability_model(model_))
      printf("\nModel supports probability estimates, but disabled in prediction.\n");

    int svm_type=svm_get_svm_type(model_);
    int nr_class=svm_get_nr_class(model_);
    double *prob_estimates=NULL;
    int j;

    // convert input vector to libPCL format
    svm_node *buff;
    buff=Malloc(struct svm_node, in.SV.size()+10);
    int i=0;
    for(i; i<in.SV.size(); i++){
      buff[i].index = in.SV[i].idx;
      if(in.SV[i].idx < scaling_.max  && scaling_.obj[in.SV[i].idx].index == 1)
	buff[i].value = in.SV[i].value/scaling_.obj[in.SV[i].idx].value;
      else
	buff[i].value = in.SV[i].value;
    }
    buff[i].index = -1;
    
    // clean the prediction vector
    prediction_.clear();

    if (predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model_));
        else
        {
            prob_estimates = (double *) malloc(nr_class*sizeof(double));
        }
    }
    prediction_.resize(1);

        double predict_label;
        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model_,buff,prob_estimates);
            prediction_[0].push_back(predict_label);
            for (j=0;j<nr_class;j++) {
                prediction_[0].push_back(prob_estimates[j]);
            }
        }
        else
        {
            predict_label = svm_predict(model_,buff);
            prediction_[0].push_back(predict_label);
        }
    

    if (predict_probability)
        free(prob_estimates);
    
    free(buff);
    return prediction_[0];

};

void pcl::SvmPredict::scaleProblem(svm_problem *input, svm_scaling scaling) {

  assert(model_->l != 0);
  assert(scaling_.max != 0);

    for (int i=0;i<input->l;i++)
    {
        int j = 0;

        while (1) {
            if (input->x[i][j].index == -1)
                break;

            if ( input->x[i][j].index < scaling.max && scaling.obj[ input->x[i][j].index ].index == 1 )
                input->x[i][j].value = input->x[i][j].value / scaling.obj[ input->x[i][j].index ].value;

            j++;
        }
    }
}

void pcl::SvmPredict::savePrediction(const char *filename) {
  
  assert(prediction_.size() > 0);
  assert(model_->l > 0);
  
    std::ofstream output;
    output.open(filename);

    int nr_class=svm_get_nr_class(model_);
    int *labels=(int *) malloc(nr_class*sizeof(int));
    svm_get_labels(model_,labels);
    output << "labels ";
    for (int j=0 ; j < nr_class; j++)
      output << labels[j] << " ";
    
    output << "\n";

    for (int i=0; i<prediction_.size(); i++) {
        for (int j=0; j<prediction_[i].size(); j++)
            output << prediction_[i][j] << " ";

        output << "\n";
    }
    output.close();
}
#endif //SVM_WRAPPER_impl_libSVM_
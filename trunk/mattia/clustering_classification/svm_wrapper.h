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

namespace pcl
{

  /** \brief The structure must be initialized and passed to the training method pcl::SVMTrain.
   *  \param svm_type {C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR}
   *  \param kernel_type {LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED}
   */

  struct SVMParam: svm_parameter
  {
    SVMParam ()
    {
      svm_type = C_SVC; // C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
      kernel_type = RBF; // LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
      degree = 3; // for poly
      gamma = 0; // 1/num_features {for poly/rbf/sigmoid}
      coef0 = 0; //  for poly/sigmoid

      nu = 0.5; // for NU_SVC, ONE_CLASS, and NU_SVR
      cache_size = 100; // in MB
      C = 1; // for C_SVC, EPSILON_SVR and NU_SVR
      eps = 1e-3; // stopping criteria
      p = 0.1; // for EPSILON_SVR
      shrinking = 1; // use the shrinking heuristics
      probability = 0; // do probability estimates

      nr_weight = 0; // for C_SVC
      weight_label = NULL; // for C_SVC
      weight = NULL; // for C_SVC
    }
  };

  struct SVMModel: svm_model
  {
    SVMModel ()
    {
      l = 0;
      probA = NULL;
      probB = NULL;
    }
  };

  struct SVMDataPoint
  {
    int idx;
    float value;

    SVMDataPoint () : idx (-1), value (0)
    {
    }
  };

  struct SVMData
  {
    double *label;
    std::vector<pcl::SVMDataPoint> SV;

    SVMData () : label (NULL)
    {
    }
  };

  class SVM
  {

    protected:
      std::vector<SVMData> training_set_;
      static void printNull (const char *s) {}

      svm_problem prob_;
      SVMModel model_;
      svm_node *x_space_;
      svm_scaling scaling_;
      SVMParam param_;
      bool labelled_training_set_;

      char *line_;
      int max_line_len_;

      char* readline (FILE *input);

      void exitInputError (int line_num)
      {
        fprintf (stderr, "Wrong input format at line %d\n", line_num);
        exit (1);
      }

      void adaptInputToLibSVM (std::vector<SVMData> training_set, svm_problem &prob);
      void adaptLibSVMToInput (std::vector<SVMData> &training_set, svm_problem prob);


      bool loadProblem (const char *filename, svm_problem &prob);

      bool saveProblem (const char *filename, bool labelled);
      bool saveProblemNorm (const char *filename, svm_problem prob_, bool labelled);

    public:
      void saveModel (const char *filename)
      {
        if (model_.l == 0)
          return;

        if (svm_save_model (filename, &model_))
        {
          fprintf (stderr, "can't save model to file %s\n", filename);
          exit (1);
        }
      };

      SVM() : line_ (NULL), max_line_len_ (10000), labelled_training_set_ (1)
      {
      }

      ~SVM ()
      {
        svm_destroy_param (&param_);

        if (scaling_.max > 0)
          free (scaling_.obj);

        if (prob_.l > 0)
        {
          free (prob_.x);
          free (prob_.y);
        }
      }

  };

  class SvmTrain : public SVM
  {

    protected:
      using SVM::labelled_training_set_;
      using SVM::model_;
      using SVM::line_;
      using SVM::max_line_len_;
      using SVM::training_set_;
      using SVM::prob_;
      using SVM::scaling_;
      using SVM::param_;

      void doCrossValidation();
      void scaleFactors (std::vector<SVMData> training_set, svm_scaling &scaling);

      int cross_validation_;
      int nr_fold_;
      bool debug_;

    public:
      SvmTrain() : debug_ (0), cross_validation_ (0), nr_fold_ (0)
      {
        svm_set_print_string_function (&printNull);
      }

      ~SvmTrain ()
      {
        if (model_.l > 0)
          svm_free_model_content (&model_);
      }

      void
      setParameters (SVMParam param)
      {
        param_ = param;
      }

      SVMParam
      getParameters ()
      {
        return param_;
      }

      SVMModel
      getOutputModel ()
      {
        return model_;
      }

      void
      setInputTrainingSet (std::vector<SVMData> training_set)
      {
        training_set_.insert (training_set_.end(), training_set.begin(), training_set.end());
      }

      std::vector<SVMData>
      getInputTrainingSet ()
      {
        return training_set_;
      }

      void resetTrainingSet ()
      {
        training_set_.clear();
      }

      // set by parse_command_line

      int trainClassifier ();

      // read in a problem (in svmlight format)
      bool loadProblem (const char *filename)
      {
        return SVM::loadProblem (filename, prob_);
      };

      /*
       * Save problem in specified file
       */

      void setDebugMode (bool in)
      {
        debug_ = in;

        if (in)
          svm_set_print_string_function (NULL);
        else
          svm_set_print_string_function (&printNull);
      };

      bool saveProblem (const char *filename)
      {
        return SVM::saveProblem (filename, 1);
      };

      bool saveProblemNorm (const char *filename)
      {
        return SVM::saveProblemNorm (filename, prob_, 1);
      };
  };



  class SvmClassify : public SVM
  {

    protected:
      void scaleProblem (svm_problem &input, svm_scaling scaling);
      bool predict_probability_, model_extern_copied_;

      using SVM::labelled_training_set_;
      using SVM::model_;
      using SVM::line_;
      using SVM::max_line_len_;
      using SVM::training_set_;
      using SVM::prob_;
      using SVM::scaling_;
      using SVM::param_;

      std::vector< std::vector<double> > prediction_;

    public:
      void setInputTrainingSet (std::vector<SVMData> training_set)
      {
        training_set_.insert (training_set_.end(), training_set.begin(), training_set.end());
        SVM::adaptInputToLibSVM (training_set_, prob_);
      }

      void resetTrainingSet()
      {
        training_set_.clear();
      }

      SvmClassify () : model_extern_copied_ (0), predict_probability_ (0)
      {
      }

      ~SvmClassify ()
      {
        if (!model_extern_copied_ && model_.l > 0)
          svm_free_model_content (&model_);
      }

      bool loadModel (const char *filename);

      void getPrediction (std::vector< std::vector<double> > &out)
      {
        out.clear ();
        out.insert (out.begin(), prediction_.begin(), prediction_.end());
      }

      void savePrediction (const char *filename);

      void setInputModel (SVMModel model)
      {
        // model (inner pointers are references)
        model_ = model;

        int i = 0;

        while (model_.scaling[i].index != -1)
          i++;

        scaling_.max = i;

        scaling_.obj = Malloc (struct svm_node, i + 1);

        scaling_.obj[i].index = -1;

        // Performing full scaling copy
        for (int j = 0; j < i; j++)
        {
          scaling_.obj[j] = model_.scaling[j];
        }

        model_extern_copied_ = 1;
      };

      bool loadProblem (const char *filename)
      {
        assert (model_.l != 0);

        bool out = SVM::loadProblem (filename, prob_);
        SVM::adaptLibSVMToInput (training_set_, prob_);
        scaleProblem (prob_, scaling_);
        return out;
      };

      bool loadProblemNorm (const char *filename)
      {
        bool out = SVM::loadProblem (filename, prob_);
        SVM::adaptLibSVMToInput (training_set_, prob_);
        return out;
      };

      /*
       * Predicts using the SVM machine on labelled input set.
       * It outputs the prediciton accuracy
       * */

      void setProbabilityEstimates (bool set)
      {
        predict_probability_ = set;
      };

      void predictionTest ();

      void predict ();

      void getLabel (std::vector<int> &labels)
      {
        int nr_class = svm_get_nr_class (&model_);
        int *labels_ = (int *) malloc (nr_class * sizeof (int));
        svm_get_labels (&model_, labels_);

        for (int j = 0 ; j < nr_class; j++)
          labels.push_back (labels_[j]);
      };

      std::vector<double> predict (SVMData in);

      /*
       * Save problem in specified file
       */
      bool saveProblem (const char *filename)
      {
        return SVM::saveProblem (filename, 0);
      };

      bool saveProblemNorm (const char *filename)
      {
        return SVM::saveProblemNorm (filename, prob_, 0);
      };
  };
}

#endif

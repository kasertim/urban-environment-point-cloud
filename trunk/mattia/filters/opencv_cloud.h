// #include <cxcore.h>
// #include <cv.h>
// #include <cvaux.h>
// #include <highgui.h>

struct freqVector {
  std::string name;
  std::vector<int> list;
};

class Display_results {
public:
  void setInput1(freqVector input1){
    input1_ = input1;
  }
  
  void setInput2(freqVector input2){
    input2_ = input2;
  }
  
  void setInput3(freqVector input3){
    input3_ = input3;
  }

  void setInput4(freqVector input4){
    input4_ = input4;
  }
  
private:
  freqVector input1_, input2_, input3_, input4_;
  //CvMat *output_;
  
  void createQuadrant(){
    //input4_.
  }
  
};


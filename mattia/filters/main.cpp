#include <list>
#include <pcl/common/time.h>
using namespace std;

// Simple example uses type int
#define MAXIT 50000000
int main(void)
{
    pcl::ScopeTime time("performance");
    float endTime;
    list<int> L;
    int i;

    time.reset();

    for (i=0; i<MAXIT;i++)
        L.push_back(i); // Insert a new element at the end
    list<int>::iterator it;
    int sum=0;
    for (it=L.begin(); it != L.end(); ++it)
        sum += *it;
    
    time.getTime();

    return 0;
}
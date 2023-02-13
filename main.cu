#include <map>
#include <string>
#include <iostream>
#include <ctime>
#include <thrust/device_vector.h>
#include "moderngpu/src/moderngpu/kernel_reduce.hxx"
#include "moderngpu/src/moderngpu/kernel_segreduce.hxx"

using namespace std;
#include "gpcACC.h"

int main(int ac, char **av)
{
    std::clock_t start;
    int x;

    if(ac < 2) {
        cout << "Usage : gpcACC [--QPS-test] | [ [-l load size(MB)] [-v] script.sql ]" << endl;
        exit(1);	
    }
    // test QPS via gpcACCExecute	-- this section is the only C++ dependency
    else if (string(av[1]) == "--QPS-test") {
        gpcACCInit(NULL);
        start = std::clock();
        for (x=0; x< 1000; x++)  {
            gpcACCExecute("A1 := SELECT  count(n_name) AS col1 FROM nation;\n DISPLAY A1 USING ('|');");
        }
        cout<< "Ave QPS is : " <<  ( 1000/ (( std::clock() - start ) / (double)CLOCKS_PER_SEC )) << endl;
        gpcACCClose();
    }
    else {				// ordinary gpcACC file mode
        cout << "Executing file:" << endl;
	return execute_file( ac, av) ;
    }
}



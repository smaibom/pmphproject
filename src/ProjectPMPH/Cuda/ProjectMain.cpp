#include "../include/ParseInput.h"

#include "ProjHelperFun.cu.h"
#include <fstream>

int main()
{
    unsigned int OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T; 
	const REAL s0 = 0.03, strike = 0.03, t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T ); 

    REAL* res = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));

	ifstream runtimes_r;
	runtimes_r.open ("runtime_cuda.txt");
	int cont = 0;
	int ind;
	vector<unsigned long int> time;
	
	while(runtimes_r>> ind){
		unsigned long int temp;
		runtimes_r >> temp;
		time.push_back(temp);
		cont++;
	}
	
	runtimes_r.close();
	
	ofstream runtimes_w;
	runtimes_w.open ("runtime_cuda.txt");
    {   // Original Program (Sequential CPU Execution)
        cout<<"\n// Running Original, Sequential Project Program"<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_OrigCPU( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
		time.push_back(elapsed);
		cont++;
		

        // validation and writeback of the result
        bool is_valid = validate   ( res, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res, OUTER_LOOP_COUNT, 
                             NUM_X, NUM_Y, NUM_T, false, 1/*Ps*/, elapsed );        
    }
	
	for(int i=0; i<cont; i++)
		runtimes_w<<i<<" "<<time[i]<<endl;
	runtimes_w.close();
	
}
   
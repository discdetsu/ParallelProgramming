#include <stdio.h>
#include <math.h>

#define N 100
#define R 3
#define BLOCK_SIZE 512

__global__ void singlethread_stencil(int* d_in, int* d_out, int M)
{
	int tid = threadIdx.x;
	if (tid == 0){
		for (int i = 0; i < M; i++){
			for (int j = -R; j <= R; j++)
				d_out[i] += d_in[i + R + j];
		}
	}

}

__global__ void multiplethreads_stencil(int* d_in, int* d_out, int M)
{ 
     int tid = (blockIdx.x*BLOCK_SIZE)+threadIdx.x;
	 if (tid < M){
	    int result = 0;
	    for (int j = -R; j <= R; j++)
	        result += d_in[tid + R + j];
	    d_out[tid] =  result;
				           
	    }
}

__global__ void faster_stencil(int* d_in, int* d_out, int M){
	
    __shared__ int temp[BLOCK_SIZE+(2*R)];
	
    int g_id = (blockIdx.x*BLOCK_SIZE)+threadIdx.x;
	int l_id = threadIdx.x+R;
	

	if(g_id < M) {
	    temp[l_id] = d_in[g_id+R];
	    if(threadIdx.x < R){
	    temp[l_id-R] = d_in[g_id];
	    temp[l_id+BLOCK_SIZE] = d_in[g_id+BLOCK_SIZE];
	}
	__syncthreads();

	int result = 0;
        for(int j=-R; j<=R; j++)
		result+= temp[l_id+R];

	d_out[g_id] =result;
    }
}

int main()
{
	int M = N-2*R;

	int h_in[N];
	int h_out[M];

	for (int i = 0; i < N; i++)
		h_in[i] = 1;

	int* d_in;
	int* d_out;

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	//Part 1: Memmory tranfer f rom host to device
	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMalloc((void**) &d_out, M*sizeof(int));

	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);

	//Part 2: Excute Kernel
    
    cudaEventRecord(start);
	singlethread_stencil<<<1, M>>>(d_in, d_out, M);
    //multiplethreads_stencil<<<(int) ceil(N/(double) BLOCK_SIZE), BLOCK_SIZE>>>(d_in, d_out, M);
    //faster_stencil<<<(int) ceil(N/(double) BLOCK_SIZE), BLOCK_SIZE>>>(d_in, d_out, M);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
	//Part 3: Memory tranfer from device to host
	cudaMemcpy(&h_out, d_out, M*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	//Part 4: Check the result
	
	for (int i = 0; i < M; i++){
		if (h_out[i] != 2*R+1){
			printf("Incorrect result!\n");
			return -1;
		}
        
        else {
            printf("Correct Result\n");
            printf("Time used: %f milliseconds\n", ms);
            return -1;
    	}
    }

}

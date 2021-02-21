#include <stdio.h>
#include <math.h>

#define N 1024

//Interleave addressing kernel_version
__global__ void interleaved_reduce(int *d_in, int *d_out)
{
	//using shared memory
	__shared__ int sm[N];

	int i = threadIdx.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	sm[i] = d_in[id];
	
	__syncthreads();

	/*int M = N/2;
	for (int s = 1; s <= N; s *= 2){
		if (i < M){
			printf("stride = %d, thread %d is active\n" , s, i);
			d_in[(2*s)*i] = d_in[(2*s)*i] + d_in[(2*s)*i+s];		
		}
		M = M/2;
	}
	if (i == 0)
		d_out[0] = d_in[0];
	*/

	for (int s = 1; s < blockDim.x; s *= 2){
		int i = 2 * s * id;
		if (i < blockDim.x){
			sm[i] += sm[i+s];
		}

		__syncthreads();
	}
	if (i == 0)
		d_out[blockIdx.x] = sm[0];
}

//Contiguous addressing kernel version
 __global__ void contiguous_reduce(int *d_in, int *d_out)
{
	/*
	 int i = threadIdx.x;
     int M = N/2;
     for (int s = M; s > 0; s /= 2){
         if (i < M){
             printf("stride = %d, thread %d is active\n" , s, i);
             d_in[i] = d_in[i] + d_in[i+s];
         }
         M = M/2;
     }
     if (i == 0)
     	d_out[0] = d_in[0];
	*/

	//using shared memory
	__shared__ int sm[N];
 
	int i = threadIdx.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	sm[i] = d_in[id];

	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s /= 2){
		if (i < s){
			sm[i] += sm[i+s];
		}
		__syncthreads();
	}
	if (i == 0)
		d_out[blockIdx.x] = sm[0];
}

int main()
{
	int h_in[N];
	int h_out = 0;

	//timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i = 0; i < N; i++)
		h_in[i] = i+1;

	int *d_in, *d_out;

	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMalloc((void**) &d_out, sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);
	
	//kernel call

	// ==================== interleaved_reduce =======================
	/*
	cudaEventRecord(start);
	interleaved_reduce<<<1, 1024>>>(d_in, d_out);
	cudaEventRecord(stop);
	*/

	// =================== contiguous_reduce =========================
	cudaEventRecord(start);
	contiguous_reduce<<<1, 1024>>>(d_in, d_out);
	cudaEventRecord(stop);

	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

	cudaFree(d_in);
	cudaFree(d_out);

	printf("Output %d\n", h_out);
	printf("Time used: %f milliseconds\n", ms);

	return -1;
}

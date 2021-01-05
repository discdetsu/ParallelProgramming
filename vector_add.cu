#include <stdio.h>

__global__ void parallel_vector_add(int* d_a, int* d_b, int* d_c, int* d_n)
{
	int i = (blockIdx.x*blockIdx.x)+threadIdx.x;
	if(i < *d_n)
	{
		printf(" I am thread #%d, and about to compute c[%d]. \n", i, i);
		d_c[i] = d_a[i] + d_b[i];
	}

	else
	{
		printf("I am thread #%d, and doing nothing.\n", i);
	}
}

int main()
{
	int n;
	scanf("%d", &n);
	
	// Declare input and output on host
	int h_a[n];
	int h_b[n];
	int h_c[n];
	for(int i = 0; i < n; i++)
	{
	    h_a[i] = i;
	    h_b[i] = n - i;
	}

	// PART I: Copy data from host to device
	int* d_a, *d_b, *d_c, *d_n;
	cudaMalloc((void **) &d_a, n*sizeof(int));
	cudaMalloc((void **) &d_b, n*sizeof(int));
	cudaMalloc((void **) &d_c, n*sizeof(int));
	cudaMalloc((void **) &d_n, sizeof(int));

	cudaMemcpy(d_a, &h_a, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int),  cudaMemcpyHostToDevice);

	// PART II: Kernel launch
	int max=(n/512);
	parallel_vector_add<<<max,512>>>(d_a,d_b,d_c,d_n);
        cudaDeviceSynchronize();
	
	// PART III: Copy data from device back to host, and free all allocate on device
	cudaMemcpy(&h_c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	for (int i = 0; i < n; i++){
		printf("%d", h_c[i]);
}

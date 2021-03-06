#include<stdio.h>
#include<math.h>

#define N 8

__global__ void exclusive_scan(int *d_in)
{
    //Phase 1 (Uptree)
    int s = 1;
        __shared__ int temp_in[N];
        int i = 2*s*(threadIdx.x+1)-1;
        temp_in[i] = d_in[i];

        __syncthreads();

    for(; s<=N-1; s<<=1)
    {
        if((i-s >= 0) && (i<N)) {
            //printf("s = %d, i= %d \n", s, i);
            int a = temp_in[i];
            int b = temp_in[i-s];
            __syncthreads();
            temp_in[i] = a+b;
            //printf("Write in[%d] = %d\n", i, a+b);
        }
        __syncthreads();
    }

    //Phase 2 (Downtree)
    if(threadIdx.x == 0)
        temp_in[N-1] = 0;

    for(s = s/2; s >= 1; s>>=1)
    {
        int i = 2*s*(threadIdx.x+1)-1;
        if((i-s >= 0) && (i<N)) {
            //printf("s = %d, i= %d \n", s, i);
            int r = temp_in[i];
            int l = temp_in[i-s];
            __syncthreads();
            temp_in[i] = l+r;
            temp_in[i-s] = r;
            __syncthreads();
            //printf("Write in[%d] = %d\n", i, a+b);
        }
        __syncthreads();
    }

        d_in[i] = temp_in[i];
}

int main()
{
        int h_in[N];
        int h_out[N];

        //timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        h_in[0] = 3;
    h_in[1] = 1;
    h_in[2] = 7;
    h_in[3] = 0;
    h_in[4] = 4;
    h_in[5] = 1;
    h_in[6] = 6;
    h_in[7] = 3;

        int *d_in;
        //int *d_out;

        cudaMalloc((void**) &d_in, N*sizeof(int));
        //cudaMalloc((void**) &d_out, N*sizeof(int));
        cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        //Implementing kernel call
        exclusive_scan<<<1, 4>>>(d_in);
        cudaEventRecord(stop);

        cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        for(int i=0; i<N; i++)
                printf("out[%d] =  %d\n", i, h_out[i]);

    cudaFree(d_in);

        printf("Time used: %f milliseconds\n", ms);

        return -1;

}

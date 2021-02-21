# ParallelProgramming

## Running time of 1D Stencil

### Single thread
![SingleThread](1D_StencilResult/single.png)

### Multiple threads
![MultipleThreads](1D_StencilResult/multiple.png)

### Multiple Faster threads
![MultipleFasterThreads](1D_StencilResult/faster.png)

## Running time of Matrix Addition

### When N is 800
![MatrixAdd](MatrixAddResult/matrix.png)

## Running time of Reduce

### Interleave addressing by using per-block shared memory
![Interleaved](ReduceResult/interleaved.png)

### Contiguous addressing by using per-block shared memory
![Contiguous](ReduceResult/contiguous.png)
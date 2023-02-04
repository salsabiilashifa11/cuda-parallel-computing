#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "time.h"

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN -1000


/* Struct Matrix
 *
 * Matrix representation consists of matrix data
 * and effective dimensions
 * */
typedef struct Matrix {
	int mat[NMAX][NMAX];	// Matrix cells
	int row_eff;			// Matrix effective row
	int col_eff;			// Matrix effective column
} Matrix;


/*
 * init_matrix group
 *
 * Initializing new matrix
 * Setting all element to 0 and effective dimensions according
 * to nrow and ncol
 *
 * [PARAM]
 * *m: newly allocated matrix
 * nrow: row dimension
 * ncol: col dimension
 * */
__host__ void init_m_cpu(Matrix *m, int nrow, int ncol) {
	m->row_eff = nrow;
	m->col_eff = ncol;

	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			m->mat[i][j] = 0;
		}
	}
}

__device__ void init_m_gpu(Matrix *m, int nrow, int ncol) {
	m->row_eff = nrow;
	m->col_eff = ncol;

	// kalo gapake ini, waktunya nambah +100ms
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			m->mat[i][j] = 0;
		}
	}
}

/*
 * input_matrix
 *
 * Returns a matrix with values from stdin input
 *
 * [PARAM]
 * nrow: row dimension
 * ncol: col dimension
 *
 * [RETURN]
 * input matrix nrow*ncol
 * */
__host__ Matrix input_matrix(int nrow, int ncol) {
	Matrix input;
	init_m_cpu(&input, nrow, ncol);

	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
			scanf("%d", &input.mat[i][j]);
		}
	}

	return input;
}

/*
 * get_m_range
 *
 * Returns the range between maximum and minimum
 * element of a matrix
 *
 * [PARAM]
 * *m: Matrix
 * 
 * [RETURN]
 * difference between maximum and minimum data within a matrix
 * */
__device__ int get_m_range(Matrix *m) {
	int max = DATAMIN;
	int min = DATAMAX;
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			int el = m->mat[i][j];
			if (el > max) max = el;
			if (el < min) min = el;
		}
	}

	return max - min;
}

/*
 * get_median
 *
 * Returns median of array n of length
 * 
 * [PARAM]
 * *n: array of integers
 * length: length of the array
 * */
__host__ int get_median(int *n, int length) {
	int mid = length / 2;
	if (length & 1) return n[mid];

	return (n[mid - 1] + n[mid]) / 2;
}


/*
 * get_floored_mean
 *
 * Returns floored mean from an array of integers
 *
 * [PARAM]
 * *n: array of integers
 * length: length of the array
 * */
__host__ long get_floored_mean(int *n, int length) {
	long sum = 0;
	for (int i = 0; i < length; i++) {
		sum += n[i];
	}

	return sum / length;
}



/*
 *  supression_op
 *
 * Returns the sum of intermediate value of special multiplication
 * operation where kernel[0][0] corresponds to target[row][col]
 *
 * [PARAM]
 * *kernel: kernel matrix
 * *target: target matrix
 * row: target row
 * col: target column
 *
 * [RETURN]
 * intermediate sum of kernel matrix multplied by target matrix
 * */
__device__ int supression_op(Matrix *kernel, Matrix *target, int row, int col) {
	int intermediate_sum = 0;
	for (int i = 0; i < kernel->row_eff; i++) {
		for (int j = 0; j < kernel->col_eff; j++) {
			intermediate_sum += kernel->mat[i][j] * target->mat[row + i][col + j];
		}
	}

	return intermediate_sum;
}

/*
 * matrix convolution
 *
 * Return the output matrix of convolution operation
 * between kernel and target
 *
 * [PARAM]
 * *kernel: kernel matrix
 * *target: target matrix
 * 
 * [RETURN]
 * output matrix as the result of convolution process
 * */
__device__ Matrix convolution(Matrix *kernel, Matrix *target) {
	Matrix out;
	int out_row_eff = target->row_eff - kernel->row_eff + 1;
	int out_col_eff = target->col_eff - kernel->col_eff + 1;

	init_m_gpu(&out, out_row_eff, out_col_eff);

	for (int i = 0; i < out.row_eff; i++) {
		for (int j = 0; j < out.col_eff; j++) {
			out.mat[i][j] = supression_op(kernel, target, i, j);
		}
	}

	return out;
}

__global__ void cuda_convolution(Matrix *d_kernel, Matrix *d_target, int d_num_target, int *d_arr_range){
	int i =  threadIdx.x + blockDim.x * blockIdx.x;

	if (i < d_num_target) {
		d_target[i] = convolution(d_kernel, &d_target[i]);
		d_arr_range[i] = get_m_range(&d_target[i]);
	}
}


/*
 * merge_sort_gpu
 *
 * Sorts array n with parallel merge sort algorithm
 * 
 * [PARAM]
 * origin: source matrix
 * destination:	destination matrix
 * size: size of ...
 * width: width of ...	
 * slices: slices of ...	
 * */
__global__ void merge_sort_gpu(int* origin, int* destination, int size, int width, int slices, int n_threads, int n_blocks) {
	int idx =  threadIdx.x + blockDim.x * blockIdx.x;
	int firstIdx = width * idx * slices;
	int midIdx, lastIdx;

	for (long slice = 0; slice < slices; slice++) {
        if (firstIdx >= size)
            break;

        midIdx = min(firstIdx + (width >> 1), size);
        lastIdx = min(firstIdx + width, size);

		int i = firstIdx;
		int j = midIdx;
		for (int k = firstIdx; k < lastIdx; k++) {
			if (i < midIdx && (j >= lastIdx || origin[i] < origin[j])) {
				destination[k] = origin[i];
				i++;
			} else {
				destination[k] = origin[j];
				j++;
			}
		}
        firstIdx += width;
    }
}

/*
 * merge_sort
 *
 * Function to call parallelized merge sort algorithm
 *
 * [PARAM]
 * data: array that will be sorted
 * size: size of the array
 * n_threads: amount of threads used
 * n_blocks: amount of blocks used
 * */
__host__ void merge_sort(int* data, int size, int n_threads, int n_blocks) {
	int* _data;
	int* _swap;
	int _n_threads;
	int _n_blocks;

	// Allocate and copy the input data to the device data
	cudaMalloc((void**) &_data, size * sizeof(int));
	cudaMalloc((void**) &_swap, size * sizeof(int));
	cudaMemcpy(_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate amount of thread and block for the device variable
	cudaMalloc((void**) &_n_threads, sizeof(int));
	cudaMalloc((void**) &_n_blocks, sizeof(int));
	cudaMemcpy(&_n_threads, &n_threads, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&_n_blocks, &n_blocks, sizeof(int), cudaMemcpyHostToDevice);

	int* origin = _data;
	int* destination = _swap;
	int nThreads = n_threads * n_blocks;

	// Slice the array then do the merge sort process with iteration size of width
	for (int width = 2; width < (size << 1); width <<= 1) {
		long slices = size / ((nThreads) * width) + 1;

		merge_sort_gpu<<<n_blocks, n_threads>>>(origin, destination, size, width, slices, n_threads, n_blocks);

		// Switch input and output array
		if (origin == _data){
			origin = _swap;
		} else origin = _data;

		if (destination == _data){
			destination = _swap;
		} else destination = _data;
	}

	// Copy the result and return it to the host
	cudaMemcpy(data, origin, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}


__host__ Matrix* alokasi_matrix(Matrix* m, int size) {
	Matrix *mat;
	cudaMalloc((void**)&mat, size);
	cudaMemcpy(mat, m, size, cudaMemcpyHostToDevice);

	return mat;
}

__host__ int* alokasi_array(int* a, int size) {
	int *arr;
	cudaMalloc((void**)&arr, size);
	cudaMemcpy(arr, a, size, cudaMemcpyHostToDevice);

	return arr;
}


int main() {
	// initialize kernel matrix
	int kernel_row, kernel_col;

	// reads kernel's row and column and initalize kernel matrix from input
	scanf("%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);

	// initialize target matrix
	int num_targets, target_row, target_col;

	// reads number of target matrices and their dimensions.
	scanf("%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));

	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
	}

	
	Matrix* _kernel = alokasi_matrix(&kernel, sizeof(Matrix));
	Matrix* _target = alokasi_matrix(arr_mat, sizeof(Matrix) * num_targets);

	int *arr_range = (int*)malloc(sizeof(int) * num_targets);

	int *_arr_range = alokasi_array(arr_range, sizeof(int) * num_targets);
	
	clock_t begin = clock();
	dim3 gridDim(num_targets);
	cuda_convolution<<<(num_targets/256)+1, 256>>>(_kernel, _target, num_targets, _arr_range);
	cudaDeviceSynchronize();

	cudaMemcpy(arr_mat, _target, sizeof(Matrix) * num_targets, cudaMemcpyDeviceToHost);
	cudaMemcpy(arr_range, _arr_range, sizeof(int) * num_targets, cudaMemcpyDeviceToHost);
	
	
	merge_sort(arr_range, num_targets, (num_targets/256) + 1, num_targets);


	// Stop measuring time and calculate the elapsed time
	clock_t end = clock();

	double elapsed = (double)(end - begin) / CLOCKS_PER_SEC;

	std::cout << "Min: " << arr_range[0] << "\n";
	std::cout << "Max: " << arr_range[num_targets-1] << "\n";
	std::cout << "Median: " << get_median(arr_range, num_targets) << "\n";
	std::cout << "Average: " << get_floored_mean(arr_range, num_targets) << "\n";
	std::cout << << "Elapsed time: " << elapsed*1000 << "ms\n";
}
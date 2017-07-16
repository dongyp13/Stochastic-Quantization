#include "caffe/util/lowbit_functions.hpp"
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

namespace caffe{

template <typename Dtype>
__global__ void binarize_kernel(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int kel)
{ 
    CUDA_KERNEL_LOOP(idx, num) {
		int n = idx / kel;
		if (weight[idx] >= 0) {
			if (mask[n] == 1) {
				binaryweight[idx] = 1;
			}
			else {
				binaryweight[idx] = weight[idx];
			}
		}
		else {
			if (mask[n] == 1) {
				binaryweight[idx] = -1;
			}
			else{
				binaryweight[idx] = weight[idx];
			}
		}
    }
}

template <typename Dtype>
void caffe_gpu_binarize(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int weight_col)
{
    const int N = num * weight_col;
    binarize_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(weight, binaryweight, mask, N, weight_col);
} 

template void caffe_gpu_binarize<float>(float* weight, float* binaryweight, const int* mask, const int num, const int weight_col);
template void caffe_gpu_binarize<double>(double* weight, double* binaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
__global__ void clip_kernel(Dtype* weight, const int n)
{ 
    CUDA_KERNEL_LOOP(idx, n){
		if(weight[idx] >= 1){
			weight[idx] = 1;
		}
		if(weight[idx] <= -1){
			weight[idx] = -1;
		}
	}
}

template <typename Dtype>
void caffe_gpu_clip(Dtype* weight, const int num, const int weight_col)
{
    const int N = num * weight_col;
    clip_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(weight, N);
} 

template void caffe_gpu_clip<float>(float* weight, const int num, const int weight_col);
template void caffe_gpu_clip<double>(double* weight, const int num, const int weight_col);

template <typename Dtype>
__global__ void scal_kernel(const Dtype* alpha, Dtype* binaryweight, const int* mask, const int num, const int kel)
{ 
    CUDA_KERNEL_LOOP(idx, num) {
		int n = idx / kel;
		if(mask[n] == 1) {
			binaryweight[idx] *= alpha[n];
		}
    }
}

template <typename Dtype>
void caffe_gpu_binary_scaling(Dtype* weight, Dtype* binaryweight, const int* mask,
	Blob<Dtype>* alpha, const int num, const int weight_col)
{
    const int N = num * weight_col;
    const int kel = weight_col;
    for (int n = 0; n < num; n++) {
		caffe_gpu_asum(kel, weight + n * kel, alpha->mutable_cpu_data() + n);
		alpha->mutable_cpu_data()[n] /= kel;
	}
    scal_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(alpha->gpu_data(), binaryweight, mask, N, weight_col);
} 

template void caffe_gpu_binary_scaling<float>(float* weight, float* binaryweight, const int* mask,
	Blob<float>* alpha, const int num, const int weight_col);
template void caffe_gpu_binary_scaling<double>(double* weight, double* binaryweight, const int* mask,
	Blob<double>* alpha, const int num, const int weight_col);

template <typename Dtype>
__global__ void delta_kernel(const Dtype* weight, Dtype* i_delta_weight, Dtype* i_delta_sign, 
	const Dtype* delta, const int num)
{ 
    CUDA_KERNEL_LOOP(idx, num){
		if((weight[idx] > delta[0]) || (weight[idx] < -delta[0])){
			i_delta_weight[idx] = weight[idx];
			i_delta_sign[idx] = 1;
		}
		else{
			i_delta_weight[idx] = 0;
			i_delta_sign[idx] = 0;
		}
    }
}

template <typename Dtype>
void caffe_gpu_ternary_scaling(Dtype* weight, Dtype* ternaryweight, Dtype* i_delta_weight, Dtype* i_delta_sign,
	const int* mask, const Dtype* delta, Blob<Dtype>* alpha, const int num, const int weight_col)
{
	const int N = num * weight_col;
    const int kel = weight_col;
	delta_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(weight, i_delta_weight, i_delta_sign, delta, N);
    for(int n=0; n<num; n++)
    {
		Dtype weight_asum;
		Dtype weight_num;
		caffe_gpu_asum(kel, i_delta_weight+n*kel, &weight_asum);
		caffe_gpu_asum(kel, i_delta_sign+n*kel, &weight_num);
		if(weight_num == 0){
			alpha->mutable_cpu_data()[n] = 0;
		}
		else{
			alpha->mutable_cpu_data()[n] = weight_asum / weight_num;
		}
	}
    scal_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(alpha->gpu_data(), ternaryweight, mask, N, weight_col);
}

template void caffe_gpu_ternary_scaling<float>(float* weight, float* ternaryweight, float* i_delta_weight, float* i_delta_sign,
	const int* mask, const float* delta, Blob<float>* alpha, const int num, const int weight_col);
template void caffe_gpu_ternary_scaling<double>(double* weight, double* ternaryweight, double* i_delta_weight, double* i_delta_sign,
	const int* mask, const double* delta, Blob<double>* alpha, const int num, const int weight_col);

template <typename Dtype>
__global__ void ternarize_kernel(const Dtype* weight, Dtype* ternaryweight, const int* mask, const Dtype* delta, const int num, const int kel)
{ 
    CUDA_KERNEL_LOOP(idx, num) {
		int n = idx / kel;
		if(mask[n] == 1) {
			if(weight[idx] > delta[0]) {
				ternaryweight[idx] = 1;
			}
			else if(weight[idx] < -delta[0]) {
				ternaryweight[idx] = -1;
			}
			else {
				ternaryweight[idx] = 0;
			}
		}
		else {
			ternaryweight[idx] = weight[idx];
		}
    }
}

template <typename Dtype>
void caffe_gpu_ternarize(Dtype* weight, Dtype* ternaryweight, const int* mask, const Dtype* delta, const int num, const int weight_col)
{
    const int N = num * weight_col;
    ternarize_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(weight, ternaryweight, mask, delta, N, weight_col);
} 

template void caffe_gpu_ternarize<float>(float* weight, float* ternaryweight, const int* mask, 
	const float* delta, const int num, const int weight_col);
template void caffe_gpu_ternarize<double>(double* weight, double* ternaryweight, const int* mask, 
	const double* delta, const int num, const int weight_col);

}
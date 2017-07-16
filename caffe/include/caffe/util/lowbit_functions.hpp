#ifndef CAFFE_UTIL_LOWBIT_FUNCTIONS_H_
#define CAFFE_UTIL_LOWBIT_FUNCTIONS_H_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

namespace caffe{

template <typename Dtype>
void caffe_cpu_binarize(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_gpu_binarize(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_clip(Dtype* weight, const int num, const int weight_col);

template <typename Dtype>
void caffe_gpu_clip(Dtype* weight, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_binary_scaling(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_gpu_binary_scaling(Dtype* weight, Dtype* binaryweight, const int* mask, Blob<Dtype>* alpha, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_ternary_scaling(Dtype* weight, Dtype* ternaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_gpu_ternary_scaling(Dtype* weight, Dtype* ternaryweight, Dtype* i_delta_weight, Dtype* i_delta_sign, 
	const int* mask, const Dtype* delta, Blob<Dtype>* alpha, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_ternarize(Dtype* weight, Dtype* ternaryweight, const int* mask, const int num, const int weight_col);
  
template <typename Dtype>
void caffe_gpu_ternarize(Dtype* weight, Dtype* ternaryweight, const int* mask, const Dtype* delta, const int num, const int weight_col);	
}

#endif  // CAFFE_UTIL_LOWBIT_FUNCTIONS_H_
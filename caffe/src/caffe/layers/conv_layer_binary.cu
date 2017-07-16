#include <vector>
#include "caffe/layers/conv_layer_binary.hpp"

namespace caffe {

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialization
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  const int num = this->num_output_;
  const int weight_col = this->kernel_dim_;
  const int N = num * weight_col;
  Dtype* binaryweight = binary_weight_.mutable_gpu_data();
  caffe_copy<Dtype>(N, weight, binaryweight);
  
  // quantize the weights and save the signs into binaryweight
  caffe_gpu_binarize<Dtype>(weight, binaryweight, this->all_quantized_.gpu_data(), num, weight_col);
  caffe_gpu_binary_scaling<Dtype>(weight, binaryweight, this->all_quantized_.gpu_data(), &alpha_, num, weight_col);
  
  // Stochastic Quantization
  if (this->sq_ && (this->ratio_ < 100)){
    // roulette selection algorithm; mask is stored in 'is_quantized'
	Roulette();
    // convert the weights to a hybrid weight
	caffe_gpu_binarize<Dtype>(weight, binaryweight, this->is_quantized_.gpu_data(), num, weight_col);
    caffe_gpu_binary_scaling<Dtype>(weight, binaryweight, this->is_quantized_.gpu_data(), &alpha_, num, weight_col);
  }
  
  // Convolution operation
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, binaryweight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Roulette() {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int num = this->num_output_;
  const int weight_col = this->kernel_dim_;
  const int N = num * weight_col;
  const Dtype* binaryweight = binary_weight_.gpu_data();
  const float ratio = this->ratio_;
  Dtype* norm = error_norm_.mutable_cpu_data();
  Dtype* ns = sum_norm_.mutable_cpu_data();
  Dtype* wc = weight_copy_.mutable_gpu_data();
  
  // calculate the quantization error(||W-Q||/||W||)
  caffe_gpu_sub(N, weight, binaryweight, wc);
  for(int n = 0; n < num; n++) {
    caffe_gpu_asum(weight_col, wc + n * weight_col, norm + n);
    caffe_gpu_asum(weight_col, weight + n * weight_col, ns + n);
  }
  for(int n = 0; n < num; n++) {
    if (ns[n] == 0) {
      norm[n] = 0;
    } else {
      norm[n] = norm[n] / ns[n]; // quantization errors are stored in 'norm'
    }
  }
  int* is_quant = is_quantized_.mutable_cpu_data();
	
  // roulette
  Dtype sum = 0;
  for(int n = 0; n < num; n++) {
    sum += norm[n];
    is_quant[n] = 1;
  }
  const int real_num = int((1 - ratio / 100) * num);
  for(int i = 0; i < real_num; i++) { // select one kernel which is set to real. the probability is equal to norm
    Dtype p;
    caffe_rng_uniform(1, Dtype(0), Dtype(1), &p);
    p *= sum;
    Dtype cur_sum = 0;
    for(int n = 0; n < num; n++) {
      if(is_quant[n] == 1) { // not selected
        if((p >= cur_sum) && (p < cur_sum + norm[n])) { // hit
          is_quant[n] = 0;
          sum -= norm[n]; // remove
          break;
		}
        else {
          cur_sum += norm[n];
        }
	  }
    }
  }
}

template void BinaryConvolutionLayer<float>::Roulette();
template void BinaryConvolutionLayer<double>::Roulette();

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* binaryweight = binary_weight_.gpu_data();
  //const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, binaryweight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionLayer);

}  // namespace caffe

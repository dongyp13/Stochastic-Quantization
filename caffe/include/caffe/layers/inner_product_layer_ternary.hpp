#ifndef CAFFE_INNER_PRODUCT_LAYER_TERNARY_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_TERNARY_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/lowbit_functions.hpp"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class TernaryInnerProductLayer : public Layer<Dtype> {
 public:
  explicit TernaryInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TernaryInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void Roulette();
  
  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
  
  Blob<Dtype> alpha_; // scaling factor
  Blob<Dtype> mean_; // weight mean
  Blob<Dtype> delta_; // delta
  
  Blob<Dtype> batch_sum_multiplier_; // helper variable
  Blob<Dtype> num_by_chans_; 
  Blob<Dtype> spatial_sum_multiplier_; // helper variable
  
  Blob<Dtype> error_norm_; // quantization errors 
  Blob<Dtype> sum_norm_; // L1 norm of real-valued weights
  
  Blob<int> is_quantized_; // indicating whether each kernel weight needs to be quantized
  Blob<int> all_quantized_; // helper variable; for all weights quantized without eqk algorithm
  
  Blob<Dtype> weight_copy_; // temporary variable
  Blob<Dtype> ternary_weight_;
  Blob<Dtype> i_delta_weight_; // helper variable for calculating alpha_ in TWN
  Blob<Dtype> i_delta_sign_; // helper variable for calculating alpha_ in TWN
  
  bool sq_; // whether use stochatic quantization
  float ratio_; // quantization ratio
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_TERNARY_HPP_

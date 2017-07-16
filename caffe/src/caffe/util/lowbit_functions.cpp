#include "caffe/util/lowbit_functions.hpp"

namespace caffe{

template <typename Dtype>
void caffe_cpu_binarize(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int weight_col)
{
    for(int n = 0; n < num; n++)
        for(int w = 0; w < weight_col; w++){
            int idx = n*weight_col + w;
            if(weight[idx] >= 0){
				if(mask[n] == 1){
					binaryweight[idx]=1;
				}
				else{
					binaryweight[idx] = weight[idx];
				}
              //binaryweight[idx]=weight[idx];
            }
            if(weight[idx] < 0){
				if(mask[n] == 1){
					binaryweight[idx] = -1;
				}
				else{
					binaryweight[idx] = weight[idx];
				}
              //binaryweight[idx]=weight[idx];
            }  
        }
}
template void caffe_cpu_binarize<float>(float* weight, float* binaryweight, const int* mask, const int num, const int weight_col);
template void caffe_cpu_binarize<double>(double* weight, double* binaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_clip(Dtype* weight, const int num, const int weight_col)
{
    for(int n=0;n<num;n++)
        for(int w=0;w<weight_col;w++)
        {
            int idx=n*weight_col+w;
            if(weight[idx]>=1)
            {
                weight[idx]=1;
            }
            if(weight[idx]<=-1)
            {
                weight[idx]=-1;
            }  
            //LOG(INFO)<<"bb"<<weights[idx];
        }
}
  
template void caffe_cpu_clip<float>(float* weight, const int num, const int weight_col);
template void caffe_cpu_clip<double>(double* weight, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_binary_scaling(Dtype* weight, Dtype* binaryweight, const int* mask, const int num, const int weight_col)
{
	const int kel = weight_col;
    Dtype alpha[num];
    Dtype a[kel];
    for(int n=0;n<num;n++)
    {  
      for(int k=0;k<kel;k++)
      {
        int idx=k+kel*n;
        a[k] = weight[idx];
      }
      alpha[n] = caffe_cpu_asum(kel,a)/kel;
//      printf("%f",alpha[n]);
    } 
   for(int n=0;n<num;n++)
    {
      for(int k=0;k<kel;k++)
      {
        int idx=k+kel*n;
		if(mask[n] == 1)
			binaryweight[idx] = alpha[n]*binaryweight[idx];
      } 
    }
}
template void caffe_cpu_binary_scaling<float>(float* weight, float* binaryweight, const int* mask, const int num, const int weight_col);
template void caffe_cpu_binary_scaling<double>(double* weight, double* binaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_ternary_scaling(Dtype* weight, Dtype* ternaryweight, const int* mask, const int num, const int weight_col)
{
	/* Need to be implemented; ignore so far */
}
template void caffe_cpu_ternary_scaling<float>(float* weight, float* ternaryweight, const int* mask, const int num, const int weight_col);
template void caffe_cpu_ternary_scaling<double>(double* weight, double* ternaryweight, const int* mask, const int num, const int weight_col);

template <typename Dtype>
void caffe_cpu_ternarize(Dtype* weight, Dtype* ternaryweight, const int* mask, const int num, const int weight_col)
{
	// incorrect; need to debug; ignore so far
      int total = num * weight_col;
      Dtype sum = 0;
      for(int i=0; i<total; i++)
	  {
            sum = sum + weight[i]; 
          }
      Dtype mean = sum / total;
      Dtype weight1[total];
      for(int i=0;i<total;i++)
          {
            weight1[i] = weight[i]-mean;
          }      
      Dtype absmean = caffe_cpu_asum(total,weight1)/total;
      Dtype absmeanp = absmean*0.7;
      Dtype absmeann = -absmean*0.7;
      for(int w=0;w<total;w++)
          {
            if(weight[w] > absmeanp)
            {
              ternaryweight[w]=1;
            }
            else if(weight[w] < absmeann)
            {
              ternaryweight[w]=-1;
            }  
            else
            {
              ternaryweight[w]=0;
            }
      }
  }
template void caffe_cpu_ternarize<float>(float* weight, float* ternaryweight, const int* mask, const int num, const int weight_col);
template void caffe_cpu_ternarize<double>(double* weight, double* ternaryweight, const int* mask, const int num, const int weight_col);
}
//#include "CollectNDistributeFpnConfigs.h"
#include "CollectNDistributeFpnLayer.h"
#include <cub/cub/cub.cuh>
#include "generate_proposals_op_util_nms_gpu.h"
#include "common_gpu.h"
#include "Utils.h" 
#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h> 

using std::max;
using std::min;
using std::floor;
using std::ceil;
using namespace CollectNDistributeFpn;

namespace utils {

	// Compute the area of an array of boxes.
	caffe2::ERArrXXf BoxesArea(const caffe2::ERArrXXf& boxes) {  
		const auto w = boxes.col(2) - boxes.col(0) + 1;//   w = (boxes[:, 2] - boxes[:, 0] + 1)
		const auto h = boxes.col(3) - boxes.col(1) + 1;//   h = (boxes[:, 3] - boxes[:, 1] + 1)
		const caffe2::ERArrXXf areas = w * h;//   areas = w * h
		/*
		int count=0;
		printf("=========================area under 100=========================\n");
		for(int i=0;i<areas.rows();i++)
		  if(areas.data()[i]<10.0f)
			count++;
		printf("count : %d",count);
			//printf("[x1 y1 x2 y2 = %.2f %.2f %.2f %.2f] %d'th area : %.5f\n",i, boxes.col(2)[i], boxes.col(0)[i], boxes.col(3)[i], boxes.col(1)[i], areas.data()[i]);
        */
		return areas;
	}

	// mapping each RoI to a FPN level 
	caffe2::ERArrXXf MapRoIsToFpnLevels(Eigen::Ref<const caffe2::ERArrXXf> rois,
										const float k_min, const float k_max,//2, 5
										const float s0, const float lvl0) {  //4, 224==ROI_CANONICAL_LEVEL, ROI_CANONICAL_SCALE
        // Compute level ids
		caffe2::ERArrXXf s = BoxesArea(rois).sqrt();   
		auto target_lvls = (lvl0 + (s / s0 + 1e-6).log() / log(2)).floor(); // np.floor(lvl0 + np.log2(s / s0 + 1e-6))
		auto target_lvls_clipped = target_lvls.min(k_max).max(k_min);       // np.clip(target_lvls, k_min, k_max)
		int fpn_count[4] = { 0,0,0,0 };
		for (int i = 0; i < target_lvls_clipped.rows(); i++) {
			if      (target_lvls_clipped(i) == 2.0)	fpn_count[0]++;
			else if (target_lvls_clipped(i) == 3.0)	fpn_count[1]++;
			else if (target_lvls_clipped(i) == 4.0)	fpn_count[2]++;
			else if (target_lvls_clipped(i) == 5.0)	fpn_count[3]++;
		}
		/*
		printf("\ntarget_lvls size = %d\n", target_lvls.rows());
		printf("\ntarget_lvls_clipped size = %d\n", target_lvls_clipped.rows());
		printf("==================SCALE = %f=================\n", s0);
		printf("LEVEL 2, 3, 4, 5= %d %d %d %d\n", fpn_count[0], fpn_count[1], fpn_count[2], fpn_count[3]);*/
		return target_lvls_clipped;
	}

	// Sort RoIs from highest to lowest based on RoI scores / limit to n results
	void SortAndLimitRoIsByScores(Eigen::Ref<const caffe2::EArrXf> scores, int n,
		caffe2::ERArrXXf & rois) {

		// CHECK(rois.rows() == scores.size());
		// Create index array with 0, 1, ... N
		std::vector<int> idxs(rois.rows());
		std::iota(idxs.begin(), idxs.end(), 0);
		
		// Reuse a comparator based on scores and store a copy of RoIs that
		// will be truncated and manipulated below
		auto comp = [&scores](int lhs, int rhs) {
			if (scores(lhs) > scores(rhs)) return true;
			if (scores(lhs) < scores(rhs)) return false;
			// To ensure the sort is stable
			return lhs < rhs;
		};

		caffe2::ERArrXXf rois_copy = rois;
		// Note that people have found nth_element + sort to be much faster
		// than partial_sort so we use it here
		if (n > 0 && n < rois.rows()) {
			std::nth_element(idxs.begin(), idxs.begin() + n, idxs.end(), comp);
			rois.resize(n, rois.cols());}
		else {n = rois.rows();}

		std::sort(idxs.begin(), idxs.begin() + n, comp);
		
		for (int i = 0; i < n; i++) { // Update RoIs based on new order
			rois.row(i) = rois_copy.row(idxs[i]); }
	}

	// Updates arr to be indices that would sort the array. Implementation of
	// https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
	void ArgSort(caffe2::EArrXi & arr) {
		// Create index array with 0, 1, ... N and sort based on array values
		std::vector<int> idxs(arr.size());
		std::iota(std::begin(idxs), std::end(idxs), 0);
		std::sort(idxs.begin(), idxs.end(), [&arr](int lhs, int rhs) {
			return arr(lhs) < arr(rhs);
			});
		// Update array to match new order
		for (int i = 0; i < arr.size(); i++) {
			arr(i) = idxs[i];
		}
	}

	// Update out_filtered and out_indices with rows from rois where lvl matches
	// value in lvls passed in.
	void RowsWhereRoILevelEquals(Eigen::Ref<const caffe2::ERArrXXf> rois,
		const caffe2::ERArrXXf & lvls, const int lvl,
		caffe2::ERArrXXf * out_filtered, caffe2::EArrXi * out_indices) {
		
		//CHECK(rois.rows() == lvls.rows());// if not, RoIs and lvls count mismatch
		// Calculate how many rows we need
		int filtered_size = (lvls == lvl).rowwise().any().count();
		// Fill in the rows and indices
		out_filtered->resize(filtered_size, rois.cols());
		out_indices->resize(filtered_size);
		for (int i = 0, filtered_idx = 0; i < rois.rows(); i++) {
			auto lvl_row = lvls.row(i);
			if ((lvl_row == lvl).any()) {
				out_filtered->row(filtered_idx) = rois.row(i);
				(*out_indices)(filtered_idx) = i;
				filtered_idx++;
			}
		}
	}

} // namespace utils

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

namespace nvinfer1 {
	CollectNDistributeFpnLayerPlugin::CollectNDistributeFpnLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
		/*mClassCount = CLASS_NUM;
		mCollectNDistributeFpnKernel.clear();
		mCollectNDistributeFpnKernel.push_back(yolo1);
		mKernelCount = mCollectNDistributeFpnKernel.size();*/
	}
	CollectNDistributeFpnLayerPlugin::~CollectNDistributeFpnLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	CollectNDistributeFpnLayerPlugin::CollectNDistributeFpnLayerPlugin(const void* data, size_t length) {
		/*
		using namespace Tn;
		const char* d = reinterpret_cast<const char*>(data), * a = d;
		read(d, mThreadCount);
		mCollectNDistributeFpnKernel.resize(mKernelCount);
		auto kernelSize = mKernelCount*sizeof(CollectNDistributeFpnKernel);
		memcpy(mCollectNDistributeFpnKernel.data(),d,kernelSize);
		d += kernelSize;

		assert(d == a + length);
		*/
	}

	void CollectNDistributeFpnLayerPlugin::serialize(void* buffer)
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer), * a = d;
		write(d, mThreadCount);
		//auto kernelSize = mKernelCount*sizeof(CollectNDistributeFpnKernel);
		//memcpy(d,mCollectNDistributeFpnKernel.data(),kernelSize);
		//d += kernelSize; 
		assert(d == a + getSerializationSize());
	}

	size_t CollectNDistributeFpnLayerPlugin::getSerializationSize(){
		return sizeof(mThreadCount) + sizeof(CollectNDistributeFpn::CollectNDistributeFpnKernel) *
			mCollectNDistributeFpnKernel.size();
	}

	int CollectNDistributeFpnLayerPlugin::initialize(){
		
		roi_min_level_ = 2; 
		roi_max_level_ = 5;

		roi_canonical_level_ = 4
		roi_canonical_scale_ = 224;

        rpn_min_level_ = 2;
		rpn_max_level_ = 6;

        rpn_post_nms_topN_ = 1000;
		
		 
		for (int i = 0; i < 5; i++){
			m_inputTotalCount += m_rpn_rois_fpnH[i]        * m_rpn_rois_fpnW[i];//W==5
			m_inputTotalCount += m_rpn_rois_probs_fpn_H[i] * m_rpn_rois_probs_fpn_W[i];
		}
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, m_inputTotalCount * sizeof(float), cudaHostAllocDefault));
		 
		for (int i = 0; i < 5; i++)
			ouputTotalCount += roiCountPerFPN_ * 4; // ?x4==W'
		m_ouputTotalCount += rpn_post_nms_topN_ * (4 + 1);
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, m_ouputTotalCount * sizeof(float),cudaHostAllocDefault));
		memset(mOutputBuffer, 0, ouputTotalCount * sizeof(float));
		/**/

		return 0;
	}
	void CollectNDistributeFpnLayerPlugin::configureWithFormat(const Dims* inputDims, int nbInputs,
		const Dims* outputDims, int nbOutputs,
		DataType type,
		PluginFormat format, int maxBatchSize) {
		//std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
		assert((type == DataType::kFLOAT || type == DataType::kHALF ||
			type == DataType::kINT8) && format == PluginFormat::kNCHW);
		mDataType = type;
		//std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " "
		//<<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
	}
	Dims CollectNDistributeFpnLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 10);/*
		bottom0,1,2,3,4: "rpn_rois_fpn2,3,4,5,6
		bottom5,6,7,8,9: "rpn_roi_probs_fpn2,3,4,5,6
		top0: "rpn_rois"#(1000,5)
		top1,2,3,4: "rois_fpn2,3,4,5
		top5: "rois_idx_restore_int32"
		
		int total_num_rois = 1000;
		int num_roi_2 = num_roi_3 = num_roi_4 = num_roi_5 = 250;

		top_shape_rois==1000? x 4  
		top_shape_roi_fpn_2,3,4,5 : 250?,4
		top_shape_roi_index : 1000? x 1    */

		int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;//6-2+1=5
		m_proposal_num = 0;
		for (int i = 0; i < num_rpn_lvls; i++) {
			m_rpn_rois_fpnH[i] = inputs[i].d[0];
			m_rpn_rois_fpnW[i] = inputs[i].d[1]; // 5

			m_proposal_num += m_rpn_rois_fpnH[i];

			m_rpn_rois_probs_fpn_H[num_rpn_lvls + i] = inputs[num_rpn_lvls + i].d[0];
			m_rpn_rois_probs_fpn_W[num_rpn_lvls + i] = inputs[num_rpn_lvls + i].d[1]; // 1 
		}  

		if index == 0 
			return DimsCHW(1, rpn_post_nms_topN_, 4);
		else if index == 1 
			return DimsCHW(1, roiCountPerFPN_ , 4);
		else if index == 2 
			return DimsCHW(1, roiCountPerFPN_, 4);
		else if index == 3 
			return DimsCHW(1, roiCountPerFPN_, 4);
		else if index == 4
			return DimsCHW(1, roiCountPerFPN_, 4);
		else if index == 5 
			return DimsCHW(1, rpn_post_nms_topN_, 1);
	} 
	template <typename Dtype>
	void CollectNDistributeFpnLayerPlugin::forwardCpu(const Dtype* inputs, Dtype* outputs, cudaStream_t stream) {
		/*bottom0,1,2,3,4: "rpn_rois_fpn2,3,4,5,6
		 bottom5,6,7,8,9: "rpn_roi_probs_fpn2,3,4,5,6
		 top0: "rpn_rois"#(1000,4)  
		 top1,2,3,4: "rois_fpn2,3,4,5
		 top5: "rois_idx_restore_int32"*/ 
		
		int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;//6-2+1=5
		// bottom.size() == 10 ==  2 * num_rpn_lvls(5)
		int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1;//5-2+1=4
		// top.size() == 6 == num_roi_lvls(4) + 2
		printf("num_rpn_lvls :%d\n", num_rpn_lvls);
		CUDA_CHECK(cudaStreamSynchronize(stream));
		int i = 0;
		int size = 0;
		float* inputData = (float*)mInputBuffer;
		for (int i = 0; i < num_rpn_lvls*2; i++) {
			if (i< num_rpn_lvls){
				size = m_rpn_rois_fpnH[i] * m_rpn_rois_fpnW[i];				
			}else{
				size = m_rpn_rois_probs_fpn_H[i] * m_rpn_rois_probs_fpn_W[i];
			}
			CUDA_CHECK(cudaMemcpyAsync(inputData, inputs[i], size * sizeof(float),
													cudaMemcpyDeviceToHost, stream));
			inputData += size;
			++i;
		}

		/* Collect rois and scores in Eigen => rois = [batch_idx, x0, y0, x1, y2], ...]
		   Combine predictions across all levels and retain the top scoring

		   roi_inputs = inputs[:num_rpn_lvls]
		   score_inputs = inputs[num_rpn_lvls:]
		   rois = np.concatenate([blob.data for blob in roi_inputs])
		   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
		*/ 
		/*
		for (int i = 0; i < num_rpn_lvls; i++) {
			//const Dtype* roi_in = bottom[i]->cpu_data();//auto&->Dtype*, const Dtype* , Input(i);
			bottom[i]->num();//roi_in.dim(0) => bottom[i]->num() 
		}*/
		
		caffe2::ERArrXXf rois(m_proposal_num, 5);
		caffe2::EArrXf scores(m_proposal_num);
		int len = 0;
		
		for (int i = 0; i < num_rpn_lvls; i++) {
			const float* roi_in = (const float*)inputs[i];//bottom[i]->cpu_data();//Input(i);
			int n = m_rpn_rois_fpnH[i];//bottom[i]->shape(0);//roi_in.dim(0); 
			Eigen::Map<const caffe2::ERArrXXf> roi(roi_in, n, 5);//roi_in.data<float>(), 

			rois.block(len, 0, n, 5) = roi;
			const float* score_in = (const float*)(inputs[num_rpn_lvls + i]);// auto& => Dtype*,  Input(num_rpn_lvls + i);

			// No need to squeeze, since we are reshaping when converting to Eigen
			// https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
			Eigen::Map<const caffe2::EArrXf> score(score_in, n);//score_in.data<float>()
			scores.segment(len, n) = score;
			len += n;

		}
		// Grab only top rpn_post_nms_topN rois		
		//   inds = np.argsort(-scores)[:rpn_post_nms_topN]
		//   rois = rois[inds, :]
		printf("\n===============  BEFORE rois.data  ====================\n");// rois.rows() == proposal_num
		for (int m = 0; m < 40;) { printf("%.5f\t", rois.data()[m]); if (++m % 5 == 0) printf("\n"); }

		utils::SortAndLimitRoIsByScores(scores, rpn_post_nms_topN_, rois); //code above

		printf("\n===============  AFTER rois.data   =====================\n"); // rois.rows() == 1000
		for (int m = 0; m < 40;) { printf("%.5f\t", rois.data()[m]); if (++m % 5 == 0) printf("\n"); }
		/*
		for(int j=0 ; j < rois.rows()  ;  j++){
			if(rois.data()[j*5 + 3] >10 )
			  CHECK_GT(rois.data()[j*5 + 3],rois.data()[j*5 + 1]);
			if(rois.data()[j*5 + 4] >10 )
			  CHECK_GT(rois.data()[j*5 + 4],rois.data()[j*5 + 2]);
		}
		*/
		// Distribute 
		const int lvl_min = roi_min_level_ ; // lvl_min = FPN.ROI_MIN_LEVEL
		const int lvl_max = roi_max_level_ ; // lvl_max = FPN.ROI_MAX_LEVEL
			  int canon_scale = roi_canonical_scale_;
		const int canon_level = roi_canonical_level_;
		auto rois_block = rois.block(0, 1, rois.rows(), 4);
		// fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
		auto lvls = utils::MapRoIsToFpnLevels(rois_block, lvl_min, lvl_max,
											  canon_scale, canon_level); 
	/*
	  for(;;)
	  if(zero_level){
		lvls = utils::MapRoIsToFpnLevels(rois_block,lvl_min, lvl_max, canon_scale--, canon_level,zero_level); //canon_scale
	  }
	  else{
		break;
	  }
	*/
	//   outputs[0].reshape(rois.shape)
	//   outputs[0].data[...] = rois
		

		float* rois_out = (float*)mOutputBuffer;// 
		printf("rois.rows(): %d, rois.cols(): %d\n", rois.rows(), rois.cols());//rois shape == (1000,5)

		//Eigen::Map<caffe2::ERArrXXf> rois_out_mat(rois_out, rois.rows(), rois.cols());//->template mutable_data<float>()
		//rois_out_mat = rois;  
		//Reshape output_0 to rois.rows() x 4 
		int totalRoiCount = 0;
		for (int i = 0, j = 0; i < rois.rows() * rois.cols();) {
			totalRoiCount++;
			if (j % 5 != 0) rois_out[i++] = rois.datas()[j++];
			else j++; } 

		//printf("%.3f   ",rois_out[i]);    if(++i%4 == 0) printf("\n");
		//printf("%.3f   ",rois.data()[i]); if(++i%5 == 0) printf("\n");
		/*
		for(int i=0;i<rois.rows()*(rois.cols()-1);){
			printf("%.3f   ",rois_out[i]);
			if(++i%4 == 0)
				printf("\n");
		}

		for(int i=0; i<rois.rows()*rois.cols();){
			printf("%.3f   ",rois.data()[i]);
			if(++i%5 == 0)
				printf("\n");
		}
		*/
		/* rois_idx_order = np.empty((0, ))
		   for (output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)))
		       idx_lvl = np.where(lvls == lvl)[0]
		       blob_roi_level = rois[idx_lvl, :]
		       outputs[output_idx + 1].reshape(blob_roi_level.shape)
		       outputs[output_idx + 1].data[...] = blob_roi_level
		       rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
		   rois_idx_restore = np.argsort(rois_idx_order)
		   blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-1])*/

		caffe2::EArrXi rois_idx_restore;
		for (int i = 0, lvl = lvl_min; i < num_roi_lvls; i++, lvl++) {//num_roi_lvls == 4
			caffe2::ERArrXXf blob_roi_level;
			caffe2::EArrXi idx_lvl;
			utils::RowsWhereRoILevelEquals(rois, lvls, lvl, &blob_roi_level, &idx_lvl);
			// Output blob_roi_level 4 * roiCountPerFPN_ 
			float* roi_out = (float*)(mOutpsutBuffer+rpn_post_nms_topN_*4 + roiCountPerFPN_*4*i);
			//roi_out->Resize(blob_roi_level.rows(), blob_roi_level.cols());  
			const vector< int > roi_out_shape{ blob_roi_level.rows(), blob_roi_level.cols() }; 
			//printf("\nblob_roi_level.rows(): (%d) blob_roi_level.cols(): (%d)\n", (int)blob_roi_level.rows(), (int)blob_roi_level.cols());
			if (blob_roi_level.rows() == 0) {
				printf("%d's fpn is empty\n", lvl);
			} // const vector< int > empty_roi_out_shape{ 0, blob_roi_level.cols() }; 
			else {
				//top[i + 1]->Reshape(roi_out_shape);
				printf("\nFPN roi counts : (%d)\n", blob_roi_level.rows());
				//Eigen::Map<caffe2::ERArrXXf> roi_out_mat( roi_out,//->template mutable_data<float>()
				//											blob_roi_level.rows(),
				//											blob_roi_level.cols());
				//											roi_out_mat = blob_roi_level;
				for (int j = 0; j < blob_roi_level.rows() * blob_roi_level.cols(); j++) { roi_out[j] = blob_roi_level.data()[j]; }
				// Append indices from idx_lvl to rois_idx_restore
				rois_idx_restore.conservativeResize(rois_idx_restore.size() + idx_lvl.size());
				rois_idx_restore.tail(idx_lvl.size()) = idx_lvl;
			}
		}

		utils::ArgSort(rois_idx_restore);              //outputs[5]; (int*)top[5]->mutable_cpu_data();//Output(OutputSize() - 1);
		int* rois_idx_restore_out = (int*)mOutpsutBuffer + rpn_post_nms_topN_ * 4 + roiCountPerFPN_ * 4 * 4;
		//rois_idx_restore_out->Resize(rois_idx_restore.size());
		printf("rois_idx_restore.size() : %d\n", rois_idx_restore.size());

		for (int i = 0; i < rois_idx_restore.size(); i++)
			rois_idx_restore_out[i] = rois_idx_restore.data()[i]; 

		CUDA_CHECK(cudaMemcpyAsync(outputs, mOutputBuffer, sizeof(float)* m_ouputTotalCount, cudaMemcpyHostToDevice, stream)); 
	}

	int CollectNDistributeFpnLayerPlugin::enqueue(int batchSize,const void*const * inputs,
																void**			   outputs,
												  void* workspace,    cudaStream_t stream){
		assert(batchSize == 1); 
		/*if (out_height == in_height && out_width == in_width) {
			CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], totalElems * type2size(mDataType),
									   cudaMemcpyDeviceToDevice, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			return 0;}*/
		switch (mDataType)
		{
			case DataType::kFLOAT:
				forwardCpu<float>((const float*)inputs[0],
					(const float*)inputs[1],
					(const float*)inputs[2],
					(const float*)inputs[3],
					(const float*)inputs[4],
					(const float*)inputs[5],
					(const float*)inputs[6],
					(const float*)inputs[7],
					(const float*)inputs[8],
					(const float*)inputs[9],
					(float*)outputs[0],
					(float*)outputs[1],
					(float*)outputs[2],
					(float*)outputs[3],
					(float*)outputs[4],
					(float*)outputs[5],
					stream);
				//forwardGpu((const float *const *)inputs,(float *)outputs[0],stream);
				break;
			case DataType::kHALF:
				forwardCpu<__half>((const __half*)inputs[0],
					(const __half*)inputs[1],
					(const __half*)inputs[2],
					(const __half*)inputs[3],
					(const __half*)inputs[4],
					(const __half*)inputs[5],
					(const __half*)inputs[6],
					(const __half*)inputs[7],
					(const __half*)inputs[8],
					(const __half*)inputs[9],
					(__half*)outputs[0],
					(__half*)outputs[1],
					(__half*)outputs[2],
					(__half*)outputs[3],
					(__half*)outputs[4],
					(__half*)outputs[5],
					stream);
				break;
			case DataType::kINT8:
				forwardCpu<u_int8_t>((const u_int8_t*)inputs[0],
					(const u_int8_t*)inputs[1],
					(const u_int8_t*)inputs[2],
					(const u_int8_t*)inputs[3],
					(const u_int8_t*)inputs[4],
					(const u_int8_t*)inputs[5],
					(const u_int8_t*)inputs[6],
					(const u_int8_t*)inputs[7],
					(const u_int8_t*)inputs[8],
					(const u_int8_t*)inputs[9],
					(u_int8_t*)outputs[0],
					(u_int8_t*)outputs[1],
					(u_int8_t*)outputs[2],
					(u_int8_t*)outputs[3],
					(u_int8_t*)outputs[4],
					(u_int8_t*)outputs[5],
					stream);
				break;
			default:
				std::cerr << "error data type" << std::endl;
		}

		return 0;
	};

}
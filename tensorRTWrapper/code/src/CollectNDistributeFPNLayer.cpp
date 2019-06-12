
#include "generate_proposals_op_util_boxes.hpp"
#include "CollectNDistributeFPNLayer.h"
#include "box_transform.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;
using namespace std;

namespace nvinfer1 {

	int CollectNDistributeFPNLayerPlugin::initialize(){
		roi_min_level_ = 2; 
		roi_max_level_ = 5;

		roi_canonical_level_ = 4;
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
            m_ouputTotalCount += roiCountPerFPN_ * 4; // ?x4==W'
		m_ouputTotalCount += rpn_post_nms_topN_ * (4 + 1);
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, m_ouputTotalCount * sizeof(float),cudaHostAllocDefault));
		memset(mOutputBuffer, 0, m_ouputTotalCount * sizeof(float));
		return 0;
	}
	void CollectNDistributeFPNLayerPlugin::configureWithFormat(const Dims* inputDims, int nbInputs,
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
	Dims CollectNDistributeFPNLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
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
		if (index == 0){
			return DimsHW(rpn_post_nms_topN_, 4);}
		else if (index == 1){
			return DimsHW(roiCountPerFPN_ , 4);}
		else if (index == 2){
			return DimsHW(roiCountPerFPN_, 4);}
		else if (index == 3){
			return DimsHW(roiCountPerFPN_, 4);}
		else if (index == 4){
			return DimsHW(roiCountPerFPN_, 4);}
		else if (index == 5){
			return DimsHW(rpn_post_nms_topN_, 1);}
	}

	template <typename Dtype>
	void CollectNDistributeFPNLayerPlugin::forwardCpu(const Dtype* inputs, Dtype* outputs, cudaStream_t stream) {
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
        Dtype* inputData = (Dtype*)mInputBuffer;

        caffe2::ERArrXXf rois(m_proposal_num, 5);
        caffe2::EArrXf scores(m_proposal_num);
        int len = 0;

		for (int i = 0; i < num_rpn_lvls*2; i++) {
            if (i < num_rpn_lvls) {
                size = m_rpn_rois_fpnH[i] * m_rpn_rois_fpnW[i];
            } else {
                size = m_rpn_rois_probs_fpn_H[i] * m_rpn_rois_probs_fpn_W[i];
            }
            CUDA_CHECK(cudaMemcpyAsync(mInputBuffer, inputData, size * sizeof(Dtype), cudaMemcpyDeviceToHost, stream));

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
            if (i < num_rpn_lvls) {
                const float *roi_in = (float *) inputData;//bottom[i]->cpu_data();//Input(i);
                int n = m_rpn_rois_fpnH[i];//bottom[i]->shape(0);//roi_in.dim(0);
                Eigen::Map<const caffe2::ERArrXXf> roi(roi_in, n, 5);//roi_in.data<float>(),
                rois.block(len, 0, n, 5) = roi;
                len += n;
            }else{
                int n = m_rpn_rois_probs_fpn_H[i];//bottom[i]->shape(0);//roi_in.dim(0);

                const float *score_in = (float *)inputData;// auto& => Dtype*,  Input(num_rpn_lvls + i);
                // No need to squeeze, since we are reshaping when converting to Eigen
                // https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
                Eigen::Map<const caffe2::EArrXf> score(score_in, n);//score_in.data<float>()
                scores.segment(len, n) = score;
                len += n;
            }
            inputData += size;
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
            lvls = utils::MapRoIsToFpnLevels(rois_block,lvl_min, lvl_max, canon_scale--, canon_level,zero_level);
            //canon_scale
          }
          else{
            break;
          }
        */
        //   outputs[0].reshape(rois.shape)
        //   outputs[0].data[...] = rois
        Dtype* rois_out = (Dtype*)mOutputBuffer;//
		printf("rois.rows(): %d, rois.cols(): %d\n", rois.rows(), rois.cols());//rois shape == (1000,5)

		//Eigen::Map<caffe2::ERArrXXf> rois_out_mat(rois_out, rois.rows(), rois.cols());//->template mutable_data<float>()
		//rois_out_mat = rois;  
		//Reshape output_0 to rois.rows() x 4 
		int totalRoiCount = 0;
		for (int i = 0, j = 0; i < rois.rows() * rois.cols();) {
			totalRoiCount++;
			if (j % 5 != 0) rois_out[i++] = rois.data()[j++];
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
			float* roi_out = (float*)(rois_out+rpn_post_nms_topN_*4 + roiCountPerFPN_*4*i);
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
		int* rois_idx_restore_out = (int*)rois_out + rpn_post_nms_topN_ * 4 + roiCountPerFPN_ * 4 * 4;
		//rois_idx_restore_out->Resize(rois_idx_restore.size());
		printf("rois_idx_restore.size() : %d\n", rois_idx_restore.size());

		for (int i = 0; i < rois_idx_restore.size(); i++)
			rois_idx_restore_out[i] = rois_idx_restore.data()[i]; 

		CUDA_CHECK(cudaMemcpyAsync(outputs, mOutputBuffer, sizeof(float)* m_ouputTotalCount, cudaMemcpyHostToDevice, stream)); 
	}

	int CollectNDistributeFPNLayerPlugin::enqueue(int batchSize,const void*const * inputs,
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
				forwardCpu<float>((const float*)&inputs[0],
					(float*)&outputs[0],
					stream);
				//forwardGpu((const float *const *)inputs,(float *)outputs[0],stream);
				break;
			case DataType::kHALF:
				forwardCpu<__half>((const __half*)&inputs[0],
					(__half*)&outputs[0],
					stream);
				break;
			case DataType::kINT8:
				forwardCpu<u_int8_t>((const u_int8_t*)&inputs[0],
					(u_int8_t*)&outputs[0],
					stream);
				break;
			default:
				std::cerr << "error data type" << std::endl;
		}
		return 0;
	};
}
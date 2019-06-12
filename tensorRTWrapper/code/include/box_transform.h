//
// Created by lee on 19. 6. 12.
//

#ifndef DEMO_BOX_TRANSFORM_H
#define DEMO_BOX_TRANSFORM_H


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
        iota(std::begin(idxs), std::end(idxs), 0);
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

        CHECK_EQ(rois.rows(),lvls.rows());// if not, RoIs and lvls count mismatch
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


#endif //DEMO_BOX_TRANSFORM_H

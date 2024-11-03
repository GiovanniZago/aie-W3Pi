#include <adf.h>
#include "kernels.h"

using namespace adf;

class simpleGraph : public graph {
    private:
        kernel WTo3Pi_k;
        kernel filter_k;

    public:
        input_plio in_H;
        input_plio in_L;
        output_plio out;

        simpleGraph() {
            WTo3Pi_k = kernel::create(WTo3Pi);

            in_H = input_plio::create(plio_32_bits, "data/PuppiSignal_224_H.csv", 360);
            in_L = input_plio::create(plio_32_bits, "data/PuppiSignal_224_L.csv", 360);

            out = output_plio::create(plio_32_bits, "data/out.csv", 360);

            // PL inputs
            connect<stream>(in_H.out[0], WTo3Pi_k.in[0]);
            connect<stream>(in_L.out[0], WTo3Pi_k.in[1]);

            // PL outputs
            connect(WTo3Pi_k.out[0], out.in[0]);

            source(WTo3Pi_k) = "kernels.cpp";
            runtime<ratio>(WTo3Pi_k) = 1;
        }
};
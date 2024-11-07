#include <adf.h>
#include "kernels.h"

using namespace adf;

class simpleGraph : public graph {
    private:
        kernel unpacker_k;
        kernel filter_k;
        kernel combinatorial_k;

    public:
        input_plio in_H;
        input_plio in_L;
        output_plio out;

        simpleGraph() {
            unpacker_k = kernel::create(unpacker);
            filter_k = kernel::create(filter);
            combinatorial_k = kernel::create(combinatorial);

            in_H = input_plio::create(plio_32_bits, "data/PuppiSignal_224_H.csv", 360);
            in_L = input_plio::create(plio_32_bits, "data/PuppiSignal_224_L.csv", 360);

            out = output_plio::create(plio_32_bits, "data/out.csv", 360);

            // PL inputs
            connect<stream>(in_H.out[0], unpacker_k.in[0]);
            connect<stream>(in_L.out[0], unpacker_k.in[1]);

            // inner connections
            connect<stream>(unpacker_k.out[0], filter_k.in[0]);
            connect<stream>(unpacker_k.out[1], filter_k.in[1]);
            connect<stream>(filter_k.out[0], combinatorial_k.in[0]);
            connect<stream>(filter_k.out[1], combinatorial_k.in[1]);

            // PL outputs
            connect<stream>(combinatorial_k.out[0], out.in[0]);

            // sources and runtime ratios
            source(unpacker_k) = "kernels.cpp";
            source(filter_k) = "kernels.cpp";
            source(combinatorial_k) = "kernels.cpp";
            runtime<ratio>(unpacker_k) = 1;
            runtime<ratio>(filter_k) = 1;
            runtime<ratio>(combinatorial_k) = 1;
        }
};
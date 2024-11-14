#include <adf.h>
#include "kernels.h"

using namespace adf;

class simpleGraph : public graph {
    private:
        kernel unpack_and_filter_k;
        kernel isolation_k;
        kernel combinatorial_k;

    public:
        input_plio in;
        output_plio out;

        simpleGraph() {
            unpack_and_filter_k = kernel::create(unpack_and_filter);
            isolation_k = kernel::create(isolation);
            combinatorial_k = kernel::create(combinatorial);

            in = input_plio::create(plio_64_bits, "data/Puppi_224.csv", 360);
            out = output_plio::create(plio_32_bits, "data/out.csv", 360);

            // PL inputs
            connect<stream>(in.out[0], unpack_and_filter_k.in[0]);

            // inner connections
            connect<stream>(unpack_and_filter_k.out[0], isolation_k.in[0]);
            connect<stream>(unpack_and_filter_k.out[1], isolation_k.in[1]);
            connect<stream>(isolation_k.out[0], combinatorial_k.in[0]);
            connect<stream>(isolation_k.out[1], combinatorial_k.in[1]);

            // PL outputs
            connect<stream>(combinatorial_k.out[0], out.in[0]);

            // sources and runtime ratios
            source(unpack_and_filter_k) = "kernels.cpp";
            source(isolation_k) = "kernels.cpp";
            source(combinatorial_k) = "kernels.cpp";
            runtime<ratio>(unpack_and_filter_k) = 1;
            runtime<ratio>(isolation_k) = 1;
            runtime<ratio>(combinatorial_k) = 1;
        }
};
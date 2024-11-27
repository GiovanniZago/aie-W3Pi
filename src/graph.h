#include <adf.h>
#include "kernels.h"

using namespace adf;

class W3PiGraph : public graph {
    private:
        kernel unpack_filter_iso_k;
        kernel combinatorial_k;

    public:
        input_plio in;
        output_plio out;

        W3PiGraph() {
            unpack_filter_iso_k = kernel::create(unpack_filter_iso);
            combinatorial_k = kernel::create(combinatorial);

            in = input_plio::create(plio_64_bits, "data/Puppi_224.csv", 360);
            out = output_plio::create(plio_32_bits, "data/out.csv", 360);

            // PL inputs
            connect<stream>(in.out[0], unpack_filter_iso_k.in[0]);

            // inner connections
            connect<stream>(unpack_filter_iso_k.out[0], combinatorial_k.in[0]);
            connect<stream>(unpack_filter_iso_k.out[1], combinatorial_k.in[1]);

            // PL outputs
            connect<stream>(combinatorial_k.out[0], out.in[0]);

            // sources and runtime ratios
            source(unpack_filter_iso_k) = "kernels.cpp";
            source(combinatorial_k) = "kernels.cpp";
            runtime<ratio>(unpack_filter_iso_k) = 1;
            runtime<ratio>(combinatorial_k) = 1;
        }
};
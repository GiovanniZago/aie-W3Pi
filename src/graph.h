#include <adf.h>
#include "kernels.h"

using namespace adf;

class simpleGraph : public graph {
    private:
        kernel unpacker_k;
        kernel filter_k;

    public:
        input_plio in_H;
        input_plio in_L;
        output_plio eta_out;
        output_plio phi_out;
        output_plio pt_out;
        output_plio pdg_id_out;

        simpleGraph() {
            unpacker_k = kernel::create(unpacker);
            filter_k = kernel::create(filter);

            in_H = input_plio::create(plio_32_bits, "data/Puppi_fix104mod1_lines_H.csv", 360);
            in_L = input_plio::create(plio_32_bits, "data/Puppi_fix104mod1_lines_L.csv", 360);

            eta_out = output_plio::create(plio_32_bits, "data/eta_out.csv", 360);
            phi_out = output_plio::create(plio_32_bits, "data/phi_out.csv", 360);
            pt_out = output_plio::create(plio_32_bits, "data/pt_out.csv", 360);
            pdg_id_out = output_plio::create(plio_32_bits, "data/pdg_id_out.csv", 360);

            // PL inputs
            connect<stream>(in_H.out[0], unpacker_k.in[0]);
            connect<stream>(in_L.out[0], unpacker_k.in[1]);
            
            // inner connections
            connect(unpacker_k.out[0], filter_k.in[0]);
            connect(unpacker_k.out[1], filter_k.in[1]);
            connect(unpacker_k.out[2], filter_k.in[2]);
            connect(unpacker_k.out[3], filter_k.in[3]);

            // buffer sizes
            dimensions(unpacker_k.out[0]) = { EV_SIZE };
            dimensions(unpacker_k.out[1]) = { EV_SIZE };
            dimensions(unpacker_k.out[2]) = { EV_SIZE };
            dimensions(unpacker_k.out[3]) = { EV_SIZE };

            dimensions(filter_k.in[0]) = { EV_SIZE };
            dimensions(filter_k.in[1]) = { EV_SIZE };
            dimensions(filter_k.in[2]) = { EV_SIZE };
            dimensions(filter_k.in[3]) = { EV_SIZE };

            dimensions(filter_k.out[0]) = { EV_SIZE };
            dimensions(filter_k.out[1]) = { EV_SIZE };
            dimensions(filter_k.out[2]) = { EV_SIZE };
            dimensions(filter_k.out[3]) = { EV_SIZE };

            // PL outputs
            connect(filter_k.out[0], eta_out.in[0]);
            connect(filter_k.out[1], phi_out.in[0]);
            connect(filter_k.out[2], pt_out.in[0]);
            connect(filter_k.out[3], pdg_id_out.in[0]);


            source(unpacker_k) = "kernels.cpp";
            source(filter_k) = "kernels.cpp";
            runtime<ratio>(unpacker_k) = 1;
            runtime<ratio>(filter_k) = 1;
        }
};
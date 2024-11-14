#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include <aie_api/utils.hpp>
#include "kernels.h"
#include <math.h>

using namespace adf;

void unpack_and_filter(input_stream<int64> * __restrict in, output_stream<int32> * __restrict out0, output_stream<int32> * __restrict out1)
{   
    // data variables
    int64 data; 
    int32 pt, eta, phi, pdg_id;

    // filter pt and pdg id
    bool is_hig_pt, is_pdg_id;
    aie::vector<int32, N_HIG> is_filter;
    int16 is_filter_idx=0;

    for (int i=0; i<EV_SIZE; i++)
    {
        data = readincr(in);

        pt = ((1 << (PT_MSB + 1)) - 1) & data;
        eta = (data << 38) >> 52;
        phi = (data << 27) >> 53;
        pdg_id = ((1 << (PDG_ID_MSB + 1)) - 1) & (data >> 37);

        is_hig_pt = pt >= HIG_PT;
        is_pdg_id = (pdg_id == 2) | (pdg_id == 3) | (pdg_id == 4) | (pdg_id == 5);

        if ((is_filter_idx < N_HIG) & (is_hig_pt) & (is_pdg_id)) 
        {
            is_filter[is_filter_idx] = i + 1;
            is_filter_idx++;
        }

        // send data out
        writeincr(out0, pt);
        writeincr(out1, eta);
        writeincr(out0, phi);
        writeincr(out1, pdg_id);
    }

    writeincr(out0, is_filter);
}

void isolation(input_stream<int32> * __restrict in0, input_stream<int32> * __restrict in1, output_stream<int16> * __restrict out0, output_stream<int16> * __restrict out1)
{
    // data variables
    aie::vector<int16, V_SIZE> etas[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES];
    aie::vector<int32, N_HIG> is_filter_int32;
    aie::vector<int16, N_HIG> is_filter;

    // isolation variables
    int16 out_idx, in_idx;
    int16 pt_sum=0;
    aie::mask<N_HIG> is_iso;
    aie::vector<int16, V_SIZE> d_eta, d_phi;
    aie::vector<int16, V_SIZE> pt_to_sum;
    aie::vector<int32, V_SIZE> dr2;
    aie::vector<float, V_SIZE> dr2_float;
    aie::accum<acc48, V_SIZE> acc;
    aie::accum<accfloat, V_SIZE> acc_float;
    aie::mask<V_SIZE> is_ge_mindr2, is_le_maxdr2, pt_cut_mask;

    // variables for the two-pi check
    const aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();
    aie::mask<V_SIZE> is_gt_pi, is_lt_mpi;
    aie::vector<int16, V_SIZE> d_phi_ptwopi, d_phi_mtwopi;
    const aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
    const aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
    const aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
    const aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

    // output variables
    aie::vector<int16, N_HIG> pts_iso_filter = aie::broadcast<int16, N_HIG>(0);
    aie::vector<int16, N_HIG> etas_iso_filter = aie::broadcast<int16, N_HIG>(0);
    aie::vector<int16, N_HIG> phis_iso_filter = aie::broadcast<int16, N_HIG>(0);
    aie::vector<int16, N_HIG> pdg_ids_iso_filter = aie::broadcast<int16, N_HIG>(0);
    aie::vector<int16, N_HIG> is_iso_filter = aie::broadcast<int16, N_HIG>(0);

    // READ DATA 
    for (int i=0; i<P_BUNCHES; i++)
    {
        for (int j=0; j<V_SIZE; j++)
        {
            pts[i][j] = readincr(in0);
            etas[i][j] = readincr(in1);
            phis[i][j] = readincr(in0);
            pdg_ids[i][j] = readincr(in1);
        }
    }

    is_filter_int32 = readincr_v<N_HIG>(in0);
    is_filter = is_filter_int32.pack();
    aie::mask<N_HIG> is_filter_mask = aie::gt(is_filter, (int16) 0);
    int16 n_filter = is_filter_mask.count();
    bool skip_event = (n_filter < 3);

    for (int i=0; i<N_HIG; i++)
    {   
        if (skip_event) continue;
        if (!is_filter[i]) continue;

        pt_sum = 0;
        out_idx = (is_filter[i] - 1) / V_SIZE;
        in_idx = (is_filter[i] - 1) % V_SIZE;

        for (int k=0; k<P_BUNCHES; k++)
        chess_prepare_for_pipelining
        {
            d_eta = aie::sub(etas[out_idx][in_idx], etas[k]);

            d_phi = aie::sub(phis[out_idx][in_idx], phis[k]);
            d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
            d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
            is_gt_pi = aie::gt(d_phi, pi_vector);
            is_lt_mpi = aie::lt(d_phi, mpi_vector);
            d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
            d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

            acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
            acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
            dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
            dr2_float = aie::to_float(dr2, 0);
            acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
            dr2_float = acc_float.to_vector<float>(0);

            is_ge_mindr2 = aie::ge(dr2_float, MINDR2_FLOAT);
            is_le_maxdr2 = aie::le(dr2_float, MAXDR2_FLOAT);
            pt_cut_mask = is_ge_mindr2 & is_le_maxdr2;

            pt_to_sum = aie::select(zeros_vector, pts[i], pt_cut_mask); // select only the pts that fall in the desired range
            pt_sum += aie::reduce_add(pt_to_sum); // update the pt sum
        }

        if (pt_sum <= (pts[out_idx][in_idx] * MAX_ISO))
        {
            pts_iso_filter[i] = pts[out_idx][in_idx];
            etas_iso_filter[i] = etas[out_idx][in_idx];
            phis_iso_filter[i] = phis[out_idx][in_idx];
            pdg_ids_iso_filter[i] = pdg_ids[out_idx][in_idx];
            is_iso_filter[i] = is_filter[i];
        }
    }

    writeincr(out0, pts_iso_filter);    
    writeincr(out1, etas_iso_filter);    
    writeincr(out0, phis_iso_filter);    
    writeincr(out1, pdg_ids_iso_filter);    
    writeincr(out0, is_iso_filter);    
}

void combinatorial(input_stream<int16> * __restrict in0, input_stream<int16> * __restrict in1, output_stream<float> * __restrict out)
{
    // data variables
    aie::vector<int16, N_HIG> pts_iso_filter, etas_iso_filter, phis_iso_filter, pdg_ids_iso_filter, is_iso_filter;

    // READ DATA 
    pts_iso_filter = readincr_v<N_HIG>(in0);
    etas_iso_filter = readincr_v<N_HIG>(in1);
    phis_iso_filter = readincr_v<N_HIG>(in0);
    pdg_ids_iso_filter = readincr_v<N_HIG>(in1);
    is_iso_filter = readincr_v<N_HIG>(in0);

    // ang sep specific variables
    int16 d_eta, d_phi;
    int32 dr2;
    float dr2_float;
    int16 eta_hig_pt_target0, eta_hig_pt_target1, eta_hig_pt_target2;
    int16 phi_hig_pt_target0, phi_hig_pt_target1, phi_hig_pt_target2;
    int16 pt_hig_pt_target0, pt_hig_pt_target1, pt_hig_pt_target2;

    // triplet variables
    int16 charge_tot, charge0, charge1, charge2;
    int16 triplet_score = 0, best_triplet_score = 0;
    float mass0, mass1, mass2;
    float px0, py0, pz0, px1, py1, pz1, px2, py2, pz2;
    float e0, e1, e2;
    float px_tot, py_tot, pz_tot, e_tot;
    float x, sinh;
    float invariant_mass;

    aie::vector<float, 4> triplet = aie::zeros<float, 4>();

    aie::mask<N_HIG> is_iso_filter_mask = aie::gt(is_iso_filter, (int16) 0);
    int16 n_iso_filter = is_iso_filter_mask.count();
    bool skip_event = (n_iso_filter < 3);

    for (int i0=0; i0<N_HIG; i0++)
    {
        if (skip_event) continue;
        if (!is_iso_filter[i0]) continue;

        pt_hig_pt_target0 = pts_iso_filter[i0];
        eta_hig_pt_target0 = etas_iso_filter[i0];
        phi_hig_pt_target0 = phis_iso_filter[i0];

        for (int i1=0; i1<N_HIG; i1++)
        {
            if (i1 == i0) continue;
            if (!is_iso_filter[i1]) continue;

            d_eta = etas_iso_filter[i1] - eta_hig_pt_target0;
            d_phi = phis_iso_filter[i1] - phi_hig_pt_target0;
            d_phi = (d_phi <= PI) ? ((d_phi >= MPI) ? d_phi : d_phi + TWOPI) : d_phi + MTWOPI;

            dr2 = d_eta * d_eta + d_phi * d_phi;
            dr2_float = dr2 * F_CONV2;

            if (dr2_float < MINDR2_ANGSEP_FLOAT) continue;

            pt_hig_pt_target1 = pts_iso_filter[i1];
            eta_hig_pt_target1 = etas_iso_filter[i1];
            phi_hig_pt_target1 = phis_iso_filter[i1];

            for (int i2=0; i2<N_HIG; i2++)
            {
                if (i2 == i0) continue;
                if (i2 == i1) continue;
                if (!is_iso_filter[i2]) continue;

                d_eta = etas_iso_filter[i2] - eta_hig_pt_target1;
                d_phi = phis_iso_filter[i2] - phi_hig_pt_target1;
                d_phi = (d_phi <= PI) ? ((d_phi >= MPI) ? d_phi : d_phi + TWOPI) : d_phi + MTWOPI;

                dr2 = d_eta * d_eta + d_phi * d_phi;
                dr2_float = dr2 * F_CONV2;

                if (dr2_float < MINDR2_ANGSEP_FLOAT) continue;

                pt_hig_pt_target2 = pts_iso_filter[i2];
                eta_hig_pt_target2 = etas_iso_filter[i2];
                phi_hig_pt_target2 = phis_iso_filter[i2];

                charge0 = (pdg_ids_iso_filter[i0] >= 4) ? ((pdg_ids_iso_filter[i0] == 4) ? -1 : 1) : ((pdg_ids_iso_filter[i0] == 2) ? -1 : 1);
                charge1 = (pdg_ids_iso_filter[i1] >= 4) ? ((pdg_ids_iso_filter[i1] == 4) ? -1 : 1) : ((pdg_ids_iso_filter[i1] == 2) ? -1 : 1);
                charge2 = (pdg_ids_iso_filter[i2] >= 4) ? ((pdg_ids_iso_filter[i2] == 4) ? -1 : 1) : ((pdg_ids_iso_filter[i2] == 2) ? -1 : 1);

                charge_tot = charge0 + charge1 + charge2;

                if ((charge_tot != 1) & (charge_tot != -1)) continue;

                mass0 = (charge0 > 0) ? MASS_M : MASS_P;
                px0 = pt_hig_pt_target0 * PT_CONV * aie::cos(phi_hig_pt_target0 * F_CONV);
                py0 = pt_hig_pt_target0 * PT_CONV * aie::sin(phi_hig_pt_target0 * F_CONV);
                x = eta_hig_pt_target0 * F_CONV;
                sinh = x + ((x * x * x) / 6);
                pz0 = pt_hig_pt_target0 * PT_CONV * sinh;
                e0 = aie::sqrt(px0 * px0 + py0 * py0 + pz0 * pz0 + mass0 * mass0);
            
                mass1 = (charge1 > 0) ? MASS_M : MASS_P;
                px1 = pt_hig_pt_target1 * PT_CONV * aie::cos(phi_hig_pt_target1 * F_CONV);
                py1 = pt_hig_pt_target1 * PT_CONV * aie::sin(phi_hig_pt_target1 * F_CONV);
                x = eta_hig_pt_target1 * F_CONV;
                sinh = x + ((x * x * x) / 6);
                pz1 = pt_hig_pt_target1 * PT_CONV * sinh;
                e1 = aie::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + mass1 * mass1);

                mass2 = (charge2 > 0) ? MASS_M : MASS_P;
                px2 = pt_hig_pt_target2 * PT_CONV * aie::cos(phi_hig_pt_target2 * F_CONV);
                py2 = pt_hig_pt_target2 * PT_CONV * aie::sin(phi_hig_pt_target2 * F_CONV);
                x = eta_hig_pt_target2 * F_CONV;
                sinh = x + ((x * x * x) / 6);
                pz2 = pt_hig_pt_target2 * PT_CONV * sinh;
                e2 = aie::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + mass2 * mass2);

                px_tot = px0 + px1 + px2;
                py_tot = py0 + py1 + py2;
                pz_tot = pz0 + pz1 + pz2;
                e_tot = e0 + e1 + e2;

                invariant_mass = aie::sqrt(e_tot * e_tot - px_tot * px_tot - py_tot * py_tot - pz_tot * pz_tot);

                if ((invariant_mass < MIN_MASS) | (invariant_mass > MAX_MASS)) continue;

                triplet_score = pt_hig_pt_target0 + pt_hig_pt_target1 + pt_hig_pt_target2;

                if (triplet_score > best_triplet_score)
                {   
                    best_triplet_score = triplet_score;
                    triplet[0] = is_iso_filter[i0] - 1;
                    triplet[1] = is_iso_filter[i1] - 1;
                    triplet[2] = is_iso_filter[i2] - 1;
                    triplet[3] = invariant_mass;
                }
            }
        }
    }

    writeincr(out, triplet);
}

// void isolation(input_buffer<int16> & __restrict in, output_stream<int16> * __restrict out0, output_stream<int16> * __restrict out1)
// {
//     // data variables
//     aie::vector<int16, V_SIZE> etas[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES], is_filter[P_BUNCHES];
    
//     // auxiliary variables
//     const aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();

//     // input iterator
//     auto in_Itr = aie::begin(in);

//     // READ DATA 
//     for (int i=0; i<P_BUNCHES; i++)
//     {
//         for (int j=0; j<V_SIZE; j++)
//         {
//             pts[i][j] = *in_Itr++;
//             etas[i][j] = *in_Itr++;
//             phis[i][j] = *in_Itr++;
//             pdg_ids[i][j] = *in_Itr++;
//             is_filter[i][j] = *in_Itr++;
//         }
//     }

//     #if defined(__X86SIM__) && defined(__X86DEBUG__)
//     printf("pts[0]:\n");
//     aie::print(pts[0]);
//     printf("\n");
//     #endif
//     #if defined(__X86SIM__) && defined(__X86DEBUG__)
//     printf("etas[0]:\n");
//     aie::print(etas[0]);
//     printf("\n");
//     #endif
//     #if defined(__X86SIM__) && defined(__X86DEBUG__)
//     printf("phis[0]:\n");
//     aie::print(phis[0]);
//     printf("\n");
//     #endif
//     #if defined(__X86SIM__) && defined(__X86DEBUG__)
//     printf("pdg_id[0]:\n");
//     aie::print(pdg_ids[0]);
//     printf("\n");
//     #endif
//     #if defined(__X86SIM__) && defined(__X86DEBUG__)
//     printf("is_filter[0]:\n");
//     aie::print(is_filter[0]);
//     printf("\n");
//     #endif
//     #if defined(__X86SIM__) && defined(__X86DEBUG__)
//     printf("\n");
//     #endif
        
//     // isolation variables
//     int16 pt_sum;
//     aie::vector<int16, V_SIZE> d_eta, d_phi;
//     aie::vector<int16, V_SIZE> pt_to_sum;
//     aie::vector<int32, V_SIZE> dr2;
//     aie::vector<float, V_SIZE> dr2_float;
//     aie::accum<acc48, V_SIZE> acc;
//     aie::accum<accfloat, V_SIZE> acc_float;
//     aie::mask<V_SIZE> is_ge_mindr2, is_le_maxdr2, pt_cut_mask;
//     aie::vector<int16, V_SIZE> pt_sums = aie::zeros<int16, V_SIZE>();
//     aie::vector<float, V_SIZE> pt_sums_float, pts_float, pts_maxiso_float;

//     // variables for the two-pi check
//     aie::mask<V_SIZE> is_gt_pi, is_lt_mpi;
//     aie::vector<int16, V_SIZE> d_phi_ptwopi, d_phi_mtwopi;
//     const aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
//     const aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
//     const aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
//     const aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

//     for (int i=0; i<P_BUNCHES; i++)
//     {
//         for (int j=0; j<V_SIZE; j++)
//         {
//             // skip particle if it has not passed the pt and pdg_id cut
//             if (!is_filter[i][j]) continue;

//             pt_sum = 0;

//             for (int k=0; k<P_BUNCHES; k++)
//             chess_prepare_for_pipelining
//             {
//                 d_eta = aie::sub(etas[i][j], etas[k]);

//                 d_phi = aie::sub(phis[i][j], phis[k]);
//                 d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
//                 d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
//                 is_gt_pi = aie::gt(d_phi, pi_vector);
//                 is_lt_mpi = aie::lt(d_phi, mpi_vector);
//                 d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
//                 d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

//                 acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
//                 acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
//                 dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
//                 dr2_float = aie::to_float(dr2, 0);
//                 acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
//                 dr2_float = acc_float.to_vector<float>(0);

//                 is_ge_mindr2 = aie::ge(dr2_float, MINDR2_FLOAT);
//                 is_le_maxdr2 = aie::le(dr2_float, MAXDR2_FLOAT);
//                 pt_cut_mask = is_ge_mindr2 & is_le_maxdr2;

//                 pt_to_sum = aie::select(zeros_vector, pts[i], pt_cut_mask); // select only the pts that fall in the desired range
//                 pt_sum += aie::reduce_add(pt_to_sum); // update the pt sum
//             }

//             pt_sums[j] = pt_sum;
//         }
        
//         pt_sums_float = aie::to_float(pt_sums, 0);
//         pts_float = aie::to_float(pts[i], 0);
//         acc_float = aie::mul(pts_float, MAX_ISO);
//         pts_maxiso_float = acc_float.to_vector<float>(0);
//         aie::mask<V_SIZE> is_iso_mask = aie::le(pt_sums_float, pts_maxiso_float);
//         aie::mask<V_SIZE> is_filter_mask = aie::eq(is_filter[i], (int16) 1);
//         aie::mask<V_SIZE> is_iso_filter_mask = is_iso_mask & is_filter_mask;

//         phis[i] = aie::select(zeros_vector, phis[i], is_iso_filter_mask); 
//         etas[i] = aie::select(zeros_vector, etas[i], is_iso_filter_mask); 
//         pts[i] = aie::select(zeros_vector, pts[i], is_iso_filter_mask); 
//         pdg_ids[i] = aie::select(zeros_vector, pdg_ids[i], is_iso_filter_mask); 

//         writeincr(out0, pts[i]);
//         writeincr(out1, etas[i]);
//         writeincr(out0, phis[i]);
//         writeincr(out1, pdg_ids[i]);
//     }
// }

// void isolation(input_stream<int32> * __restrict in0, input_stream<int32> * __restrict in1, output_stream<int32> * __restrict out0, output_stream<int32> * __restrict out1)
// {
//     // data variables
//     int16 pts[P_BUNCHES][V_SIZE], etas[P_BUNCHES][V_SIZE], phis[P_BUNCHES][V_SIZE], pdg_ids[P_BUNCHES][V_SIZE];
//     bool is_filter[P_BUNCHES][V_SIZE];
    
//     // auxiliary variables
//     const aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();

//     // READ DATA 
//     for (int i=0; i<P_BUNCHES; i++)
//     {
//         for (int j=0; j<V_SIZE; j++)
//         {
//             pts[i][j] = readincr(in0);
//             etas[i][j] = readincr(in1);
//             phis[i][j] = readincr(in0);
//             pdg_ids[i][j] = readincr(in1);
//             is_filter[i][j] = readincr(in0);
//         }
//     }

//     // isolation variables
//     bool is_iso[P_BUNCHES][V_SIZE] = { 0 };
//     bool is_filter_iso;
//     int16 pt_sum;

//     // variables for the two-pi check
//     const aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
//     const aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
//     const aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
//     const aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

//     for (int i=0; i<P_BUNCHES; i++)
//     {
//         for (int j=0; j<V_SIZE; j++)
//         {
//             // skip particle if it has not passed the pt and pdg_id cut
//             if (!is_filter[i][j]) continue;

//             pt_sum = 0;

//             for (int k=0; k<P_BUNCHES; k++)
//             chess_prepare_for_pipelining
//             {
//                 aie::vector<int16, V_SIZE> etas_vec = aie::load_v<V_SIZE>(etas[k]);
//                 aie::vector<int16, V_SIZE> d_eta = aie::sub(etas[i][j], etas_vec);

//                 aie::vector<int16, V_SIZE> phis_vec = aie::load_v<V_SIZE>(phis[k]);
//                 aie::vector<int16, V_SIZE> d_phi = aie::sub(phis[i][j], phis_vec);
//                 aie::vector<int16, V_SIZE> d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
//                 aie::vector<int16, V_SIZE> d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
//                 aie::mask<V_SIZE> is_gt_pi = aie::gt(d_phi, pi_vector);
//                 aie::mask<V_SIZE> is_lt_mpi = aie::lt(d_phi, mpi_vector);
//                 d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
//                 d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

//                 aie::accum<acc48, V_SIZE> acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
//                 acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
//                 aie::vector<int32, V_SIZE> dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
//                 aie::vector<float, V_SIZE> dr2_float = aie::to_float(dr2, 0);
//                 aie::accum<accfloat, V_SIZE> acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
//                 dr2_float = acc_float.to_vector<float>(0);

//                 aie::mask<V_SIZE> is_ge_mindr2 = aie::ge(dr2_float, MINDR2_FLOAT);
//                 aie::mask<V_SIZE> is_le_maxdr2 = aie::le(dr2_float, MAXDR2_FLOAT);
//                 aie::mask<V_SIZE> pt_cut_mask = is_ge_mindr2 & is_le_maxdr2;
                
//                 aie::vector<int16, V_SIZE> pts_vec = aie::load_v<V_SIZE>(pts[k]);
//                 aie::vector<int16, V_SIZE> pt_to_sum = aie::select(zeros_vector, pts_vec, pt_cut_mask); // select only the pts that fall in the desired range
//                 pt_sum += aie::reduce_add(pt_to_sum); // update the pt sum
//             }

//             is_iso[i][j] = (pt_sum <= (pts[i][j] * MAX_ISO));
//         }
//     }

//     for (int i=0; i<P_BUNCHES; i++)
//     {
//         for (int j=0; j<V_SIZE; j++)
//         {
//             is_filter_iso = is_filter[i][j] & is_iso[i][j];
//             pts[i][j] = is_filter_iso ? pts[i][j] : 0;
//             etas[i][j] = is_filter_iso ? etas[i][j] : 0;
//             phis[i][j] = is_filter_iso ? phis[i][j] : 0;
//             pdg_ids[i][j] = is_filter_iso ? pdg_ids[i][j] : 0;

//             writeincr(out0, (int32) pts[i][j]);
//             writeincr(out1, (int32) etas[i][j]);
//             writeincr(out0, (int32) phis[i][j]);
//             writeincr(out1, (int32) pdg_ids[i][j]);
//             writeincr(out0, (int32) is_filter_iso);
//         }
//     }
// }

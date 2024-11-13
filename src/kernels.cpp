#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include <aie_api/utils.hpp>
#include "kernels.h"
#include <math.h>

using namespace adf;

void unpack_and_filter(input_stream<int64> * __restrict in, output_stream<int16> * __restrict out0, output_stream<int16> * __restrict out1)
{   
    // data variables
    int64 data; 
    int16 pt, eta, phi, pdg_id;
    aie::vector<int16, V_SIZE> pt_vec, eta_vec, phi_vec, pdg_id_vec;

    // auxiliary variables
    int64 foo, foo1, foo2;

    // filter pt and pdg id
    bool is_hig_pt, is_pdg_id, is_filter;
    int32 filter_mask_int=0;

    for (int i=0; i<P_BUNCHES; i++)
    {
        filter_mask_int = 0;

        for (int j=0; j<V_SIZE; j++)
        chess_prepare_for_pipelining
        {
            data = readincr(in);

            pt = ((1 << (PT_MSB + 1)) - 1) & data;
            pt_vec[j] = pt;
            is_hig_pt = pt >= HIG_PT;

            eta = (data << 38) >> 52;
            eta_vec[j] = eta;

            phi = (data << 27) >> 53;
            phi_vec[j] = phi;

            pdg_id = ((1 << (PDG_ID_MSB + 1)) - 1) & (data >> 37);
            pdg_id_vec[j] = pdg_id;
            is_pdg_id = (pdg_id == 2) | (pdg_id == 3) | (pdg_id == 4) | (pdg_id == 5);

            is_filter = is_hig_pt & is_pdg_id;
            filter_mask_int |= is_filter ? (1 << j) : 0;
        }

        // send data out
        writeincr(out0, pt_vec);
        writeincr(out1, eta_vec);
        writeincr(out0, phi_vec);
        writeincr(out1, pdg_id_vec);
        writeincr(out0, filter_mask_int);
    }
}

void isolation(input_stream<int16> * __restrict in0, input_stream<int16> * __restrict in1, output_stream<int16> * __restrict out0, output_stream<int16> * __restrict out1)
{
    // data variables
    aie::vector<int16, V_SIZE> etas[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES];
    int32 filter_mask_int[P_BUNCHES];
    aie::mask<V_SIZE> filter_mask;
    
    // auxiliary variables
    const aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();

    // READ DATA 
    for (int i=0; i<P_BUNCHES; i++)
    {
        pts[i] = readincr_v<V_SIZE>(in0);
        etas[i] = readincr_v<V_SIZE>(in1);
        phis[i] = readincr_v<V_SIZE>(in0);
        pdg_ids[i] = readincr_v<V_SIZE>(in1);
        filter_mask_int[i] = readincr(in0);
    }

    // isolation variables
    int16 pt_sum;
    aie::vector<int16, V_SIZE> d_eta, d_phi;
    aie::vector<int16, V_SIZE> pt_to_sum;
    aie::vector<int32, V_SIZE> dr2;
    aie::vector<float, V_SIZE> dr2_float;
    aie::accum<acc48, V_SIZE> acc;
    aie::accum<accfloat, V_SIZE> acc_float;
    aie::mask<V_SIZE> is_ge_mindr2, is_le_maxdr2, pt_cut_mask;
    aie::vector<int16, V_SIZE> pt_sums = aie::zeros<int16, V_SIZE>();
    aie::vector<float, V_SIZE> pt_sums_float, pts_float, pts_maxiso_float;

    // variables for the two-pi check
    aie::mask<V_SIZE> is_gt_pi, is_lt_mpi;
    aie::vector<int16, V_SIZE> d_phi_ptwopi, d_phi_mtwopi;
    const aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
    const aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
    const aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
    const aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

    for (int i=0; i<P_BUNCHES; i++)
    {
        filter_mask = aie::mask<V_SIZE>::from_uint32(filter_mask_int[i]);

        for (int j=0; j<V_SIZE; j++)
        {
            // skip particle if it has not passed the pt and pdg_id cut
            if (!filter_mask.test(j)) continue;

            pt_sum = 0;

            for (int k=0; k<P_BUNCHES; k++)
            chess_prepare_for_pipelining
            {
                d_eta = aie::sub(etas[i][j], etas[k]);

                d_phi = aie::sub(phis[i][j], phis[k]);
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

            pt_sums[j] = pt_sum;
        }
        
        pt_sums_float = aie::to_float(pt_sums, 0);
        pts_float = aie::to_float(pts[i], 0);
        acc_float = aie::mul(pts_float, MAX_ISO);
        pts_maxiso_float = acc_float.to_vector<float>(0);
        aie::mask<V_SIZE> iso_mask = aie::le(pt_sums_float, pts_maxiso_float);
        aie::mask<V_SIZE> total_mask = iso_mask & filter_mask;

        phis[i] = aie::select(zeros_vector, phis[i], total_mask); 
        etas[i] = aie::select(zeros_vector, etas[i], total_mask); 
        pts[i] = aie::select(zeros_vector, pts[i], total_mask); 
        pdg_ids[i] = aie::select(zeros_vector, pdg_ids[i], total_mask); 

        writeincr(out0, pts[i]);
        writeincr(out1, etas[i]);
        writeincr(out0, phis[i]);
        writeincr(out1, pdg_ids[i]);

        #if defined(__X86SIM__) && defined(__X86DEBUG__)
        printf("\n");
        #endif
    }
}

void combinatorial(input_stream<int16> * __restrict in0, input_stream<int16> * __restrict in1, output_stream<float> * __restrict out)
{
    // data variables
    aie::vector<int16, V_SIZE> etas[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES];
    // auxiliary variables
    aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();

    // READ DATA 
    for (int i=0; i<P_BUNCHES; i++)
    {
        pts[i] = readincr_v<V_SIZE>(in0);
        etas[i] = readincr_v<V_SIZE>(in1);
        phis[i] = readincr_v<V_SIZE>(in0);
        pdg_ids[i] = readincr_v<V_SIZE>(in1);
    }

    // ANGULAR SEPARATION OF HIGH PT PARTICLES TO FIND THE TRIPLET
    // variables to compute required quantities
    int16 eta_cur, phi_cur, pt_cur, pt_sum;
    aie::vector<int16, V_SIZE> d_eta, d_phi;
    aie::vector<int16, V_SIZE> pt_to_sum;
    aie::vector<int32, V_SIZE> dr2;
    aie::vector<float, V_SIZE> dr2_float;
    aie::accum<acc48, V_SIZE> acc_d_eta2, acc_dr2;
    aie::accum<accfloat, V_SIZE> acc_dr2_float;
    aie::mask<V_SIZE> is_ge_mindr2, is_le_maxdr2, pt_cut_mask;

    // variables for the two-pi check
    aie::mask<V_SIZE> is_gt_pi, is_lt_mpi;
    aie::vector<int16, V_SIZE> d_phi_ptwopi, d_phi_mtwopi;
    aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
    aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
    aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
    aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

    // ang sep specific variables
    int16 hig_target_idx0, hig_target_idx1, hig_target_idx2;
    int16 eta_hig_pt_target0, eta_hig_pt_target1, eta_hig_pt_target2;
    int16 phi_hig_pt_target0, phi_hig_pt_target1, phi_hig_pt_target2;
    int16 pt_hig_pt_target0, pt_hig_pt_target1, pt_hig_pt_target2;
    aie::mask<V_SIZE> mask_hig_pt[P_BUNCHES], mask_hig_pt_cur0[P_BUNCHES], mask_hig_pt_cur1[P_BUNCHES];
    aie::mask<V_SIZE> angsep0[P_BUNCHES], angsep1[P_BUNCHES];

    int16 d_eta_scalar, d_phi_scalar;
    int32 dr2_scalar;
    float dr2_float_scalar;

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

    for (int i=0; i<P_BUNCHES; i++)
    {   
        mask_hig_pt[i] = aie::ge(pts[i], HIG_PT);

        for (int j=0; j<V_SIZE; j++)
        {   
            if (!mask_hig_pt[i].test(j)) continue;

            hig_target_idx0 = j;
            eta_hig_pt_target0 = etas[i][hig_target_idx0];
            phi_hig_pt_target0 = phis[i][hig_target_idx0];
            pt_hig_pt_target0 = pts[i][hig_target_idx0];

            mask_hig_pt_cur0[i] = mask_hig_pt[i];
            mask_hig_pt_cur0[i].clear(hig_target_idx0);

            for (int k=0; k<P_BUNCHES; k++)
            {
                d_eta = aie::sub(eta_hig_pt_target0, etas[k]);

                d_phi = aie::sub(phi_hig_pt_target0, phis[k]);
                d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                is_gt_pi = aie::gt(d_phi, pi_vector);
                is_lt_mpi = aie::lt(d_phi, mpi_vector);
                d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                acc_d_eta2 = aie::mul_square(d_eta); // acc = d_eta ^ 2
                acc_dr2 = aie::mac_square(acc_d_eta2, d_phi); // acc = acc + d_phi ^ 2
                dr2 = acc_dr2.to_vector<int32>(0); // convert accumulator into vector
                dr2_float = aie::to_float(dr2, 0);
                acc_dr2_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                dr2_float = acc_dr2_float.to_vector<float>(0);

                is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                angsep0[k] = is_ge_mindr2 & mask_hig_pt_cur0[k];

                for (int jj=0; jj<V_SIZE; jj++)
                {
                    if (!angsep0[k].test(jj)) continue;

                    hig_target_idx1 = jj;
                    eta_hig_pt_target1 = etas[k][hig_target_idx1];
                    phi_hig_pt_target1 = phis[k][hig_target_idx1];
                    pt_hig_pt_target1 = pts[k][hig_target_idx1];

                    mask_hig_pt_cur1[k] = mask_hig_pt_cur0[k];
                    mask_hig_pt_cur1[k].clear(hig_target_idx1);

                    for (int kk=0; kk<P_BUNCHES; kk++)
                    {
                        d_eta = aie::sub(eta_hig_pt_target1, etas[kk]);

                        d_phi = aie::sub(phi_hig_pt_target1, phis[kk]);
                        d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                        d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                        is_gt_pi = aie::gt(d_phi, pi_vector);
                        is_lt_mpi = aie::lt(d_phi, mpi_vector);
                        d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                        d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                        acc_d_eta2 = aie::mul_square(d_eta); // acc = d_eta ^ 2
                        acc_dr2 = aie::mac_square(acc_d_eta2, d_phi); // acc = acc + d_phi ^ 2
                        dr2 = acc_dr2.to_vector<int32>(0); // convert accumulator into vector
                        dr2_float = aie::to_float(dr2, 0);
                        acc_dr2_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                        dr2_float = acc_dr2_float.to_vector<float>(0);

                        is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                        angsep1[kk] = is_ge_mindr2 & mask_hig_pt_cur1[kk];

                        for (int jjj=0; jjj<V_SIZE; jjj++)
                        {
                            if (!angsep1[kk].test(jjj)) continue;

                            hig_target_idx2 = jjj;
                            eta_hig_pt_target2 = etas[kk][hig_target_idx2];
                            phi_hig_pt_target2 = phis[kk][hig_target_idx2];
                            pt_hig_pt_target2 = pts[kk][hig_target_idx2];

                            d_eta_scalar = eta_hig_pt_target2 - eta_hig_pt_target0;
                            d_phi_scalar = phi_hig_pt_target2 - phi_hig_pt_target0;
                            d_phi_scalar = (d_phi_scalar <= PI) ? ((d_phi_scalar >= MPI) ? d_phi_scalar : d_phi_scalar + TWOPI) : d_phi_scalar + MTWOPI;

                            dr2_scalar = d_eta_scalar * d_eta_scalar + d_phi_scalar * d_phi_scalar;
                            dr2_float_scalar = dr2_scalar * F_CONV2;

                            if (dr2_float_scalar >= MINDR2_ANGSEP_FLOAT)
                            {
                                charge0 = (pdg_ids[i][hig_target_idx0] >= 4) ? ((pdg_ids[i][hig_target_idx0] == 4) ? -1 : 1) : ((pdg_ids[i][hig_target_idx0] == 2) ? -1 : 1);
                                charge1 = (pdg_ids[k][hig_target_idx1] >= 4) ? ((pdg_ids[k][hig_target_idx1] == 4) ? -1 : 1) : ((pdg_ids[k][hig_target_idx1] == 2) ? -1 : 1);
                                charge2 = (pdg_ids[kk][hig_target_idx2] >= 4) ? ((pdg_ids[kk][hig_target_idx2] == 4) ? -1 : 1) : ((pdg_ids[kk][hig_target_idx2] == 2) ? -1 : 1);

                                charge_tot = charge0 + charge1 + charge2;

                                if ((charge_tot == 1) | (charge_tot == -1))
                                {
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

                                    if ((invariant_mass >= MIN_MASS) && (invariant_mass <= MAX_MASS))
                                    {
                                        triplet_score = pt_hig_pt_target0 + pt_hig_pt_target1 + pt_hig_pt_target2;
                                        
                                        if (triplet_score > best_triplet_score)
                                        {   
                                            best_triplet_score = triplet_score;
                                            triplet[0] = i * V_SIZE + hig_target_idx0;
                                            triplet[1] = k * V_SIZE + hig_target_idx1;
                                            triplet[2] = kk * V_SIZE + hig_target_idx2;
                                            triplet[3] = invariant_mass;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    writeincr(out, triplet);
}
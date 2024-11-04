#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include <aie_api/utils.hpp>
#include "kernels.h"
#include <math.h>

using namespace adf;

void WTo3Pi(input_stream<int32> * __restrict in_H, input_stream<int32> * __restrict in_L, output_stream<float> * __restrict out)
{   
    // data variables
    aie::vector<int32, V_SIZE> data_H[P_BUNCHES], data_L[P_BUNCHES]; 
    aie::vector<int16, V_SIZE> etas[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES];

    // auxiliary variables
    aie::vector<int32, V_SIZE> foo, foo1, foo2;
    aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();

    for (int i=0; i<P_BUNCHES; i++)
    {
        data_H[i] = readincr_v<V_SIZE>(in_H);
        data_L[i] = readincr_v<V_SIZE>(in_L);
    }

    for (int i=0; i<P_BUNCHES; i++)
    chess_prepare_for_pipelining
    {
        // unpack pt as an unsigned int
        foo = aie::bit_and((int32)((1 << (PT_MSB + 1)) - 1), data_L[i]);
        // halve the size from 32b to 16b
        pts[i] = foo.pack();

        // unpack eta as singed int
        // align MSB of eta to the MSB of the 32b word
        // since MSB of eta is on the 26th bit, we need a left shift of 6 bits
        foo = aie::upshift(data_L[i], 6); 
        // donwshift by 20 bits in order to align the LSB of eta to the LSB of 
        // the 32b word, preserving the sign
        foo = aie::downshift(foo, 20);
        // halve the size from 32b to 16b
        etas[i] = foo.pack();

        // unpack phi as signed int (more tricky, because phi hops the two 32b words)
        // isolate the bits of phi in the Lower 32b word
        foo1 = aie::downshift(data_L[i], PHI_SHIFT_L);
        foo1 = aie::bit_and((int32)((1 << (PHI_MSB_L + 1)) - 1), foo1);
        // isolate the bits of phi in the Higher 32b word
        foo2 = aie::bit_and((int32)((1 << (PHI_MSB_H + 1)) - 1), data_H[i]); 

        for (int j=0; j<V_SIZE; j++)
        {   
            foo[j] = (foo2[j] << 6) | foo1[j];
        }
        
        // align MSB of phi to the MSB of the 32b word
        foo = aie::upshift(foo, 21); 
        // align LSB of phi to the LSB of the 32b word, preserving the sign
        foo = aie::downshift(foo, 21);
        // halve the size from 32b to 16b
        phis[i] = foo.pack();


        // unpack pdg_id as an unsigned int
        foo = aie::downshift(data_H[i], PDG_ID_SHIFT);
        foo = aie::bit_and((int32)((1 << (PDG_ID_MSB + 1)) - 1), foo);
        // halve the size from 32b to 16b
        pdg_ids[i] = foo.pack();
    }
    
    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("first eta chunk:\n");
    aie::print(etas[0]);
    printf("\n");
    #endif

    // FILTER PDG ID AND ISOLATION, DIVIDE INTO PT CATEGORIES
    // pdg id variables
    aie::mask<V_SIZE> pdg_id_tot_mask[P_BUNCHES], pdg_id_mask1, pdg_id_mask2, pdg_id_mask3, pdg_id_mask4;

    // isolation variables
    int16 eta_cur, phi_cur, pt_cur, pt_sum;
    aie::vector<int16, V_SIZE> d_eta, d_phi;
    aie::vector<int16, V_SIZE> pt_to_sum;
    aie::vector<int32, V_SIZE> dr2;
    aie::vector<float, V_SIZE> dr2_float;
    aie::accum<acc48, V_SIZE> acc;
    aie::accum<accfloat, V_SIZE> acc_float;
    aie::mask<V_SIZE> is_ge_mindr2, is_le_maxdr2, pt_cut_mask;
    aie::mask<V_SIZE> iso_mask[P_BUNCHES], filter_mask[P_BUNCHES];

    // variables for the two-pi check
    aie::mask<V_SIZE> is_gt_pi, is_lt_mpi;
    aie::vector<int16, V_SIZE> d_phi_ptwopi, d_phi_mtwopi;
    aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
    aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
    aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
    aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

    // pt groups variables
    aie::mask<V_SIZE> mask_hig_pt[P_BUNCHES];

    for (int i=0; i<P_BUNCHES; i++)
    {
        pdg_id_mask1 = aie::eq((int16) 0b010, pdg_ids[i]); // 2
        pdg_id_mask2 = aie::eq((int16) 0b011, pdg_ids[i]); // 3
        pdg_id_mask3 = aie::eq((int16) 0b100, pdg_ids[i]); // 4
        pdg_id_mask4 = aie::eq((int16) 0b101, pdg_ids[i]); // 5

        pdg_id_mask1 = pdg_id_mask1 | pdg_id_mask2;
        pdg_id_mask3 = pdg_id_mask3 | pdg_id_mask4;
        pdg_id_tot_mask[i] = pdg_id_mask1 | pdg_id_mask3;

        for (int j=0; j<V_SIZE; j++)
        {
            eta_cur = etas[i][j];
            phi_cur = phis[i][j];
            pt_cur = pts[i][j];
            pt_sum = 0;
            iso_mask[i].clear(j);

            for (int k=0; k<P_BUNCHES; k++)
            {
                d_eta = aie::sub(eta_cur, etas[k]);

                d_phi = aie::sub(phi_cur, phis[k]);
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

            if (pt_sum <= (MAX_ISO * pt_cur)) iso_mask[i].set(j);
        }

        filter_mask[i] = iso_mask[i] & pdg_id_tot_mask[i];

        etas[i] = aie::select(zeros_vector, etas[i], filter_mask[i]); 
        phis[i] = aie::select(zeros_vector, phis[i], filter_mask[i]); 
        pts[i] = aie::select(zeros_vector, pts[i], filter_mask[i]); 
        pdg_ids[i] = aie::select(zeros_vector, pdg_ids[i], filter_mask[i]); 

        // assign masks for pt groups
        mask_hig_pt[i] = aie::ge(pts[i], HIG_PT);
    }

    // ANGULAR SEPARATION OF HIGH PT PARTICLES TO FIND THE TRIPLET
    // ang sep variables
    int16 hig_target_idx0, hig_target_idx1, hig_target_idx2;
    int16 eta_hig_pt_target0, eta_hig_pt_target1, eta_hig_pt_target2;
    int16 phi_hig_pt_target0, phi_hig_pt_target1, phi_hig_pt_target2;
    int16 pt_hig_pt_target0, pt_hig_pt_target1, pt_hig_pt_target2;
    aie::mask<V_SIZE> mask_hig_pt_cur0[P_BUNCHES], mask_hig_pt_cur1[P_BUNCHES];
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
        for (int j=0; j<V_SIZE; j++)
        {   
            if (!mask_hig_pt[i].test(j)) continue;

            hig_target_idx0 = j;
            #if defined(__X86SIM__) && defined(__X86DEBUG__)
            printf("hig_target_idx0 = %d (j)\n", hig_target_idx0);
            #endif
            eta_hig_pt_target0 = etas[i][hig_target_idx0];
            phi_hig_pt_target0 = phis[i][hig_target_idx0];
            pt_hig_pt_target0 = pts[i][hig_target_idx0];

            mask_hig_pt_cur0[i] = mask_hig_pt[i];
            mask_hig_pt_cur0[i].clear(hig_target_idx0);

            #if defined(__X86SIM__) && defined(__X86DEBUG__)
            printf("mask_hig_pt_cur0[i = %d]:\n", i);
            aie::print(mask_hig_pt_cur0[i]);
            printf("\n");
            #endif

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

                acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
                acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
                dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
                dr2_float = aie::to_float(dr2, 0);
                acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                dr2_float = acc_float.to_vector<float>(0);

                is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                angsep0[k] = is_ge_mindr2 & mask_hig_pt_cur0[k];

                for (int jj=0; jj<V_SIZE; jj++)
                {
                    if (!angsep0[k].test(jj)) continue;

                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                    printf("angsep0[k = %d]:\n", k);
                    aie::print(angsep0[k]);
                    printf("\n");
                    #endif

                    hig_target_idx1 = jj;
                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                    printf("hig_target_idx1 = %d (jj)\n", hig_target_idx1);
                    #endif
                    eta_hig_pt_target1 = etas[k][hig_target_idx1];
                    phi_hig_pt_target1 = phis[k][hig_target_idx1];
                    pt_hig_pt_target1 = pts[k][hig_target_idx1];

                    mask_hig_pt_cur1[k] = mask_hig_pt_cur0[k];
                    mask_hig_pt_cur1[k].clear(hig_target_idx1);

                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                    printf("mask_hig_pt_cur1[k = %d]:\n", k);
                    aie::print(mask_hig_pt_cur1[i]);
                    printf("\n");
                    #endif

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

                        acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
                        acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
                        dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
                        dr2_float = aie::to_float(dr2, 0);
                        acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                        dr2_float = acc_float.to_vector<float>(0);

                        is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                        angsep1[kk] = is_ge_mindr2 & mask_hig_pt_cur1[kk];

                        for (int jjj=0; jjj<V_SIZE; jjj++)
                        {
                            if (!angsep1[kk].test(jjj)) continue;

                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("angsep1[kk = %d]:\n", kk);
                            aie::print(angsep1[kk]);
                            printf("\n");
                            #endif

                            hig_target_idx2 = jjj;
                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("hig_target_idx2 = %d (jjj)\n", hig_target_idx2);
                            #endif
                            eta_hig_pt_target2 = etas[kk][hig_target_idx2];
                            phi_hig_pt_target2 = phis[kk][hig_target_idx2];
                            pt_hig_pt_target2 = pts[kk][hig_target_idx2];

                            d_eta_scalar = eta_hig_pt_target2 - eta_hig_pt_target0;
                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("d_eta_scalar = %d \n", d_eta_scalar);
                            #endif

                            d_phi_scalar = phi_hig_pt_target2 - phi_hig_pt_target0;
                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("d_phi_scalar = %d \n", d_phi_scalar);
                            #endif
                            d_phi_scalar = (d_phi_scalar <= PI) ? ((d_phi_scalar >= MPI) ? d_phi_scalar : d_phi_scalar + TWOPI) : d_phi_scalar + MTWOPI;
                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("d_phi_scalar (adjusted) = %d \n", d_phi_scalar);
                            #endif

                            dr2_scalar = d_eta_scalar * d_eta_scalar + d_phi_scalar * d_phi_scalar;
                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("dr2_scalar = %i \n", dr2_scalar);
                            #endif
                            dr2_float_scalar = dr2_scalar * F_CONV2;
                            #if defined(__X86SIM__) && defined(__X86DEBUG__)
                            printf("dr2_float_scalar = %f \n", dr2_float_scalar);
                            #endif

                            if (dr2_float_scalar >= MINDR2_ANGSEP_FLOAT)
                            {
                                charge0 = (pdg_ids[i][hig_target_idx0] >= 4) ? ((pdg_ids[i][hig_target_idx0] == 4) ? -1 : 1) : ((pdg_ids[i][hig_target_idx0] == 2) ? -1 : 1);
                                charge1 = (pdg_ids[k][hig_target_idx1] >= 4) ? ((pdg_ids[k][hig_target_idx1] == 4) ? -1 : 1) : ((pdg_ids[k][hig_target_idx1] == 2) ? -1 : 1);
                                charge2 = (pdg_ids[kk][hig_target_idx2] >= 4) ? ((pdg_ids[kk][hig_target_idx2] == 4) ? -1 : 1) : ((pdg_ids[kk][hig_target_idx2] == 2) ? -1 : 1);

                                charge_tot = charge0 + charge1 + charge2;
                                #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                printf("charge_tot = %d \n", charge_tot);
                                #endif

                                if ((charge_tot == 1) | (charge_tot == -1))
                                {
                                    mass0 = (charge0 > 0) ? MASS_M : MASS_P;
                                    px0 = pt_hig_pt_target0 * PT_CONV * aie::cos(phi_hig_pt_target0 * F_CONV);
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("px0 = %f \n", px0);
                                    #endif
                                    py0 = pt_hig_pt_target0 * PT_CONV * aie::sin(phi_hig_pt_target0 * F_CONV);
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("py0 = %f \n", py0);
                                    #endif
                                    x = eta_hig_pt_target0 * F_CONV;
                                    sinh = x + ((x * x * x) / 6);
                                    pz0 = pt_hig_pt_target0 * PT_CONV * sinh;
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("pz0 = %f \n", pz0);
                                    #endif
                                    e0 = aie::sqrt(px0 * px0 + py0 * py0 + pz0 * pz0 + mass0 * mass0);
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("e0 = %f \n", e0);
                                    #endif
                            
                                    mass1 = (charge1 > 0) ? MASS_M : MASS_P;
                                    px1 = pt_hig_pt_target1 * PT_CONV * aie::cos(phi_hig_pt_target1 * F_CONV);
                                    py1 = pt_hig_pt_target1 * PT_CONV * aie::sin(phi_hig_pt_target1 * F_CONV);
                                    x = eta_hig_pt_target1 * F_CONV;
                                    sinh = x + ((x * x * x) / 6);
                                    pz1 = pt_hig_pt_target1 * PT_CONV * sinh;
                                    e1 = aie::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + mass1 * mass1);
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("e1 = %f \n", e1);
                                    #endif

                                    mass2 = (charge2 > 0) ? MASS_M : MASS_P;
                                    px2 = pt_hig_pt_target2 * PT_CONV * aie::cos(phi_hig_pt_target2 * F_CONV);
                                    py2 = pt_hig_pt_target2 * PT_CONV * aie::sin(phi_hig_pt_target2 * F_CONV);
                                    x = eta_hig_pt_target2 * F_CONV;
                                    sinh = x + ((x * x * x) / 6);
                                    pz2 = pt_hig_pt_target2 * PT_CONV * sinh;
                                    e2 = aie::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + mass2 * mass2);
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("e2 = %f \n", e2);
                                    #endif

                                    px_tot = px0 + px1 + px2;
                                    py_tot = py0 + py1 + py2;
                                    pz_tot = pz0 + pz1 + pz2;
                                    e_tot = e0 + e1 + e2;
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("e_tot = %f \n", e_tot);
                                    #endif

                                    invariant_mass = aie::sqrt(e_tot * e_tot - px_tot * px_tot - py_tot * py_tot - pz_tot * pz_tot);
                                    #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                    printf("invariant_mass = %f \n", invariant_mass);
                                    #endif

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
                                        #if defined(__X86SIM__) && defined(__X86DEBUG__)
                                        printf("triplet:\n");
                                        aie::print(triplet);
                                        printf("\n");
                                        #endif
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

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("\n\n");
    #endif
}


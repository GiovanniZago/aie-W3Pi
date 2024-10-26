#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include <aie_api/utils.hpp>
#include "kernels.h"

using namespace adf;

void unpacker(input_stream<int32> * __restrict in_H, input_stream<int32> * __restrict in_L, 
                output_buffer<int32> & __restrict eta_out, output_buffer<int32> & __restrict phi_out, output_buffer<int32> & __restrict pt_out, output_buffer<int32> & __restrict pdg_id_out)
{   
    // data variables
    aie::vector<int32, V_SIZE> data_H[P_BUNCHES], data_L[P_BUNCHES]; 
    aie::vector<int32, V_SIZE> etas[P_BUNCHES], phis_H[P_BUNCHES], phis_L[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES];

    // auxiliary variables
    aie::vector<int32, V_SIZE> foo, complement, msb;
    aie::mask<V_SIZE> msb_msk;

    auto eta_out_vItr = aie::begin_vector<V_SIZE>(eta_out);
    auto phi_out_vItr = aie::begin_vector<V_SIZE>(phi_out);
    auto pt_out_vItr = aie::begin_vector<V_SIZE>(pt_out);
    auto pdg_id_out_vItr = aie::begin_vector<V_SIZE>(pdg_id_out);

    for (int i=0; i<P_BUNCHES; i++)
    {
        data_H[i] = readincr_v<V_SIZE>(in_H);
        data_L[i] = readincr_v<V_SIZE>(in_L);
    }

    data_H[0].set(0, 0);
    data_L[0].set(0, 0);

    for (int i=0; i<(P_BUNCHES); i++)
    chess_prepare_for_pipelining
    {
        // unpack eta
        foo = aie::downshift(data_L[i], ETA_SHIFT);
        etas[i] = aie::bit_and((int32)((1 << (ETA_MSB + 1)) - 1), foo); // select twelve bits belonging to eta

        msb = aie::downshift(etas[i], ETA_MSB);
        msb = aie::bit_and((int32) 0x1, msb); // isolate the MSB of eta in order to see the sign
        msb_msk = aie::eq(msb, (int32) 1); // create a mask with 1 where msb is 1

        complement = aie::add(etas[i], (int32) ((-1) * (1 << (ETA_MSB + 1)))); // reverse two's complement to all the elements of eta
        etas[i] = aie::select(etas[i], complement, msb_msk);

        // unpack phi
        foo = aie::downshift(data_L[i], PHI_SHIFT_L);
        phis_L[i] = aie::bit_and((int32)((1 << (PHI_MSB_L + 1)) - 1), foo);
        phis_H[i] = aie::bit_and((int32)((1 << (PHI_MSB_H + 1)) - 1), data_H[i]); 

        for (int j=0; j<V_SIZE; j++)
        {   
            phis[i][j] = (phis_H[i][j] << 6) | phis_L[i][j];
        }

        msb = aie::downshift(phis[i], PHI_MSB);
        msb = aie::bit_and((int32) 0x1, msb); 
        msb_msk = aie::eq(msb, (int32) 1); 

        complement = aie::add(phis[i], (int32) ((-1) * (1 << (PHI_MSB + 1)))); 
        phis[i] = aie::select(phis[i], complement, msb_msk);

        // unpack pt (easier, no downshift and no sign)
        pts[i] = aie::bit_and((int32)((1 << (PT_MSB + 1)) - 1), data_L[i]);

        // unpack pdg_id
        foo = aie::downshift(data_H[i], PDG_ID_SHIFT);
        pdg_ids[i] = aie::bit_and((int32)((1 << (PDG_ID_MSB + 1)) - 1), foo);

        // send data out
        *eta_out_vItr++ = etas[i];
        *phi_out_vItr++ = phis[i];
        *pt_out_vItr++ = pts[i];
        *pdg_id_out_vItr++ = pdg_ids[i];
    }
}

void filter(input_buffer<int32> & __restrict eta_in, input_buffer<int32> & __restrict phi_in, input_buffer<int32> & __restrict pt_in, input_buffer<int32> & __restrict pdg_id_in,
                output_buffer<int32> & __restrict eta_out, output_buffer<int32> & __restrict phi_out, output_buffer<int32> & __restrict pt_out, output_buffer<int32> & __restrict pdg_id_out)
{
    // DATA VECTORS
    aie::vector<int32, V_SIZE> etas[P_BUNCHES], phis[P_BUNCHES], pts[P_BUNCHES], pdg_ids[P_BUNCHES], foo;

    // I/O ITERATORS
    auto eta_in_vItr = aie::begin_vector<V_SIZE>(eta_in);
    auto phi_in_vItr = aie::begin_vector<V_SIZE>(phi_in);
    auto pt_in_vItr = aie::begin_vector<V_SIZE>(pt_in);
    auto pdg_id_in_vItr = aie::begin_vector<V_SIZE>(pdg_id_in);

    auto eta_out_vItr = aie::begin_vector<V_SIZE>(eta_out);
    auto phi_out_vItr = aie::begin_vector<V_SIZE>(phi_out);
    auto pt_out_vItr = aie::begin_vector<V_SIZE>(pt_out);
    auto pdg_id_out_vItr = aie::begin_vector<V_SIZE>(pdg_id_out);

    // auxiliary zero vector
    const aie::vector<int32, V_SIZE> zeros_vector = aie::zeros<int32, V_SIZE>();

    // VARIABLES FOR FILTERING PT CUTS AND PDG ID
    // filtering pt
    aie::mask<V_SIZE> mask_min_pt[P_BUNCHES], mask_med_pt[P_BUNCHES], mask_hig_pt[P_BUNCHES];

    // filtering pdg_id
    aie::mask<V_SIZE> pdg_id_tot_mask[P_BUNCHES], pdg_id_mask1, pdg_id_mask2, pdg_id_mask3, pdg_id_mask4;

    // VARIABLES TO CALCULATE ISOLATION FOR EACH PARTICLE
    // general variables
    int32 eta_cur, phi_cur, pt_cur, pt_sum;
    aie::vector<int32, V_SIZE> d_eta, d_phi, dr2;
    aie::accum<acc64, V_SIZE> acc;
    aie::vector<float, V_SIZE> dr2_float;
    aie::accum<accfloat, V_SIZE> acc_float;
    aie::mask<V_SIZE> iso_mask[P_BUNCHES], filter_mask[P_BUNCHES];

    // variables for the two-pi check
    aie::mask<V_SIZE> is_gt_pi, is_lt_mpi;
    aie::vector<int32, V_SIZE> d_phi_ptwopi, d_phi_mtwopi;
    aie::vector<int32, V_SIZE> pi_vector = aie::broadcast<int32, V_SIZE>(PI);
    aie::vector<int32, V_SIZE> mpi_vector = aie::broadcast<int32, V_SIZE>(MPI);
    aie::vector<int32, V_SIZE> twopi_vector = aie::broadcast<int32, V_SIZE>(TWOPI);
    aie::vector<int32, V_SIZE> mtwopi_vector = aie::broadcast<int32, V_SIZE>(MTWOPI);

    // variables to apply the pt cuts
    aie::mask<V_SIZE> is_ge_mindr2, is_le_maxdr2, pt_cut_mask;
    aie::vector<int32, V_SIZE> pt_to_sum;

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("**************NEW ITERATION**************\n\n");
    #endif

    for (int i=0; i<P_BUNCHES; i++)
    chess_prepare_for_pipelining
    {
        etas[i] = *eta_in_vItr++;
        phis[i] = *phi_in_vItr++;
        pts[i] = *pt_in_vItr++;
        pdg_ids[i] = *pdg_id_in_vItr++;

        // PERFORM PDG_ID CUT -> ISOLATION -> DIVIDE INTO PT CATEGORIES
        pdg_id_mask1 = aie::eq((int32) 0b010, pdg_ids[i]);
        pdg_id_mask2 = aie::eq((int32) 0b011, pdg_ids[i]);
        pdg_id_mask3 = aie::eq((int32) 0b100, pdg_ids[i]);
        pdg_id_mask4 = aie::eq((int32) 0b101, pdg_ids[i]);

        pdg_id_mask1 = pdg_id_mask1 | pdg_id_mask2;
        pdg_id_mask3 = pdg_id_mask3 | pdg_id_mask4;
        pdg_id_tot_mask[i] = pdg_id_mask1 | pdg_id_mask3;
        
        // CALCULATE ISOLATION
        for (int j=0; j<V_SIZE; j++)
        {
            eta_cur = etas[i][j];
            phi_cur = phis[i][j];
            pt_cur = pts[i][j];
            iso_mask[i].clear(j);
            pt_sum = 0;

            for (int k=0; k<P_BUNCHES; k++)
            {
                d_eta = aie::sub(eta_cur, etas[k]);

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ISODEBUG__)
                if ((i==0) && (j==2))
                {
                    printf("d_eta:\n");
                    aie::print(d_eta);
                    printf("\n");
                }
                #endif

                d_phi = aie::sub(phi_cur, phis[k]);
                d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                is_gt_pi = aie::gt(d_phi, pi_vector);
                is_lt_mpi = aie::lt(d_phi, mpi_vector);
                d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ISODEBUG__)
                if ((i==0) && (j==2))
                {
                    printf("d_phi:\n");
                    aie::print(d_phi);
                    printf("\n");
                }
                #endif

                acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
                acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
                dr2 = acc.to_vector<int32>(0); // convert accumulator into vector

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ISODEBUG__)
                if ((i==0) && (j==2))
                {
                    printf("dr2:\n");
                    aie::print(dr2);
                    printf("\n");
                }
                #endif
                
                // actually one can convert the cuts mindr2 and maxdr2 to integers instead of converting the vectors
                dr2_float = aie::to_float(dr2, 0);
                acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                dr2_float = acc_float.to_vector<float>(0);

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ISODEBUG__)
                if ((i==0) && (j==2))
                {
                    printf("dr2 float:\n");
                    aie::print(dr2_float);
                    printf("\n");
                }
                #endif

                is_ge_mindr2 = aie::ge(dr2_float, MINDR2_FLOAT);
                is_le_maxdr2 = aie::le(dr2_float, MAXDR2_FLOAT);
                pt_cut_mask = is_ge_mindr2 & is_le_maxdr2;

                pt_to_sum = aie::select(zeros_vector, pts[i], pt_cut_mask); // select only the pts that fall in the desired range
                pt_sum += aie::reduce_add(pt_to_sum); // update the pt sum
            }
            
            // apply the threshold on pt sum
            if (pt_sum <= (MAX_ISO * pt_cur)) iso_mask[i].set(j);
        }
        
        // filter particles according to their isolation and pdg_id
        // This is key because the PT categories have to be made
        // after filtering the pdg_id and the isolation, otherwise
        // we are missing - as said - the isolation or even wrong 
        // particles.
        filter_mask[i] = iso_mask[i] & pdg_id_tot_mask[i]; 

        etas[i] = aie::select(zeros_vector, etas[i], filter_mask[i]); 
        phis[i] = aie::select(zeros_vector, phis[i], filter_mask[i]); 
        pts[i] = aie::select(zeros_vector, pts[i], filter_mask[i]); 
        pdg_ids[i] = aie::select(zeros_vector, pdg_ids[i], filter_mask[i]); 

        // filter out hadrons and electrons with at least
        // MIN_PT value of pt
        mask_min_pt[i] = aie::ge(pts[i], MIN_PT);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86PTCUTDEBUG__)
        printf("pts bunch %d:\n", i);
        aie::print(pts[i]);
        printf("\n");
        printf("mask_min_pt bunch %d:\n", i);
        aie::print(mask_min_pt[i]);
        printf("\n");
        #endif

        // filter out hadrons and electrons with at least
        // MED_PT value of pt
        mask_med_pt[i] = aie::ge(pts[i], MED_PT);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86PTCUTDEBUG__)
        printf("mask_med_pt bunch %d:\n", i);
        aie::print(mask_med_pt[i]);
        printf("\n");
        #endif

        // filter out hadrons and electrons with at least
        // HIG_PT value of pt
        mask_hig_pt[i] = aie::ge(pts[i], HIG_PT);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86PTCUTDEBUG__)
        printf("mask_hig_pt bunch %d:\n", i);
        aie::print(mask_hig_pt[i]);
        printf("\n\n");
        #endif
        
        // xor mask_min_pt and mask_med_pt in order to keep inside
        // mask_min_pt the ones only when pt is at least min_pt but
        // less than med_pt. Do the same but with med_pt and hig_pt
        mask_min_pt[i] = (mask_min_pt[i] & (~mask_med_pt[i])) | ((~mask_min_pt[i]) & mask_med_pt[i]);
        mask_med_pt[i] = (mask_med_pt[i] & (~mask_hig_pt[i])) | ((~mask_med_pt[i]) & mask_hig_pt[i]);
    }   

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("pdg_id data:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(pdg_ids[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("phis data:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(phis[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("etas data:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(etas[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("pts data:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(pts[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("pdg_id mask:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(pdg_id_tot_mask[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("iso_mask:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(iso_mask[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("pdg_id final mask AND iso_mask:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(filter_mask[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("min pt mask:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(mask_min_pt[i]);
    }
    printf("\n");
    #endif
    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("med pt mask:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(mask_med_pt[i]);
    }
    printf("\n");
    #endif
    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("hig pt mask:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(mask_hig_pt[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("\n\n");
    printf("CALCULATING ANGULAR SEPARATION\n\n");
    #endif
    
    // GROUPING PARTICLES INTO PT CATEGORIES
    aie::vector<int32, V_SIZE> etas_min_pt[P_BUNCHES], phis_min_pt[P_BUNCHES];
    aie::vector<int32, V_SIZE> etas_med_pt[P_BUNCHES], phis_med_pt[P_BUNCHES];
    aie::vector<int32, V_SIZE> etas_hig_pt[P_BUNCHES], phis_hig_pt[P_BUNCHES];

    aie::vector<int32, V_SIZE> angsep_idx_med_min[P_BUNCHES], angsep_idx_hig_min[P_BUNCHES], angsep_idx_hig_med[P_BUNCHES];
    int32 index_cur;

    static int32 index_array[P_BUNCHES][V_SIZE] = {
        {0, 1, 2, 3, 4, 5, 6, 7},     
        {8, 9, 10, 11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20, 21, 22, 23}, 
        {24, 25, 26, 27, 28, 29, 30, 31}, 
        {32, 33, 34, 35, 36, 37, 38, 39}, 
        {40, 41, 42, 43, 44, 45, 46, 47}, 
        {48, 49, 50, 51, 52, 53, 54, 55}, 
        {56, 57, 58, 59, 60, 61, 62, 63}, 
        {64, 65, 66, 67, 68, 69, 70, 71}, 
        {72, 73, 74, 75, 76, 77, 78, 79}, 
        {80, 81, 82, 83, 84, 85, 86, 87}, 
        {88, 89, 90, 91, 92, 93, 94, 95}, 
        {96, 97, 98, 99, 100, 101, 102, 103}
    };

    const aie::vector<int32, V_SIZE> index_vector[P_BUNCHES]{
        {*(v8int32 *) index_array[0]},
        {*(v8int32 *) index_array[1]},
        {*(v8int32 *) index_array[2]},
        {*(v8int32 *) index_array[3]},
        {*(v8int32 *) index_array[4]},
        {*(v8int32 *) index_array[5]},
        {*(v8int32 *) index_array[6]},
        {*(v8int32 *) index_array[7]},
        {*(v8int32 *) index_array[8]},
        {*(v8int32 *) index_array[9]},
        {*(v8int32 *) index_array[10]},
        {*(v8int32 *) index_array[11]},
        {*(v8int32 *) index_array[12]}
    };

    // group etas and phis into pt cut groups and initialize angsep index vectors
    for (int i=0; i<P_BUNCHES; i++)
    {
        etas_min_pt[i] = aie::select(zeros_vector, etas[i], mask_min_pt[i]);
        phis_min_pt[i] = aie::select(zeros_vector, phis[i], mask_min_pt[i]);
        
        etas_med_pt[i] = aie::select(zeros_vector, etas[i], mask_med_pt[i]);
        phis_med_pt[i] = aie::select(zeros_vector, phis[i], mask_med_pt[i]);

        etas_hig_pt[i] = aie::select(zeros_vector, etas[i], mask_hig_pt[i]);
        phis_hig_pt[i] = aie::select(zeros_vector, phis[i], mask_hig_pt[i]);
        
        angsep_idx_med_min[i] = aie::broadcast<int32, V_SIZE>(0);
        angsep_idx_hig_min[i] = aie::broadcast<int32, V_SIZE>(0);
        angsep_idx_hig_med[i] = aie::broadcast<int32, V_SIZE>(0);
    }

    // calculating angular separation between particles in different
    // pt categories
    for (int i=0; i<P_BUNCHES; i++)
    {
        for (int j=0; j<V_SIZE; j++)
        {
            eta_cur = etas_min_pt[i][j];
            phi_cur = phis_min_pt[i][j];
            index_cur = index_vector[i][j];

            // the following line is essential because if the current index 
            // represents a particle that has not passed the min pt selection
            // then the calculation of the angular separation from any other
            // particle is invalid, and thus, just in case the separation is
            // asserted, we insert in angsep_idx_med_min a zero (equivalent to 
            // no separation asserted)
            index_cur = (mask_min_pt[i].test(j) == 0) ? 0 : index_cur+1;

            for (int k=0; k<P_BUNCHES; k++)
            {

                // (1) angular separation between med and min
                d_eta = aie::sub(eta_cur, etas_med_pt[k]);

                d_phi = aie::sub(phi_cur, phis_med_pt[k]);
                d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                is_gt_pi = aie::gt(d_phi, pi_vector);
                is_lt_mpi = aie::lt(d_phi, mpi_vector);
                d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
                acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
                dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
                
                // actually one can convert the cuts mindr2 and maxdr2 to integers instead of converting the vectors
                dr2_float = aie::to_float(dr2, 0);
                acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                dr2_float = acc_float.to_vector<float>(0);

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ANGSEPDEBUG__)
                if ((i==1) && (j==4) && (k==0))
                {
                    printf("med/min dr2_float:\n");
                    aie::print(dr2_float);
                    printf("\n");
                }
                #endif

                is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                angsep_idx_med_min[k] = aie::select(angsep_idx_med_min[k], index_cur, is_ge_mindr2);

                // the following line is needed to make sure that the ang separation is calculated
                // using elements from etas_med_pt that are not zero (where med_pt_mask[k] is 1)
                angsep_idx_med_min[k] = aie::select(zeros_vector, angsep_idx_med_min[k], mask_med_pt[k]);

                // (2) angular separation between high and min
                d_eta = aie::sub(eta_cur, etas_hig_pt[k]);

                d_phi = aie::sub(phi_cur, phis_hig_pt[k]);
                d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                is_gt_pi = aie::gt(d_phi, pi_vector);
                is_lt_mpi = aie::lt(d_phi, mpi_vector);
                d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
                acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
                dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
                
                // actually one can convert the cuts mindr2 and maxdr2 to integers instead of converting the vectors
                dr2_float = aie::to_float(dr2, 0);
                acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                dr2_float = acc_float.to_vector<float>(0);

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ANGSEPDEBUG__)
                if ((i==1) && (j==4) && (k==0))
                {
                    printf("high/min dr2_float:\n");
                    aie::print(dr2_float);
                    printf("\n");
                }
                #endif

                is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ANGSEPDEBUG__)
                if ((i==1) && (j==4) && (k==0))
                {
                    printf("high/min is_ge_mindr2:\n");
                    aie::print(is_ge_mindr2);
                    printf("\n");
                }
                #endif

                angsep_idx_hig_min[k] = aie::select(angsep_idx_hig_min[k], index_cur, is_ge_mindr2);
                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ANGSEPDEBUG__)
                if ((i==1) && (j==4) && (k==0))
                {
                    printf("angsep_idx_hig_min bunch %d before hig pt filter:\n", k);
                    aie::print(angsep_idx_hig_min[k]);
                    printf("\n");
                }
                #endif

                // the following line is needed to make sure that the ang separation is calculated
                // using elements from etas_hig_pt that are not zero (where med_pt_mask[k] is 1)
                angsep_idx_hig_min[k] = aie::select(zeros_vector, angsep_idx_hig_min[k], mask_hig_pt[k]);
                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__x86ANGSEPDEBUG__)
                if ((i==1) && (j==4) && (k==0))
                {
                    printf("angsep_idx_hig_min bunch %d after hig pt filter:\n", k);
                    aie::print(angsep_idx_hig_min[k]);
                    printf("\n");
                }
                #endif
            }
        }

        for (int j=0; j<V_SIZE; j++)
        {
            eta_cur = etas_med_pt[i][j];
            phi_cur = phis_med_pt[i][j];
            index_cur = index_vector[i][j];

            // the following line is essential because if the current index 
            // represents a particle that has not passed the med pt selection
            // then the calculation of the angular separation from any other
            // particle is invalid, and thus, just in case the separation is
            // asserted, we insert in angsep_idx_hig_med a zero (equivalent to 
            // no separation asserted)
            index_cur = (mask_med_pt[i].test(j) == 0) ? 0 : index_cur+1;

            for (int k=0; k<P_BUNCHES; k++)
            {
                // (3) angular separation between high and med
                d_eta = aie::sub(eta_cur, etas_hig_pt[k]);

                d_phi = aie::sub(phi_cur, phis_hig_pt[k]);
                d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                is_gt_pi = aie::gt(d_phi, pi_vector);
                is_lt_mpi = aie::lt(d_phi, mpi_vector);
                d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                acc = aie::mul_square(d_eta); // acc = d_eta ^ 2
                acc = aie::mac_square(acc, d_phi); // acc = acc + d_phi ^ 2
                dr2 = acc.to_vector<int32>(0); // convert accumulator into vector
                
                // actually one can convert the cuts mindr2 and maxdr2 to integers instead of converting the vectors
                dr2_float = aie::to_float(dr2, 0);
                acc_float = aie::mul(dr2_float, F_CONV2); // dr2_float = dr2_int * ((pi / 720) ^ 2)
                dr2_float = acc_float.to_vector<float>(0);

                is_ge_mindr2 = aie::ge(dr2_float, MINDR2_ANGSEP_FLOAT);
                angsep_idx_hig_med[k] = aie::select(angsep_idx_hig_med[k], index_cur, is_ge_mindr2);

                // the following line is needed to make sure that the ang separation is calculated
                // using elements from etas_hig_pt that are not zero (where med_pt_mask[k] is 1)
                angsep_idx_hig_med[k] = aie::select(zeros_vector, angsep_idx_hig_med[k], mask_hig_pt[k]);
            }
        }
    }

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("angsep_idx_med_min:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(angsep_idx_med_min[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("angsep_idx_hig_min:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(angsep_idx_hig_min[i]);
    }
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("angsep_idx_hig_med:\n");
    for (int i=0; i<P_BUNCHES; i++)
    {
        aie::print(angsep_idx_hig_med[i]);
    }
    printf("\n");
    #endif
    
    aie::mask<V_SIZE> is_idx_eq1[P_BUNCHES], is_idx_eq2[P_BUNCHES];
    int32 idx_target1, idx_target2;
    bool flag = false;

    for (int i=0; i<P_BUNCHES; i++)
    {
        for (int j=0; j<V_SIZE; j++)
        {
            index_cur = index_vector[i][j];

            for (int k=0; k<P_BUNCHES; k++)
            {
                is_idx_eq1[k] = aie::eq(angsep_idx_hig_min[k], index_cur+1);

                // idx_target represents the index of the particle, inside the vector angsep_idx_hig_min[k],
                // belonging to the high pt cut that is ang sep from the one 
                // belonging to the min pt cut, represented by index_cur
                idx_target1 = is_idx_eq1[k].clz(); 
                if (idx_target1 == V_SIZE) continue;

                // now we have to check the content at index idx_target1 of angsep_idx_hig_med[k]
                // In this way we get idx_target2. Then, we check the content of angsep_idx_med_min
                // at idx_target2 - 1 to see if it matches index_cur. If yes, we have found an 
                // angularly separated triplet. Pay attention that now idx target 2 is in the range
                // 0 - 103 so we have to acces angsep_idx_med_min in another way
                idx_target2 = angsep_idx_hig_med[k][idx_target1];
                idx_target2 -= 1;
                if (idx_target2 == index_cur) 
                {
                    flag = true;
                    break;
                }
            }

        }

        *eta_out_vItr++ = etas[i];
        *phi_out_vItr++ = phis[i];
        *pt_out_vItr++ = pts[i];
        *pdg_id_out_vItr++ = pdg_ids[i];
    }
        
    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("triplet found flag:\n");
    printf("%d", flag);
    printf("\n");
    #endif
}


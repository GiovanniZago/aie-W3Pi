#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#define V_SIZE 8
#define EV_SIZE 104
#define P_BUNCHES 13

#define __X86DEBUG__ 
// #define __X86PTCUTDEBUG__
// #define __x86ISODEBUG__
// #define __x86ANGSEPDEBUG__

static const int32 ETA_SHIFT = 14;
static const int32 ETA_MSB = 11; // "assuming" lsb is zero
static const int32 PHI_SHIFT_L = 26;
static const int32 PHI_MSB_L = 5;
static const int32 PHI_MSB_H = 4;
static const int32 PHI_MSB = 10;
static const int32 PT_SHIFT = 0;
static const int32 PT_MSB = 13;
static const int32 PDG_ID_SHIFT = 5;
static const int32 PDG_ID_MSB = 2;
static const int32 MIN_PT = 7;
static const int32 MED_PT = 12;
static const int32 HIG_PT = 15;
static const int32 PI = 720;
static const int32 MPI = -720;
static const int32 TWOPI = 1440;
static const int32 MTWOPI = -1440;
static const float MINDR2_FLOAT = 0.01 * 0.01;
static const float MAXDR2_FLOAT = 0.25 * 0.25;
static const float MINDR2_ANGSEP_FLOAT = 0.5 * 0.5;
static const float PI_FLOAT = 3.1415926;
static const float F_CONV2 = (PI_FLOAT / PI) * (PI_FLOAT / PI);
static const float MAX_ISO = 0.5;

using namespace adf;

void unpacker(input_stream<int32> * __restrict in_H, input_stream<int32> * __restrict in_L, 
                output_buffer<int32> & __restrict eta_out, output_buffer<int32> & __restrict phi_out, output_buffer<int32> & __restrict pt_out, output_buffer<int32> & __restrict pdg_id_out);


void filter(input_buffer<int32> & __restrict eta_in, input_buffer<int32> & __restrict phi_in, input_buffer<int32> & __restrict pt_in, input_buffer<int32> & __restrict pdg_id_in,
                output_buffer<int32> & __restrict eta_out, output_buffer<int32> & __restrict phi_out, output_buffer<int32> & __restrict pt_out, output_buffer<int32> & __restrict pdg_id_out);

void isolation(input_buffer<int32> & __restrict eta_in, input_buffer<int32> & __restrict phi_in, input_buffer<int32> & __restrict pt_in, input_buffer<int32> & __restrict pdg_id_in,
                output_buffer<int32> & __restrict eta_out, output_buffer<int32> & __restrict phi_out, output_buffer<int32> & __restrict pt_out, output_buffer<int32> & __restrict pdg_id_out);
#endif
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#define V_SIZE 32
#define EV_SIZE 224
#define P_BUNCHES 7
#define BUF_SIZE 224

// #define __X86DEBUG__ 
// #define __X86PTCUTDEBUG__
// #define __x86ISODEBUG__
// #define __x86ANGSEPDEBUG__

static const int16 PT_MSB = 13;
static const int16 PHI_SHIFT_L = 26;
static const int16 PHI_MSB_L = 5;
static const int16 PHI_MSB_H = 4;
static const int16 PDG_ID_SHIFT = 5;
static const int16 PDG_ID_MSB = 2;
static const int16 MIN_PT = 28;
static const int16 MED_PT = 48;
static const int16 HIG_PT = 60;
static const int16 PI = 720;
static const int16 MPI = -720;
static const int16 TWOPI = 1440;
static const int16 MTWOPI = -1440;
static const float MIN_MASS = 60.0;
static const float MAX_MASS = 100.0;
static const float MINDR2_FLOAT = 0.01 * 0.01;
static const float MAXDR2_FLOAT = 0.25 * 0.25;
static const float MINDR2_ANGSEP_FLOAT = 0.5 * 0.5;
static const float PI_FLOAT = 3.1415926;
static const float F_CONV = PI_FLOAT / PI;
static const float F_CONV2 = (PI_FLOAT / PI) * (PI_FLOAT / PI);
static const float PT_CONV = 0.25;
static const float MAX_ISO = 0.5;
static const float MASS_M = 0.1349768;
static const float MASS_P = 0.13957039;

using namespace adf;

void unpacker(input_stream<int32> * __restrict in_H, input_stream<int32> * __restrict in_L, 
                output_buffer<int16> & __restrict pt_out, output_buffer<int16> & __restrict eta_out, 
                output_buffer<int16> & __restrict phi_out, output_buffer<int16> & __restrict pdg_id_out);

void filter(input_buffer<int16> & __restrict pt_in, input_buffer<int16> & __restrict eta_in, 
                input_buffer<int16> & __restrict phi_in, input_buffer<int16> & __restrict pdg_id_in,
                output_buffer<int16> & __restrict pt_out, output_buffer<int16> & __restrict eta_out, 
                output_buffer<int16> & __restrict phi_out, output_buffer<int16> & __restrict pdg_id_out);

void combinatorial(input_buffer<int16> & __restrict pt_in, input_buffer<int16> & __restrict eta_in, 
                    input_buffer<int16> & __restrict phi_in, input_buffer<int16> & __restrict pdg_id_in,
                    output_stream<float> * __restrict out);

#endif
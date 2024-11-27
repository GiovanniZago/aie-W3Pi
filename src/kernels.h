#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#define V_SIZE 32
#define EV_SIZE 224
#define P_BUNCHES 7

// #define __X86DEBUG__ 
// #define __X86PTCUTDEBUG__
// #define __x86ISODEBUG__
// #define __x86ANGSEPDEBUG__

static const int16 N_MIN = 16;
static const int16 PT_MSB = 13;
static const int16 ETA_MSB = 11;
static const int16 PHI_SHIFT_L = 26;
static const int16 PHI_MSB_L = 5;
static const int16 PHI_MSB_H = 4;
static const int16 PHI_MSB = 10;
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
static const float MASS_P = 0.13957039;

using namespace adf;

void unpack_filter_iso(input_stream<int64> * __restrict in, output_stream<int16> * __restrict out0, output_stream<int16> * __restrict out1);

void combinatorial(input_stream<int16> * __restrict in0, input_stream<int16> * __restrict in1, output_stream<float> * __restrict out);

#endif
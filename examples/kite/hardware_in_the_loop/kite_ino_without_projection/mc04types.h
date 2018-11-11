#ifndef MC04TYPES_H

    #define MC04TYPES_H

    #ifndef USE_MPC_STDINT

        #define USE_MPC_STDINT
        #include <stdint.h>

    #else
    
        #include "edgeAI_stdint.h"

    #endif

    typedef float float32_t;
    typedef double float64_t;
    typedef long double float128_t;

#endif

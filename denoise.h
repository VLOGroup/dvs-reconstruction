#ifndef DENOISE_H
#define DENOISE_H

#include <iu/iucore.h>

enum METHOD {
    TV_L1,
    TV_L2,
    TV_LogL2,
    TV_LogEntropy,
};

namespace  cuda {

    void initDenoise(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *timestamp);
    void solveTVIncrementalManifold(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *f, iu::ImageGpu_32f_C1 *t,
                                    float lambda, float lambda_time,
                                    int iterations, float u_min, float u_max, METHOD method);
    void solveTVIncrementalManifoldAdaptiveLambda(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *f, iu::ImageGpu_32f_C1 *t,
                                                  float curr_time, float lambda, float lambda_time,
                                                  int iterations, float u_min, float u_max, METHOD method);
    void solveTVIncrementalManifoldOccurenceLambda(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *f, iu::ImageGpu_32f_C1 *t,
                                                   iu::ImageGpu_32u_C1 *o, float lambda, float lambda_time,
                                                   int iterations, float u_min, float u_max, METHOD method);
   } // namespace cuda

#endif

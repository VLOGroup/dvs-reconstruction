// This file is part of dvs-reconstruction.
//
// Copyright (C) 2016 Christian Reinbacher <reinbacher at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// dvs-reconstruction is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// dvs-reconstruction is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#ifndef DENOISE_H
#define DENOISE_H

#include <iu/iucore.h>

enum METHOD {
    TV_L1,
    TV_L2,
    TV_LogL2,
    TV_KLD,
};

namespace  cuda {

    void initDenoise(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *timestamp);
    void solveTVIncrementalManifold(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *f, iu::ImageGpu_32f_C1 *t,
                                    float lambda, float lambda_time,
                                    int iterations, float u_min, float u_max, METHOD method);
   } // namespace cuda

#endif

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
#ifndef COMMON_H
#define COMMON_H
#include "iu/iucore.h"
#include "event.h"
#include <string>
#include <vector>

#define GPU_BLOCK_SIZE 16
#define TIME_CONSTANT 1e-6f

// IO functions
void saveEvents(std::string filename, std::vector<Event> &events);
void saveState(std::string filename, const iu::ImageGpu_32f_C1 *mat, bool as_png=false, bool as_npy=false);
void loadEvents(std::string filename, std::vector<Event>& events, bool skip_events=false, bool flip_ud=false);
void loadState(std::string filename, iu::ImageGpu_32f_C1 *mat, float u_min);

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    //cudaDeviceSynchronize();
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );

    }
#endif

}
#endif // COMMON_H

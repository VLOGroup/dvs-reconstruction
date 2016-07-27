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
#ifndef COMMON_CUH
#define COMMON_CUH

#include <iu/iucore.h>
#include <iu/iucutil.h>


namespace cuda {

  enum UpsampleMethod {
    UPSAMPLE_LINEAR,
    UPSAMPLE_NEAREST
  };

  void setEvents(iu::ImageGpu_32f_C1 *output, iu::ImageGpu_32f_C1 *old_timestamp, iu::LinearHostMemory_32f_C4 *events_host, float C1, float C2);
  void setEvents(iu::ImageGpu_32f_C1 *output,iu::ImageGpu_32f_C1 * old_timestamp, iu::ImageGpu_32u_C1 *occurences, iu::LinearHostMemory_32f_C4 *events_host, float C1, float C2);
  void setEvents(iu::ImageGpu_32f_C1 *output, iu::LinearHostMemory_32f_C4 *events_host, float C1, float C2);
  void upsample(iu::ImageGpu_32f_C1 *in, iu::ImageGpu_32f_C1 *out, UpsampleMethod method, bool exponentiate = false);
}
#endif

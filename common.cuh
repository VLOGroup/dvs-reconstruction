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

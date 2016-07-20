#ifndef DAVISCAMERAWORKER_H
#define DAVISCAMERAWORKER_H

#include <QThread>
#include <libcaer/libcaer.h>
#include <libcaer/devices/davis.h>
#include <queue>

#include "event.h"
#include "denoisingworker.h"

class DAVISCameraWorker : public QThread
{
    Q_OBJECT
   void run() Q_DECL_OVERRIDE;
public:
    DAVISCameraWorker(DenoisingWorker *worker = 0);
public slots:
    void stop(void){running_=false;}
    void snap(void);

protected:
    bool init(void);
    void deinit(void);
    std::vector<Event> events_buffer_;
    iu::ImageCpu_32f_C1 *frame_buffer_;
    caerDeviceHandle davis240_handle_;
    DenoisingWorker *ugly_;
    bool running_;
    bool snap_;
};

#endif // DAVISCAMERAWORKER_H

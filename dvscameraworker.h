#ifndef DVSCAMERAWORKER_H
#define DVSCAMERAWORKER_H

#include <QThread>
#include <libcaer/libcaer.h>
#include <libcaer/devices/dvs128.h>
#include <queue>

#include "event.h"
#include "denoisingworker.h"

class DVSCameraWorker : public QThread
{
    Q_OBJECT
   void run() Q_DECL_OVERRIDE;
public:
    DVSCameraWorker(DenoisingWorker *worker = 0);
public slots:
    void stop(void){running_=false;}

protected:
    bool init(void);
    void deinit(void);
    std::vector<Event> events_buffer_;
    caerDeviceHandle dvs128_handle_;
    DenoisingWorker *ugly_;
    bool running_;
};

#endif // DVSCAMERAWORKER_H

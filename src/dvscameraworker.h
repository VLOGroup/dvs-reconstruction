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

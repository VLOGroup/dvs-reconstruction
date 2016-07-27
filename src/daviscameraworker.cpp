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
#include "daviscameraworker.h"

//DAVIS Bias types
#define CF_N_TYPE(COARSE, FINE) (struct caer_bias_coarsefine) \
        { .coarseValue = (uint8_t)(COARSE), .fineValue = (uint8_t)(FINE), \
        .enabled = true, .sexN = true, \
        .typeNormal = true, .currentLevelNormal = true }

#define CF_P_TYPE(COARSE, FINE) (struct caer_bias_coarsefine) \
        { .coarseValue = (uint8_t)(COARSE), .fineValue = (uint8_t)(FINE), \
        .enabled = true, .sexN = false, \
        .typeNormal = true, .currentLevelNormal = true }

void DAVISCameraWorker::run()
{
    if(init()) {
        running_ = true;
        snap_ = false;
        while(running_)
        {
            // get event and update timestamps
            caerEventPacketContainer packetContainer = caerDeviceDataGet(davis240_handle_);
            if (packetContainer == NULL) {
                msleep(1);
                continue; // Skip if nothing there.
            }
            events_buffer_.clear();
            int32_t packetNum = caerEventPacketContainerGetEventPacketsNumber(packetContainer);
            for (int32_t i = 0; i < packetNum; i++) {
                caerEventPacketHeader packetHeader = caerEventPacketContainerGetEventPacket(packetContainer, i);
                if (packetHeader == NULL) {
                    continue; // Skip if nothing there.
                }
                // Packet 0 is always the special events packet for DVS128, while packet is the polarity events packet.
                if (i == POLARITY_EVENT) {

                    caerPolarityEventPacket polarity = (caerPolarityEventPacket) packetHeader;
                    for (int32_t caerPolarityIteratorCounter = 0; caerPolarityIteratorCounter < caerEventPacketHeaderGetEventNumber(&(polarity)->packetHeader);caerPolarityIteratorCounter++) {
                        caerPolarityEvent caerPolarityIteratorElement = caerPolarityEventPacketGetEvent(polarity, caerPolarityIteratorCounter);
                        if (!caerPolarityEventIsValid(caerPolarityIteratorElement)) { continue; }
                        Event event;
                        event.t = caerPolarityEventGetTimestamp(caerPolarityIteratorElement)*1e-6;
                        event.x = caerPolarityEventGetX(caerPolarityIteratorElement); // don't know why it is other way round?
                        event.y = caerPolarityEventGetY(caerPolarityIteratorElement);
                        event.polarity = caerPolarityEventGetPolarity(caerPolarityIteratorElement)?1.0f:-1.0f;
    //                    if(undistortPoint(event,params.K_cam,params.radial))
                        events_buffer_.push_back(event);
                    }
                } else if (i == FRAME_EVENT && snap_) {
                    caerFrameEventPacket frame = (caerFrameEventPacket) packetHeader;
                    caerFrameEvent event = caerFrameEventPacketGetEvent(frame,0);
                    uint16_t *image = caerFrameEventGetPixelArrayUnsafe(event);
                    // copy it over to a float32 iu::ImageCpu
                    float* dst_data = frame_buffer_->data();
                    for(int idx=0;idx<180*240;idx++)
                        dst_data[idx] = (image[idx] >> 7)/256.0f;
                    ugly_->setOutput(frame_buffer_);
                    snap_ = false;
                }
            }
            caerEventPacketContainerFree(packetContainer);
            ugly_->addEvents(events_buffer_);
        }
        deinit();
    }
}

DAVISCameraWorker::DAVISCameraWorker(DenoisingWorker *worker):ugly_(worker)
{
    frame_buffer_ = new iu::ImageCpu_32f_C1(240,180);
}

bool DAVISCameraWorker::init()
{
    // init camera
    // Open a DVS128, give it a device ID of 1, and don't care about USB bus or SN restrictions.
    davis240_handle_ = caerDeviceOpen(1, CAER_DEVICE_DAVIS_FX2, 0, 0, NULL);
    if (davis240_handle_ == NULL) {
        return false;
    }
//    // Let's take a look at the information we have on the device.
//    struct caer_dvs128_info dvs128_info = caerDVS128InfoGet(davis240_handle_);

//    printf("%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, Logic: %d.\n", dvs128_info.deviceString,
//        dvs128_info.deviceID, dvs128_info.deviceIsMaster, dvs128_info.dvsSizeX, dvs128_info.dvsSizeY,
//        dvs128_info.logicVersion);
    caerDeviceSendDefaultConfig(davis240_handle_);
    // good indoor settings
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_APS, DAVIS_CONFIG_APS_EXPOSURE, 8000);
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_APS, DAVIS_CONFIG_APS_FRAME_DELAY, 0);

    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_APS, DAVIS_CONFIG_APS_RUN, false); // change to true if you want pictures
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_RUN, true);
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_IMU, DAVIS_CONFIG_IMU_RUN, false);
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRBP,
        caerBiasCoarseFineGenerate(CF_P_TYPE(3, 72)));
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRSFBP,
        caerBiasCoarseFineGenerate(CF_P_TYPE(3, 96)));

    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_DIFFBN,
        caerBiasCoarseFineGenerate(CF_N_TYPE(2, 39)));
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_ONBN,
        caerBiasCoarseFineGenerate(CF_N_TYPE(4, 200)));
    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_OFFBN,
        caerBiasCoarseFineGenerate(CF_N_TYPE(1, 62)));

    caerDeviceConfigSet(davis240_handle_, DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_REFRBP,
        caerBiasCoarseFineGenerate(CF_P_TYPE(3, 52)));

    caerDeviceDataStart(davis240_handle_, NULL, NULL, NULL, NULL, NULL);
    caerDeviceConfigSet(davis240_handle_, CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
    return true;
}

void DAVISCameraWorker::snap(void)
{
    snap_=true;
    caerDeviceConfigSet(davis240_handle_,DAVIS_CONFIG_APS,DAVIS_CONFIG_APS_SNAPSHOT,1);
}

void DAVISCameraWorker::deinit()
{
    caerDeviceDataStop(davis240_handle_);

    caerDeviceClose(&davis240_handle_);
}

#include "dvscameraworker.h"

void DVSCameraWorker::run()
{
    if(init()) {
        running_ = true;
        while(running_)
        {
            // get event and update timestamps
            caerEventPacketContainer packetContainer = caerDeviceDataGet(dvs128_handle_);
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
                }
            }
            caerEventPacketContainerFree(packetContainer);
            ugly_->addEvents(events_buffer_);
        }
        deinit();
    }
}

DVSCameraWorker::DVSCameraWorker(DenoisingWorker *worker):ugly_(worker)
{

}

bool DVSCameraWorker::init()
{
    // init camera
    // Open a DVS128, give it a device ID of 1, and don't care about USB bus or SN restrictions.
    dvs128_handle_ = caerDeviceOpen(1, CAER_DEVICE_DVS128, 0, 0, NULL);
    if (dvs128_handle_ == NULL) {
        return false;
    }
//    // Let's take a look at the information we have on the device.
//    struct caer_dvs128_info dvs128_info = caerDVS128InfoGet(dvs128_handle_);

//    printf("%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, Logic: %d.\n", dvs128_info.deviceString,
//        dvs128_info.deviceID, dvs128_info.deviceIsMaster, dvs128_info.dvsSizeX, dvs128_info.dvsSizeY,
//        dvs128_info.logicVersion);
    caerDeviceSendDefaultConfig(dvs128_handle_);

    // Values taken from DVS_FAST
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_CAS, 1992);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_DIFF, 13125);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_DIFFON, 209996);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_DIFFOFF, 132);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_FOLL, 271);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_INJGND, 1108364);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PR, 217);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PUX, 8159221);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PUY, 16777215);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_REFR, 969);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_REQ, 309590);
    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_REQPD, 16777215);
    // Values taken from DVS_SLOW
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_CAS, 54);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_DIFF, 30153);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_DIFFON, 482443);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_DIFFOFF, 132);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_FOLL, 51);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_INJGND, 1108364);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PR, 3);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PUX, 8159221);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PUY, 16777215);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_REFR, 6);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_REQ, 159147);
//    caerDeviceConfigSet(dvs128_handle_, DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_REQPD, 16777215);

    caerDeviceDataStart(dvs128_handle_, NULL, NULL, NULL, NULL, NULL);
    caerDeviceConfigSet(dvs128_handle_, CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
    return true;
}

void DVSCameraWorker::deinit()
{
    caerDeviceDataStop(dvs128_handle_);

    caerDeviceClose(&dvs128_handle_);
}


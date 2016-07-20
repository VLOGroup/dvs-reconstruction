#include <QMutexLocker>
#include <fstream>
#include "denoisingworker.h"
#include "common.cuh"
#include "common.h"
#include "cnpy.h"
#include "iu/iuio.h"
#include "iu/iumath.h"

DenoisingWorker::DenoisingWorker(int scale, int width, int height)
{
    scale_ = scale;
    width_ = width;
    height_ = height;
    output_ = new iu::ImageGpu_32f_C1(width_,height_);
    output_events_ = new iu::ImageGpu_32f_C1(width_,height_);
    input_ = new iu::ImageGpu_32f_C1(width_,height_);
    old_timestamp_ = new iu::ImageGpu_32f_C1(width_,height_);
    occurences_ = new iu::ImageGpu_32u_C1(width_,height_);

    output_disp_ = new iu::ImageGpu_32f_C1(width_*scale,height_*scale);
    output_events_disp_ = new iu::ImageGpu_32f_C1(width_*scale,height_*scale);
    output_time_disp_ = new iu::ImageGpu_32f_C1(width_*scale,height_*scale);
    initial_output_ = NULL;
    C1_ = 1.15;
    C2_ = 1.3;
    lambda_ = 90;
    lambda_t_ = 2;
    events_per_image_=1000;
    image_skip_ = 50;
    iterations_ = 50;
    u0_ = 1.5f;
    u_min_ = 1.f;
    u_max_ = 2.f;
    method_ = TV_LogEntropy;

    debug_ = false;
}

void DenoisingWorker::addEvents(std::vector<Event> &events)
{
    QMutexLocker lock(&mutex_events_);
    for(int i=0;i<events.size();i++) {
        events_.push(events[i]);
        all_events_.push_back(events[i]);
    }
}

void DenoisingWorker::saveCurrentState(std::string filename)
{
    saveState(filename,output_);
    saveState(filename + "events",output_events_);
    saveState(filename + "time",old_timestamp_);
}

void DenoisingWorker::saveEvents(std::string filename)
{
    ::saveEvents(filename,all_events_);
}

void DenoisingWorker::loadInitialState(std::string filename)
{
    if(initial_output_)
        delete initial_output_;
    loadState(filename,initial_output_,u_min_);
}

void DenoisingWorker::setOutput(iu::ImageCpu_32f_C1* image)
{
    assert(image->height() == height_ && image->width() == width_);
    iu::copy(image,output_);
    // make sure the image is within range (no contrast stretching)
    float minVal,maxVal;
    iu::math::minMax(*output_,minVal,maxVal);
    printf("Min: %f, Max: %f\n",minVal,maxVal);
    iu::math::addC(*output_,u_min_-minVal,*output_);
    iu::math::fill(*old_timestamp_,0.f);
    iu::copy(output_,input_);
}

void DenoisingWorker::run()
{
    if(initial_output_)
        iu::copy(initial_output_,output_);
    else
        iu::math::fill(*output_,u0_);
    iu::copy(output_,input_);
    cuda::initDenoise(output_, old_timestamp_);
    iu::math::fill(*old_timestamp_,0.f);
    iu::math::fill(*occurences_,0);
    all_events_.clear();

    running_ = true;
    image_id_ = 0;
    //int event_id = 0;
    while(running_) {
        if(events_.size()>events_per_image_) {
            mutex_events_.lock();
            std::vector<Event> temp_events;
            for (int i=0;i<events_per_image_;i++) {
                temp_events.push_back(events_.front());
                events_.pop();
            }
            mutex_events_.unlock();
            denoise(temp_events);
        } else
            msleep(1);
    }
}

void DenoisingWorker::stop()
{
    running_ = false;
    clearEvents();
}

void DenoisingWorker::denoise(std::vector<Event> &events)
{
    iu::IuCudaTimer timer;

    // create host memory for the event data
    iu::LinearHostMemory_32f_C4 events_host(events.size());
    for (int i=0;i<events.size();i++) {
        *events_host.data(i) = make_float4(events[i].x,events[i].y,events[i].polarity,events[i].t);
    }
    cuda::setEvents(input_,old_timestamp_,&events_host,C1_,C2_);
//    cuda::setEvents(input_,old_timestamp_,occurences_,&events_host,C1_,C2_);
    timer.start();
    cuda::solveTVIncrementalManifold(output_,input_,old_timestamp_,lambda_,lambda_t_,iterations_,u_min_,u_max_,method_);
//    cuda::solveTVIncrementalManifoldAdaptiveLambda(output_,input_,old_timestamp_,events[events.size()-1].t,lambda_,lambda_t_,iterations_,u_min_,u_max_,method_);
//    cuda::solveTVIncrementalManifoldOccurenceLambda(output_,input_,old_timestamp_,occurences_,lambda_,lambda_t_,iterations_,u_min_,u_max_,method_);
    if((image_id_++ % image_skip_)==0) {

        float minVal,maxVal;
        double wallclock=timer.elapsed();
        cuda::upsample(output_,output_disp_,cuda::UPSAMPLE_LINEAR);

        emit update_info(tr("Time/image: %1ms (= %2 fps)").arg(wallclock).arg(1000.f/wallclock),0);
        emit update_output(output_disp_,u_min_,u_max_);
        iu::math::fill(*output_events_,u0_);
        cuda::setEvents(output_events_,&events_host,C1_,C2_);
        cuda::upsample(output_events_,output_events_disp_,cuda::UPSAMPLE_NEAREST);
        emit update_events(output_events_disp_,u0_/C2_,u0_*C1_);
        iu::math::minMax(*old_timestamp_,minVal,maxVal);
        cuda::upsample(old_timestamp_,output_time_disp_,cuda::UPSAMPLE_NEAREST);
        emit update_time(output_time_disp_,minVal,maxVal);
        if(debug_){
            QString filename = tr("%1").arg((short int)(image_id_/image_skip_),5,10,QChar('0'));
            saveState("image" + filename.toStdString(),output_,true,false);
            saveState("events" + filename.toStdString(),output_events_,true,false);
            saveState("time" + filename.toStdString(),old_timestamp_,true,false);
            // save timestamp of last event
            std::ofstream timestampfile("frametimestamps.txt",std::ofstream::app);
            timestampfile << events[events.size()-1].t << std::endl;
            timestampfile.close();
        }
    }
}

void DenoisingWorker::clearEvents()
{
    QMutexLocker lock(&mutex_events_);
    std::queue<Event> empty;
    std::swap(events_,empty);
}


void DenoisingWorker::updateLambda(double value)
{
    lambda_=value;
}

void DenoisingWorker::updateLambdaT(double value)
{
    lambda_t_=value;
}

void DenoisingWorker::updateC1(double value)
{
    C1_=value;
}

void DenoisingWorker::updateC2(double value)
{
    C2_=value;
}

void DenoisingWorker::updateU0(double value)
{
    u0_ = value;
    if(initial_output_)
        delete initial_output_;
    initial_output_=NULL;
    emit update_info(tr("reset u0 to %1").arg(value),0);
}


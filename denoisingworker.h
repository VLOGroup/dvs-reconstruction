#ifndef DENOISINGWORKER_H
#define DENOISINGWORKER_H

#include <QThread>
#include <queue>
#include <QMutex>

#include "iu/iucore.h"
#include "event.h"
#include "denoise.h"

class DenoisingWorker : public QThread
{
    Q_OBJECT
    void run() Q_DECL_OVERRIDE;
public:
    DenoisingWorker(int scale=1, int width=128, int height=128);
    void addEvents(std::vector<Event>& events);
    void setOutput(iu::ImageCpu_32f_C1* image);
    void saveCurrentState(std::string filename);
    void saveEvents(std::string filename);
    void loadInitialState(std::string filename);
    void setDataTerm(METHOD method){method_ = method;}

signals:
    void update_output(iu::ImageGpu_32f_C1*,float,float);
    void update_events(iu::ImageGpu_32f_C1*,float,float);
    void update_time(iu::ImageGpu_32f_C1*,float,float);
    void update_info(const QString&,int);

public slots:
    void stop();
    void updateLambda(double value);
    void updateLambdaT(double value);
    void updateC1(double value);
    void updateC2(double value);
    void updateEventsPerImage(int value){events_per_image_ = value;}
    void updateIterations(int value){iterations_ = value;}
    void updateImageSkip(int value){image_skip_ = value;}
    void updateU0(double value);
    void updateUMin(double value){u_min_ = value;}
    void updateUMax(double value){u_max_ = value;}
    void updateDebug(bool value){debug_ = value;}

protected:
    void denoise(std::vector<Event>& events);
    void clearEvents(void);

    double lambda_;
    double lambda_t_;
    double C1_;
    double C2_;
    bool running_;
    int image_id_;
    int scale_;
    int events_per_image_;
    int iterations_;
    int image_skip_;
    int width_;
    int height_;
    double u0_;
    double u_min_;
    double u_max_;
    bool debug_;
    METHOD method_;

    std::queue<Event>  events_;
    std::vector<Event>  all_events_;
    QMutex mutex_events_;
    iu::ImageGpu_32f_C1 *initial_output_;
    iu::ImageGpu_32f_C1 *output_;
    iu::ImageGpu_32f_C1 *output_disp_;
    iu::ImageGpu_32f_C1 *output_events_;
    iu::ImageGpu_32f_C1 *output_events_disp_;
    iu::ImageGpu_32f_C1 *old_timestamp_;
    iu::ImageGpu_32u_C1 *occurences_;
    iu::ImageGpu_32f_C1 *output_time_disp_;
    iu::ImageGpu_32f_C1 *input_;

};

#endif // DENOISINGWORKER_H

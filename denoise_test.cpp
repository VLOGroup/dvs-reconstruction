#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

#include <iu/iucore.h>
#include <iu/iuio.h>
#include <iu/iumath.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "denoise.h"
#include "common.cuh"
#include "common.h"
#include "cnpy.h"
#include "event.h"

using std::string;
using std::cout;
using std::endl;

namespace po = boost::program_options;

constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

enum {
    NONE,
    OCC,
    TIME
};

int main(int argc, char *argv[])
{
    // parameter parsing using boost program_options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("event-file,f", po::value<std::string>(), "File that contains events")
        ("initial-image,n", po::value<std::string>(), "File that contains u_0")
        ("events-per-image,e", po::value<int>()->default_value(1000),"Events per reconstructed frame")
        ("lambda-data,d", po::value<double>()->default_value(180.0),"Lambda for Data Term")
        ("lambda-time,t", po::value<double>()->default_value(2.0),"Lambda for Time Manifold")
        ("iterations,i", po::value<int>()->default_value(50),"Number of iterations per image")
        ("u-min", po::value<double>()->default_value(1.0),"Minimum value of reconstruction")
        ("u-max", po::value<double>()->default_value(2.0),"Maximum value of reconstruction")
        ("c1", po::value<double>()->default_value(1.15),"Positive threshold")
        ("c2", po::value<double>()->default_value(1.20),"Negative threshold")
        ("method,m", po::value<std::string>()->default_value("TVEntropy"), "Method. Possible options: TVL1, TVL2, TVLog2, TVEntropy")
        ("adaptive-lambda,a", po::value<std::string>()->default_value("NONE"), "Auto adapt lambda?. Possible options: NONE, OCC, TIME")
        ("width,w", po::value<int>()->default_value(128),"Width")
        ("height,h", po::value<int>()->default_value(128),"Height")
        ("output-folder,o", po::value<std::string>()->default_value("./"),"Folder where all the file will be written")
        ("gpu,g", po::value<int>()->default_value(0),"GPU")
    ;
    po::positional_options_description p;
    p.add("event-file", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (!vm.count("event-file")) {
        cout << desc  << "\n";
        return 1;
    }
    int width = vm["width"].as<int>();
    int height = vm["height"].as<int>();
    double lambda = vm["lambda-data"].as<double>();
    double lambda_t = vm["lambda-time"].as<double>();
    double u_min = vm["u-min"].as<double>();
    double u_max = vm["u-max"].as<double>();
    double C1 = vm["c1"].as<double>();
    double C2 = vm["c2"].as<double>();
    int iterations = vm["iterations"].as<int>();
    std::string outputfolder = vm["output-folder"].as<string>();
    METHOD method;
    switch(str2int(vm["method"].as<std::string>().c_str()))
    {
        case str2int("TVEntropy"):
            method = TV_LogEntropy;
            break;
        case str2int("TVL1"):
            method = TV_L1;
            break;
        case str2int("TVL2"):
            method = TV_L2;
            break;
        case str2int("TVLog2"):
            method = TV_LogL2;
            break;
        default:
            cout << desc  << "\n";
            return 1;
    }
    unsigned char adapt_lambda;
    switch(str2int(vm["adaptive-lambda"].as<std::string>().c_str()))
    {
        case str2int("NONE"):
            adapt_lambda = NONE;
            break;
        case str2int("OCC"):
            adapt_lambda = OCC;
            break;
        case str2int("TIME"):
            adapt_lambda = TIME;
            break;
        default:
            cout << desc  << "\n";
            return 1;
    }

    // read events from file
    std::vector <Event> events;
    loadEvents(vm["event-file"].as<std::string>(),events);

    // select gpu
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(vm["gpu"].as<int>()<numDevices)
        cudaSetDevice(vm["gpu"].as<int>());


    // prepare cuda memory
    iu::ImageGpu_32f_C1 input(width,height);
    iu::ImageGpu_32f_C1 manifold(width,height);
    iu::ImageGpu_32f_C1 output(width,height);
    iu::ImageGpu_32u_C1 occurences(width,height);

    iu::math::fill(occurences,0);
    cuda::initDenoise(&input,&manifold);
    iu::LinearHostMemory_32f_C4 events_host(vm["events-per-image"].as<int>());

    // prepare output folder
    boost::filesystem::create_directories(outputfolder);

    // iterate of the "images"
    int event_id=0;
    std::ofstream timestampfile(outputfolder + "/frametimestamps.txt");
    while(event_id<events.size()-events_host.length()){
        // always process packets
        for(int i=0;i<events_host.length();i++)
            *events_host.data(i) = make_float4(events[i+event_id].x,events[i+event_id].y,events[i+event_id].polarity,events[i+event_id].t);
        event_id += events_host.length();
        switch (adapt_lambda) {
            case NONE:
                cuda::setEvents(&input,&manifold,&events_host,C1,C2);
                cuda::solveTVIncrementalManifold(&output,&input,&manifold,lambda,lambda_t,iterations,u_min,u_max,method);
                break;
            case TIME:
                cuda::setEvents(&input,&manifold,&events_host,C1,C2);
                cuda::solveTVIncrementalManifoldAdaptiveLambda(&output,&input,&manifold,events[events.size()-1].t,lambda,lambda_t,iterations,u_min,u_max,method);
                break;
            case OCC:
                cuda::setEvents(&input,&manifold,&occurences,&events_host,C1,C2);
                cuda::solveTVIncrementalManifoldOccurenceLambda(&output,&input,&manifold,&occurences,lambda,lambda_t,iterations,u_min,u_max,method);
                break;
        }

        // save to file
        std::stringstream outfilename;
        outfilename << outputfolder << "/image" << std::setfill('0') << std::setw(6) << event_id/events_host.length();
        saveState(outfilename.str(),&output,true);
        timestampfile << events[event_id].t << std::endl;
    }
    timestampfile.close();
//
//    iu::ImageCpu_32f_C1 out_cpu(image.size());
//    IuSize sz = image.size();
//    const unsigned int shape[] = {sz.width,sz.height};
//    iu::copy(&output_gpu,&out_cpu);
//    // save current image as npy
//    cnpy::npy_save("output.npy",out_cpu.data(),shape,2);
//
//    iu::imshow(&output_gpu,"output",true);
//    cv::waitKey();

    return EXIT_SUCCESS;

}


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
        ("initial-image,n", po::value<std::string>()->default_value(""), "File that contains u_0")
        ("events-per-image,e", po::value<int>()->default_value(1000),"Events per reconstructed frame")
        ("lambda-data,d", po::value<double>()->default_value(180.0),"Lambda for Data Term")
        ("lambda-time,t", po::value<double>()->default_value(2.0),"Lambda for Time Manifold")
        ("iterations,i", po::value<int>()->default_value(50),"Number of iterations per image")
        ("u-min", po::value<double>()->default_value(1.0),"Minimum value of reconstruction")
        ("u-max", po::value<double>()->default_value(2.0),"Maximum value of reconstruction")
        ("c1", po::value<double>()->default_value(1.15),"Positive threshold")
        ("c2", po::value<double>()->default_value(1.20),"Negative threshold")
        ("method,m", po::value<std::string>()->default_value("TVEntropy"), "Method. Possible options: TVL1, TVL2, TVLog2, TVEntropy")
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
            method = TV_KLD;
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

    // read events from file
    std::vector <Event> events;
    loadEvents(vm["event-file"].as<std::string>(),events);

    // select gpu
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(vm["gpu"].as<int>()<numDevices)
        CudaSafeCall(cudaSetDevice(vm["gpu"].as<int>()));

    // prepare cuda memory
    iu::ImageGpu_32f_C1 input(width,height);
    iu::ImageGpu_32f_C1 manifold(width,height);
    iu::ImageGpu_32f_C1 output(width,height);
    iu::ImageGpu_32u_C1 occurences(width,height);

    iu::math::fill(occurences,0);
    iu::math::fill(input,(u_min+u_max)/2.f);
    iu::math::fill(output,(u_min+u_max)/2.f);
    cuda::initDenoise(&input,&manifold);
    iu::LinearHostMemory_32f_C4 events_host(vm["events-per-image"].as<int>());

    // see if we have an initial image
    std::string initial_image_filename = vm["initial-image"].as<std::string>();
    if(initial_image_filename.compare("")!=0) {
        loadState(initial_image_filename,&input,u_min);
        iu::copy(&input,&output);
    }

    // prepare output folder
    boost::filesystem::create_directories(outputfolder);

    // iterate over the "images"
    int event_id=0;
    std::ofstream timestampfile(outputfolder + "/frametimestamps.txt");
    while(event_id<events.size()-events_host.numel()){
        // always process packets
        for(int i=0;i<events_host.numel();i++)
            *events_host.data(i) = make_float4(events[i+event_id].x,events[i+event_id].y,events[i+event_id].polarity,events[i+event_id].t);
        event_id += events_host.numel();
        cuda::setEvents(&input,&manifold,&events_host,C1,C2);
        cuda::solveTVIncrementalManifold(&output,&input,&manifold,lambda,lambda_t,iterations,u_min,u_max,method);

        // save to file
        std::stringstream outfilename;
        outfilename << outputfolder << "/image" << std::setfill('0') << std::setw(6) << event_id/events_host.numel();
        saveState(outfilename.str(),&output,true,false);
        timestampfile << events[event_id].t << std::endl;
    }
    timestampfile.close();

    return EXIT_SUCCESS;

}


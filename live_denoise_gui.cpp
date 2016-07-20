

// system includes
#include <fstream>
#include <QApplication>
#include <QVBoxLayout>
#include "iu/iugui.h"

#include "event.h"
#include "scopedtimer.h"
#include "denoisingmainwindow.h"

int main(int argc, char**argv)
{
    // read events
    std::vector<Event> events;
//    readevents(events,argv[1]);

    // start main window

    QApplication app(argc, argv);
    DenoisingMainWindow window(NULL,events);
    window.show();

    return app.exec();



}

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

// system includes
#include <fstream>
#include <QApplication>
#include <QVBoxLayout>
#include "iu/iugui.h"

#include "event.h"
#include "reconstructionmainwindow.h"

int main(int argc, char**argv)
{
    // read events
    std::vector<Event> events;
//    readevents(events,argv[1]);

    // start main window

    QApplication app(argc, argv);
    ReconstructionMainWindow window(NULL,events);
    window.show();

    return app.exec();



}

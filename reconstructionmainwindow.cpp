#include "reconstructionmainwindow.h"
#include <QGridLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QApplication>
#include <QDockWidget>
#include <qmdisubwindow.h>
#include <QGroupBox>
#include <QMenuBar>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QToolBar>
#include <fstream>
#include "common.h"

ReconstructionMainWindow::ReconstructionMainWindow(QWidget *parent, std::vector<Event>& events) : QMainWindow(parent)
{
    flipud_ = false;
    skip_initial_ = false;
#ifdef DAVIS240
    int scale = 2;
    int width = 240;
    int height = 180;
    denoise_worker_ = new DenoisingWorker(scale,width,height);
    camera_worker_ = new DAVISCameraWorker(denoise_worker_);
#else
    int scale = 4;
    int width = 128;
    int height = 128;
    denoise_worker_ = new DenoisingWorker(scale,width,height);
    camera_worker_ = new DVSCameraWorker(denoise_worker_);
#endif


    events_ = events;

    mdi_area_ = new QMdiArea(this);
    setCentralWidget(mdi_area_);

    output_win_ = new iu::Qt5ImageGpuWidget(IuSize(width*scale,height*scale),this);
    events_win_ = new iu::Qt5ImageGpuWidget(IuSize(width*scale,height*scale),this);
    time_win_ = new iu::Qt5ImageGpuWidget(IuSize(width*scale,height*scale),this);
    QMdiSubWindow* window =  mdi_area_->addSubWindow(output_win_);
    window->setGeometry(QRect(0,0,width*scale+10,height*scale+40));
    window->setWindowTitle("Output Denoised");
    window->setWindowFlags(Qt::CustomizeWindowHint|Qt::WindowTitleHint);
    window->setMaximumSize(QSize(width*scale+10,height*scale+40));
    window->setMinimumSize(QSize(width*scale+10,height*scale+40));
    window =  mdi_area_->addSubWindow(time_win_);
    window->setGeometry(QRect(width*scale+10,0,width*scale+10,height*scale+40));
    window->setWindowTitle("Time Since Last Event");
    window->setWindowFlags(Qt::CustomizeWindowHint|Qt::WindowTitleHint);
    window->setMaximumSize(QSize(width*scale+10,height*scale+40));
    window->setMinimumSize(QSize(width*scale+10,height*scale+40));
    window =  mdi_area_->addSubWindow(events_win_);
    window->setGeometry(QRect(2*(width*scale+10),0,width*scale+10,height*scale+40));
    window->setWindowFlags(Qt::CustomizeWindowHint|Qt::WindowTitleHint);
    window->setMaximumSize(QSize(width*scale+10,height*scale+40));
    window->setMinimumSize(QSize(width*scale+10,height*scale+40));
    window->setWindowTitle("Current Events");
    status_bar_ = statusBar();
//    update_gl_timer_.setInterval(16);
//    connect(&update_gl_timer_,SIGNAL(timeout()),output_win_,SLOT(repaint()));

    show();

    while(!output_win_->isValid())    // wait until events are processed and window is created
        QApplication::instance()->processEvents();

    QSurfaceFormat fmt = output_win_->context()->format();
    if (fmt.profile() == QSurfaceFormat::NoProfile)
        printf("OpenGL profile: None/Unknown\n");
    if (fmt.profile() == QSurfaceFormat::CoreProfile)
        printf("OpenGL profile: Core\n");
    if (fmt.profile() == QSurfaceFormat::CompatibilityProfile)
        printf("OpenGL profile: Comaptibility\n");
    printf("OpenGL version: %d.%d\n", fmt.majorVersion(), fmt.minorVersion());
    printf("Swap behaviour: %d\n", fmt.swapBehavior());
    printf("Render type: %d\n", fmt.renderableType());

    dock_ = new QDockWidget("Parameters", this);
    QGridLayout* layout = new QGridLayout;
    spin_lambda_ = new QDoubleSpinBox;
    spin_lambda_->setMinimum(0.1);
    spin_lambda_->setMaximum(10000);
    spin_lambda_->setSingleStep(10);
    spin_lambda_->setValue(90);
    spin_lambda_t_ = new QDoubleSpinBox;
    spin_lambda_t_->setMinimum(0.);
    spin_lambda_t_->setMaximum(1000);
    spin_lambda_t_->setSingleStep(0.01);
    spin_lambda_t_->setValue(2);
    spin_C1_ = new QDoubleSpinBox;
    spin_C1_->setMinimum(0.0);
    spin_C1_->setSingleStep(0.05);
    spin_C1_->setValue(1.15);
    spin_C2_ = new QDoubleSpinBox;
    spin_C2_->setMinimum(0.0);
    spin_C2_->setSingleStep(0.05);
    spin_C2_->setValue(1.3);
    spin_u0_ = new QDoubleSpinBox;
    spin_u0_->setMinimum(1e-3);
    spin_u0_->setSingleStep(1e-3);
    spin_u0_->setValue(1.5f);
    spin_u_min_ = new QDoubleSpinBox;
    spin_u_min_->setMinimum(1e-3);
    spin_u_min_->setSingleStep(1e-3);
    spin_u_min_->setValue(1.f);
    spin_u_max_ = new QDoubleSpinBox;
    spin_u_max_->setMinimum(1e-3);
    spin_u_max_->setSingleStep(1e-3);
    spin_u_max_->setValue(2.f);
    spin_events_per_image_ = new QSpinBox;
    spin_events_per_image_->setMinimum(1);
    spin_events_per_image_->setMaximum(10000);
    spin_events_per_image_->setValue(1000);
    spin_events_per_image_->setSingleStep(100);
    spin_image_skip_ = new QSpinBox;
    spin_image_skip_->setMinimum(1);
    spin_image_skip_->setMaximum(10000);
    spin_image_skip_->setValue(50);
    spin_image_skip_->setSingleStep(1);
    spin_iterations_ = new QSpinBox;
    spin_iterations_->setMinimum(1);
    spin_iterations_->setMaximum(1000);
    spin_iterations_->setValue(50);
    spin_iterations_->setSingleStep(1);
    action_start_ = new QAction(QIcon(":play.png"),tr("&Start algorithm"),this);
    action_stop_ = new QAction(QIcon(":pause.png"),tr("S&top algorithm"),this);
    action_camera_ = new QAction(QIcon(":camera.png"),tr("St&art camera"),this);
    check_flipud_events_ = new QCheckBox("Flip U/D?");
    check_flipud_events_->setChecked(false);
    check_flipud_events_->setToolTip("Flip y-axis when loading a recorded sequence");
    check_skip_events_ = new QCheckBox("Skip initial on events?");
    check_skip_events_->setChecked(false);
    check_skip_events_->setToolTip("Throw away the first ON events when loading a recorded sequence");
    check_debug_mode_ = new QCheckBox("Debug Mode?");
    check_debug_mode_->setChecked(false);
    check_debug_mode_->setToolTip("Save all immediate images to executable folder");

    QGroupBox* parameters = new QGroupBox;
    QLabel* label_lambda = new QLabel("Lambda events:");
    spin_lambda_->setToolTip("Influence of new events (increase if using less events per image)");
    QLabel* label_lambda_t = new QLabel("Lambda manifold:");
    spin_lambda_t_->setToolTip("Influence of the event manifold with respect to image resolution");
    QLabel* label_C1 = new QLabel("C1:");
    spin_C1_->setToolTip("Positive threshold of event camera");
    QLabel* label_C2 = new QLabel("C2:");
    spin_C2_->setToolTip("Negative threshold of event camera");
    QLabel* label_u0 = new QLabel("u0:");
    spin_u0_->setToolTip("Initial gray value of reconstruction");
    QLabel* label_u_min = new QLabel("u_min:");
    spin_u_min_->setToolTip("Minimum gray value of reconstruction");
    QLabel* label_u_max = new QLabel("u_max:");
    spin_u_max_->setToolTip("Maximum gray value of reconstruction");
    QLabel* label_iterations = new QLabel("Iterations:");
    spin_iterations_->setToolTip("Optimization iterations per reconstructed image (k in the paper)");
    QLabel* label_image_skip = new QLabel("Show every nth image:");
    spin_image_skip_->setToolTip("Only display every xth image on screen");
    QLabel* label_events_per_image = new QLabel("Events/image:");
    spin_events_per_image_->setToolTip("Accumulate x events before reconstructing an image");
    QSpacerItem *space = new QSpacerItem(1,1,QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

    combo_algorithm_ = new QComboBox();
    combo_algorithm_->addItem("TV Log Entropy");
    combo_algorithm_->addItem("TV Log L2");
    combo_algorithm_->addItem("TV L2");
    combo_algorithm_->addItem("TV L1");
    QLabel* label_algorithm = new QLabel("Data Term");
    combo_algorithm_->setToolTip("Select the data term (see paper)");

    layout->addWidget(label_lambda,           0, 0, 1, 1);
    layout->addWidget(spin_lambda_,           0, 1, 1, 1);
    layout->addWidget(label_C1,               1, 0, 1, 1);
    layout->addWidget(spin_C1_,               1, 1, 1, 1);
    layout->addWidget(label_C2,               2, 0, 1, 1);
    layout->addWidget(spin_C2_,               2, 1, 1, 1);
    layout->addWidget(label_events_per_image, 3, 0, 1, 1);
    layout->addWidget(spin_events_per_image_, 3, 1, 1, 1);
    layout->addWidget(label_image_skip,       4, 0, 1, 1);
    layout->addWidget(spin_image_skip_,       4, 1, 1, 1);
    layout->addItem(space,                    5, 0,-1,-1);

    parameters->setLayout(layout);
    dock_->setWidget(parameters);
    dock_->setFeatures(QDockWidget::NoDockWidgetFeatures);
    dock_->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Minimum);
    addDockWidget(Qt::LeftDockWidgetArea, dock_);
    QSpacerItem *advanced_space = new QSpacerItem(1,1,QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

    // Advanced parameters
    advanced_dock_ = new QDockWidget("Advanced parameters", this);
	QGridLayout* advanced_layout = new QGridLayout;
	QGroupBox* advanced_parameters = new QGroupBox;
	advanced_layout->addWidget(label_lambda_t,         1, 0, 1, 1);
	advanced_layout->addWidget(spin_lambda_t_,         1, 1, 1, 1);
	advanced_layout->addWidget(label_u0,               2, 0, 1, 1);
	advanced_layout->addWidget(spin_u0_,               2, 1, 1, 1);
	advanced_layout->addWidget(label_u_min,            3, 0, 1, 1);
	advanced_layout->addWidget(spin_u_min_,            3, 1, 1, 1);
	advanced_layout->addWidget(label_u_max,            4, 0, 1, 1);
	advanced_layout->addWidget(spin_u_max_,            4, 1, 1, 1);
	advanced_layout->addWidget(label_iterations,       5, 0, 1, 1);
	advanced_layout->addWidget(spin_iterations_,       5, 1, 1, 1);
	advanced_layout->addWidget(check_flipud_events_,   6, 0, 1, 2);
	advanced_layout->addWidget(check_skip_events_,     7, 0, 1, 2);
	advanced_layout->addWidget(check_debug_mode_,      8, 0, 1, 2);
    advanced_layout->addWidget(label_algorithm,        9, 0, 1, 1);
    advanced_layout->addWidget(combo_algorithm_,       9, 1, 1, 1);
    advanced_layout->addItem(advanced_space,          10, 0,-1,-1);

	advanced_parameters->setLayout(advanced_layout);
	advanced_dock_->setWidget(advanced_parameters);
	advanced_dock_->setFeatures(QDockWidget::NoDockWidgetFeatures);
	advanced_dock_->setVisible(false);
    advanced_dock_->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::Minimum);
    addDockWidget(Qt::LeftDockWidgetArea, advanced_dock_);

    connect(spin_lambda_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateLambda(double)));
    connect(spin_lambda_t_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateLambdaT(double)));
    connect(spin_C1_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateC1(double)));
    connect(spin_C2_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateC2(double)));
    connect(spin_u0_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateU0(double)));
    connect(spin_u_min_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateUMin(double)));
    connect(spin_u_max_,SIGNAL(valueChanged(double)),denoise_worker_,SLOT(updateUMax(double)));
    connect(spin_events_per_image_,SIGNAL(valueChanged(int)),denoise_worker_,SLOT(updateEventsPerImage(int)));
    connect(spin_image_skip_,SIGNAL(valueChanged(int)),denoise_worker_,SLOT(updateImageSkip(int)));
    connect(spin_iterations_,SIGNAL(valueChanged(int)),denoise_worker_,SLOT(updateIterations(int)));
    connect(denoise_worker_,SIGNAL(update_output(iu::ImageGpu_32f_C1*,float,float)),output_win_,SLOT(update_image(iu::ImageGpu_32f_C1*,float,float)));
    connect(denoise_worker_,SIGNAL(update_events(iu::ImageGpu_32f_C1*,float,float)),events_win_,SLOT(update_image(iu::ImageGpu_32f_C1*,float,float)));
    connect(denoise_worker_,SIGNAL(update_time(iu::ImageGpu_32f_C1*,float,float)),time_win_,SLOT(update_image(iu::ImageGpu_32f_C1*,float,float)));
    connect(denoise_worker_,SIGNAL(update_info(const QString&,int)),status_bar_,SLOT(showMessage(const QString&,int)));
    connect(action_start_,SIGNAL(triggered(bool)),this,SLOT(startDenoising()));
    connect(action_stop_,SIGNAL(triggered(bool)),this,SLOT(stopDenoising()));
    connect(action_camera_,SIGNAL(triggered(bool)),this,SLOT(startCamera()));
    connect(check_flipud_events_,SIGNAL(clicked(bool)),this,SLOT(setFlipUD(bool)));
    connect(check_skip_events_,SIGNAL(clicked(bool)),this,SLOT(setSkipInitial(bool)));
    connect(check_debug_mode_,SIGNAL(clicked(bool)),denoise_worker_,SLOT(updateDebug(bool)));
    connect(combo_algorithm_,SIGNAL(currentIndexChanged(int)),this,SLOT(changeDataTerm(int)));

    // Menu Stuff
    action_open_ = new QAction(QIcon(":fileopenevents.png"),tr("&Load events from file"),this);
    connect(action_open_,SIGNAL(triggered(bool)),this,SLOT(loadEvents()));
    action_load_state_ = new QAction(QIcon(":fileopen.png"),tr("L&oad initial state"),this);
    connect(action_load_state_,SIGNAL(triggered(bool)),this,SLOT(loadState()));
    action_save_state_ = new QAction(QIcon(":filesave.png"),tr("&Save current state"),this);
    connect(action_save_state_,SIGNAL(triggered(bool)),this,SLOT(saveState()));
    action_save_events_ = new QAction(QIcon(":filesaveevents.png"),tr("Sa&ve events to file"),this);
    connect(action_save_events_,SIGNAL(triggered(bool)),this,SLOT(saveEvents()));
#ifdef DAVIS240
    action_snap_ = new QAction(QIcon(":reset.png"),tr("Snap DAVIS frame"),this);
    connect(action_snap_,SIGNAL(triggered(bool)),camera_worker_,SLOT(snap(void)));
#endif
    action_view_advanced_ = new QAction(tr("&Advanced options"),this);
    action_view_advanced_->setCheckable(true);
    action_view_advanced_->setChecked(false);
    connect(action_view_advanced_,SIGNAL(triggered(bool)),this,SLOT(toggleAdvancedParameters()));
    action_view_about_ = new QAction(tr("Abo&ut"),this);
    connect(action_view_about_,SIGNAL(triggered(bool)),this,SLOT(showAbout()));

    menu_file_ = menuBar()->addMenu(tr("&File"));
    menu_file_->addAction(action_open_);
    menu_file_->addAction(action_save_events_);
    menu_file_->addAction(action_load_state_);
    menu_file_->addAction(action_save_state_);

    menu_view_ = menuBar()->addMenu(tr("&View"));
    menu_view_->addAction(action_view_advanced_);
    menu_view_->addAction(action_view_about_);

    QToolBar *toolbar = new QToolBar;
    toolbar->setMovable(false);
    toolbar->addAction(action_start_);
    toolbar->addAction(action_stop_);
    toolbar->addAction(action_camera_);
#ifdef DAVIS240
    toolbar->addAction(action_snap_);
#endif
    toolbar->addSeparator();
    toolbar->addAction(action_open_);
    toolbar->addAction(action_save_events_);
    toolbar->addAction(action_load_state_);
    toolbar->addAction(action_save_state_);
    addToolBar(Qt::LeftToolBarArea,toolbar);

    // Window title
    setWindowTitle("Real-Time DVS Denoising");
    setGeometry(QRect(0,0,(width*scale+10)*3+advanced_dock_->geometry().width()+220,(height*scale+10)+90));
    setWindowIcon(QIcon(":vlo_logo.png"));
}

ReconstructionMainWindow::~ReconstructionMainWindow()
{
    stopDenoising();
    denoise_worker_->wait();
    camera_worker_->wait();
}

void ReconstructionMainWindow::startDenoising()
{
    denoise_worker_->stop();
    if(events_.empty()) { // start camera thread
        camera_worker_->start();
    } else {
        denoise_worker_->addEvents(events_);
    }
    denoise_worker_->start();
}

void ReconstructionMainWindow::stopDenoising()
{
    denoise_worker_->stop();
    camera_worker_->stop();
}

void ReconstructionMainWindow::startCamera()
{
    events_.clear();
    startDenoising();
}

void ReconstructionMainWindow::loadEvents()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Event File"), "", tr("Event Files (*.aer2 *.dat)"));
    status_bar_->showMessage("Loading...",0);
    readevents(fileName.toStdString(),skip_initial_,flipud_);
}

void ReconstructionMainWindow::loadState()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open State File"), "/home/christian/data/testdata/event_camera_testdata/own_recordings", tr("Image Files (*.npy *.png)"));
    denoise_worker_->loadInitialState(fileName.toStdString());
    status_bar_->showMessage("Loaded u0",0);
}

void ReconstructionMainWindow::saveState()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    tr("Save State File"),"/home/christian/data/testdata/event_camera_testdata",tr("All Files, no ext (*.*)"));
    denoise_worker_->saveCurrentState(fileName.toStdString());
}

void ReconstructionMainWindow::saveEvents()
{
    stopDenoising();
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    tr("Save Events to File"),"/home/christian/data/testdata/event_camera_testdata",tr("Event Files (*.aer2)"));
    denoise_worker_->saveEvents(fileName.toStdString());
}

void ReconstructionMainWindow::showAbout()
{
    QMessageBox::about(this,"About","Demo application for our publication\n"
                                    "Real-Time Image Reconstruction for Event Cameras using Manifold Regularization\n"
                                    "(c) Institute for Computer Graphics and Vision\n"
                                    "    Vision, Learning and Optimization Group, 2016\n");
}

void ReconstructionMainWindow::toggleAdvancedParameters() {
    advanced_dock_->setVisible(action_view_advanced_->isChecked());
}

void ReconstructionMainWindow::changeDataTerm(int value)
{
    switch(value) {
        case 0: denoise_worker_->setDataTerm(TV_LogEntropy); break;
        case 1: denoise_worker_->setDataTerm(TV_LogL2); break;
        case 2: denoise_worker_->setDataTerm(TV_L2); break;
        case 3: denoise_worker_->setDataTerm(TV_L1); break;
    }
}

void ReconstructionMainWindow::readevents(std::string filename,bool skip_events, bool flip_ud)
{
    QFileInfo info(filename.c_str());
    events_.clear();
    if(info.suffix()=="aer2") {
        ::loadEvents(filename,events_,skip_events,flip_ud);
    } else if(info.suffix()=="dat"){ // read Bardow files
        Event temp_event;
        float first_timestamp=0;
        float time;
        float last_timestamp=0;
        bool normalize_time=true;
        std::ifstream ifs;
        ifs.open(filename.c_str(),std::ios::in|std::ios::binary);
        if(ifs.good()) {
            unsigned int data;

            while(!ifs.eof()) {
                ifs.read((char*)&data,4);
                time = data;
//                if(first_timestamp==0) {
//                    first_timestamp=time;
//                }
                time-=first_timestamp;
                ifs.read((char*)&data,4);
                temp_event.x = (data & 0x000001FF);
                temp_event.y = (data & 0x0001FE00) >> 9;
                temp_event.polarity = (data & 0x00020000) >> 17;
//                if(flip_ud)
//                    temp_event.y = 127-temp_event.y;
                temp_event.t = time*TIME_CONSTANT;
                temp_event.polarity=temp_event.polarity>0?1:-1;
                events_.push_back(temp_event);
            }
            ifs.close();
        }
    }
    status_bar_->showMessage(tr("Loaded a file with %1 events").arg(events_.size()),0);
}

ReconstructionMainWindow::ReconstructionMainWindow()
{

}

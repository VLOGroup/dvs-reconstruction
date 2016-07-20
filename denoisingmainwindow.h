#ifndef DENOISINGMAINWINDOW_H
#define DENOISINGMAINWINDOW_H

#include <QMainWindow>
#include <QDoubleSpinBox>
#include <QStatusBar>
#include <QMdiArea>
#include <QPushButton>
#include <QTimer>
#include <QAction>
#include <QMenu>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>

#include <vector>
#include "event.h"
#include "iu/iugui.h"
#include "denoisingworker.h"
#ifdef DAVIS240
#include "daviscameraworker.h"
#else
#include "dvscameraworker.h"
#endif

class DenoisingMainWindow : public QMainWindow
{
    Q_OBJECT
  public:
    DenoisingMainWindow();
    DenoisingMainWindow(QWidget *parent, std::vector<Event>& events);
    ~DenoisingMainWindow();

  protected slots:
    void startDenoising();
    void stopDenoising();
    void startCamera();
    void loadEvents();
    void loadState();
    void saveState();
    void saveEvents();
    void showAbout();
    void toggleAdvancedParameters();
    void setFlipUD(bool value){flipud_=value;}
    void setSkipInitial(bool value){skip_initial_ = value;}
    void changeDataTerm(int value);

  protected:
    void readevents(std::string filename, bool skip_events=true, bool flip_ud=true);

    iu::Qt5ImageGpuWidget *output_win_;
    iu::Qt5ImageGpuWidget *time_win_;
    iu::Qt5ImageGpuWidget *events_win_;
    std::vector<Event> events_;
    DenoisingWorker *denoise_worker_;
#ifdef DAVIS240
    DAVISCameraWorker *camera_worker_;
#else
    DVSCameraWorker *camera_worker_;
#endif

    QMdiArea *mdi_area_;
    QStatusBar *status_bar_;
    QTimer update_gl_timer_;
    QDockWidget* dock_;
    QDockWidget* advanced_dock_;

    QDoubleSpinBox *spin_lambda_;
    QDoubleSpinBox *spin_lambda_t_;
    QDoubleSpinBox *spin_C1_;
    QDoubleSpinBox *spin_C2_;
    QDoubleSpinBox *spin_u_min_;
    QDoubleSpinBox *spin_u_max_;
    QDoubleSpinBox *spin_u0_;
    QComboBox *combo_algorithm_;

    QSpinBox *spin_iterations_;
    QSpinBox *spin_image_skip_;
    QSpinBox *spin_events_per_image_;
    QCheckBox *check_skip_events_;
    QCheckBox *check_flipud_events_;

    QAction *action_start_;
    QAction *action_stop_;
    QAction *action_camera_;
#ifdef DAVIS240
    QAction *action_snap_;
#endif
    QCheckBox *check_debug_mode_;

    QMenu *menu_file_;
    QAction *action_open_;
    QAction *action_load_state_;
    QAction *action_save_state_;
    QAction *action_save_events_;
    QMenu *menu_view_;
    QAction *action_view_advanced_;
    QAction *action_view_about_;

    bool flipud_;
    bool skip_initial_;
    bool simple_mode_;
};

#endif // DENOISINGMAINWINDOW_H

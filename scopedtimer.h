/*
 * scopedtimer.h
 *
 *  Created on: July 18, 2011
 *      Author: Peter Innerhofer
 */

#ifndef SCOPED_TIMER_HPP
#define SCOPED_TIMER_HPP

#include <iostream>

#if !defined(WIN32)
# include <sys/time.h>
#else
# include <windows.h>
#endif
#include <time.h>



//! A class measuring the time until leaving the scope.
/*! A typical usage is like:
 * \code
 * {
 *    Timer t("Time used in scope: ");
 *    // do anything
 * }
 * \endcode
 */
class ScopedTimer
{
public:
      //! Returns the current time in us.
      static inline double
      getCurrentTime()
      {
#if defined(WIN32)
         LARGE_INTEGER t, freq;
         ::QueryPerformanceFrequency(&freq);
         ::QueryPerformanceCounter(&t);
         return 1000000.0 * t.QuadPart / freq.QuadPart;
#else
         timeval tv;
         gettimeofday(&tv, 0);
         return tv.tv_sec * 1000000.0 + tv.tv_usec;
#endif
      }

      //! Constructs the timer with a given message and starts timing.
      ScopedTimer(char const * msg)
         : _msg(msg), _accum(0), _elapsed(0)
      {
         _start = getCurrentTime();
      }

      //! Constructs the timer with an accumulator and starts timing.
      ScopedTimer(double& accum)
         : _msg(0), _accum(&accum), _elapsed(0)
      {
         _start = getCurrentTime();
      }

      ScopedTimer()
         : _msg(0), _accum(0), _elapsed(0)
      {
         _start = getCurrentTime();
      }

      //! Destructs the timer and emits a message and/or updates the accumulator.
      ~ScopedTimer()
      {
         double const end = getCurrentTime();
         if (_msg != 0)
         {
            std::cout << _msg << (end-_start)/1000000 << "sec" << std::endl;
         }
         if (_accum != 0)
         {
            *_accum += (end-_start)/1000000;
         }
      }

      void printTime(char const * msg2 = 0)
      {
         double const end = getCurrentTime();
         if (_msg != 0 && msg2 != 0)
         {
             std::cout << _msg << " " << msg2 << (end-_start)/1000000 << "sec" << std::endl;
         }
         else if (_msg != 0)
         {
             std::cout << _msg << (end-_start)/1000000 << "sec" << std::endl;
         }

      }

      void start()
      {
         _elapsed=0;
         _start = getCurrentTime();
      }

      void stop()
      {
         if (_start != 0)
         {
            _elapsed+=getCurrentTime()-_start;
            _start=0;
         }
      }

      void cont()
      {
         _start = getCurrentTime();
      }

      double getElapsedSeconds()
      {
         double time(getCurrentTime());
         if (_start != 0) _elapsed+=time-_start;
         _start=time;
         return(_elapsed/1000000.0);
      }

   protected:
      char const   * _msg;
      double         _start;
      double       * _accum;
      double         _elapsed;
};


#endif

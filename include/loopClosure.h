# pragma once

#include "slamBase.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/types_slam3d.h>// vertex type

#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

enum FRAME_CHECK_RESULT{NO_MATCH = 0, TOO_FAR, TOO_CLOSE, KEY_FRAME};

FRAME_CHECK_RESULT checkKeyFrame( FRAME & f1, FRAME & f2, g2o::SparseOptimizer &optimizer, bool isLoop=false);

void checkNearLoop(vector<FRAME> & frames, FRAME & currFrame,  g2o::SparseOptimizer & optimizer);

void checkRandomLoop(vector<FRAME> & frames, FRAME & currFrame,  g2o::SparseOptimizer & optimizer);
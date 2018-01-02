#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"
#include "loopClosure.h"

//g2o headers
#include <g2o/types/slam3d/types_slam3d.h>// vertex type
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

int main( int argc, char** argv )
{
    ParameterReader parameters;
    int startId  =   atoi( parameters.getData( "start_index" ).c_str() );
    int endId    =   atoi( parameters.getData( "end_index"   ).c_str() );

    // initialize
    cout<<"Start initializing ..."<<endl;
    int currId = startId;
    vector<FRAME> keyFrames;
    FRAME lastFrame = readFrame( currId, parameters ); 
    // last and current comparison

    string detector = parameters.getData( "detector" );
    string descriptor = parameters.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS intrinPara = getDefaultCamera();
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );

    keyFrames.push_back(lastFrame);
    //PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, intrinPara );
    
    //pcl::visualization::CloudViewer viewer("viewer");

    // pointcloud
    //bool needVisualize = parameters.getData("visualize_pointcloud")==string("yes");

    typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
    typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 

    // initialize solver
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3* vertex = new g2o::VertexSE3();
    vertex->setId( currId );
    vertex->setEstimate( Eigen::Isometry3d::Identity());// first estimate is identity matrix
    vertex->setFixed( true); // initial position is fixed
    optimizer.addVertex (vertex);

    int lastId = currId;

    bool checkLoop = parameters.getData("check_loop_closure")==string("yes");

    for (++currId ; currId <= endId; ++currId)
    {
        cout<<"file #"<<currId<<endl;
        FRAME currFrame = readFrame( currId, parameters );
        computeKeyPointsAndDesp( currFrame, detector, descriptor );
        FRAME_CHECK_RESULT result = checkKeyFrame(keyFrames.back(), currFrame, optimizer);

        if(result == KEY_FRAME)
        {
            cout<<"successfully added one more frame"<<endl;
            if(checkLoop)
            {
                checkNearLoop(keyFrames, currFrame, optimizer);
                checkRandomLoop(keyFrames, currFrame, optimizer);
            }
            keyFrames.push_back(currFrame);
        }
        else if(result == TOO_CLOSE)
        {
            cout<<"two frames are too close, not key frame"<<endl;
        }
        else if(result == TOO_FAR)
        {
            cout<<"too far away frames, error discarded"<<endl;
        }
        else
        {
            cout<<"too less inliers, error discarded"<<endl;
        }
    }

    cout<<"start graph optimization, total vertices:"<<optimizer.vertices().size()<<endl;
    optimizer.save("./data/result_before.g2o");
    optimizer.initializeOptimization();
    optimizer.optimize( 50 );
    optimizer.save("./data/result_after.g2o");

    optimizer.clear();

    return 0;
}

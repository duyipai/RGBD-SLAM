#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"


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

// first part api
// 给定index，读取一帧数据
FRAME readFrame( int index, ParameterReader& pd );
// 度量运动的大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );

int main( int argc, char** argv )
{
    ParameterReader parameters;
    int startId  =   atoi( parameters.getData( "start_index" ).c_str() );
    int endId    =   atoi( parameters.getData( "end_index"   ).c_str() );

    // initialize
    cout<<"Start initializing ..."<<endl;
    int currId = startId;
    FRAME lastFrame = readFrame( currIdx, parameters ); 
    // last and current comparison

    string detector = parameters.getData( "detector" );
    string descriptor = parameters.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS intrinPara = getDefaultCamera();
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );
    PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, intrinPara );
    
    pcl::visualization::CloudViewer viewer("viewer");

    // 是否显示点云
    bool needVisualize = parameters.getData("visualize_pointcloud")==string("yes");

    int min_inliers = atoi( parameters.getData("min_inliers").c_str() );
    double max_norm = atof( parameters.getData("max_norm").c_str() );

    // start g2o optimization
    // initlization of solvers
    g2o::LinearSolverCSparse< g2o::BlockSolver_6_3::PoseMatrixType> * linearSolver = 
        new g2o::LinearSolverCSparse< g2o::BlockSolver_6_3::PoseMatrixType>();
    linearSolver->setBlockOrdering( false);
    g2o::BlockSolver_6_3 * blockSolver = new g2o::BlockSolver_6_3 ( linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3* vertex = new g2o::VertexSE3();
    vertex->setId( currId );
    vertex->setEstimate( Eigen::Isometry3d::Identity());// first estimate is identity matrix
    v->setFixed( true); // initial position is fixed
    optimizer.addVertex (vertex);

    int lastId = currId;

    for (++currId ; currId <= endId; ++currId)
    {
        cout<<"file #"<<currId<<endl;
        
    }
    return 0;
}

// first part api
FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");
    
    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    return f;
}

// first part api
double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}
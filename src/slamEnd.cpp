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
FRAME readFrame( int index, ParameterReader& pd );

double normofTransform( cv::Mat rvec, cv::Mat tvec );

int main( int argc, char** argv )
{
    ParameterReader parameters;
    int startId  =   atoi( parameters.getData( "start_index" ).c_str() );
    int endId    =   atoi( parameters.getData( "end_index"   ).c_str() );

    // initialize
    cout<<"Start initializing ..."<<endl;
    int currId = startId;
    FRAME lastFrame = readFrame( currId, parameters ); 
    // last and current comparison

    string detector = parameters.getData( "detector" );
    string descriptor = parameters.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS intrinPara = getDefaultCamera();
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );
    PointCloud::Ptr cloud = image2PointCloud( lastFrame.rgb, lastFrame.depth, intrinPara );
    
    pcl::visualization::CloudViewer viewer("viewer");

    // pointcloud
    bool needVisualize = parameters.getData("visualize_pointcloud")==string("yes");

    int min_inliers = atoi( parameters.getData("min_inliers").c_str() );
    double max_norm = atof( parameters.getData("max_norm").c_str() );

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

    for (++currId ; currId <= endId; ++currId)
    {
        cout<<"file #"<<currId<<endl;
        FRAME currFrame = readFrame( currId, parameters );
        computeKeyPointsAndDesp( currFrame, detector, descriptor );
        RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, intrinPara);// motion estimate from two frames
        
        if (result.inliers < min_inliers)
        {
            cout<<"too less inliers, discard current frame"<<endl;
            continue;
        }
        cout<<result.inliers<<endl;
        double norm = normofTransform(result.rvec, result.tvec);
        if(norm > max_norm)
        {
            cout<<"norm ="<<norm<<" is too large, discard the frame"<<endl;
            continue;
        }
        cout<<"nrom ="<<norm<<endl;

        Eigen::Isometry3d matrixT = cvMat2Eigen(result.rvec, result.tvec);

        cout<<"matrix T is "<<matrixT.matrix()<<endl;
        // add vertex
        vertex = new g2o::VertexSE3();
        vertex->setId(currId);
        vertex->setEstimate(Eigen::Isometry3d::Identity());
        optimizer.addVertex(vertex);
        // create edge
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices() [0] = optimizer.vertex( lastId ); 
        edge->vertices() [1] = vertex;
        // create information matrix
        Eigen::Matrix<double, 6, 6> informationMatrix = Eigen::Matrix< double, 6, 6 >::Identity();
        for(int i = 0; i < 6; ++i)
        {
            informationMatrix(i, i) = 100;
        }
        edge->setInformation( informationMatrix );
        edge->setMeasurement(  matrixT );
        optimizer.addEdge(edge);
        lastFrame = currFrame;
        lastId = currId;
    }

    cout<<"start graph optimization, total vertices:"<<optimizer.vertices().size()<<endl;
    optimizer.save("./data/result_before.g2o");
    optimizer.initializeOptimization();
    optimizer.optimize( 50 );
    optimizer.save("./data/result_after.g2o");

    optimizer.clear();

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
    cout<<"before reading image"<<endl;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;
    cout<<"before reading depth"<<endl;
    f.depth = cv::imread( filename, -1 );
    return f;
}

// first part api
double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

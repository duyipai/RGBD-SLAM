#include "loopClosure.h"

FRAME_CHECK_RESULT checkKeyFrame( FRAME & f1, FRAME & f2, g2o::SparseOptimizer &optimizer, bool isLoopCheck)
{
    static ParameterReader reader;
    static int minInlier = atoi(reader.getData("min_inliers").c_str());
    static double maxNorm = atof(reader.getData("max_norm").c_str());
    static double keyFrame_norm = atof(reader.getData("keyframe_threshold").c_str());
    static double loopFrame_norm = atof(reader.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
    if( result.inliers < minInlier )
    {
        return NO_MATCH;// not many match points
    }
    double norm = normofTransform( result.rvec, result.tvec);
    if( !isLoopCheck )
    {
        if(norm > maxNorm )
            return TOO_FAR;
    }
    else
    {
        if(norm > loopFrame_norm)
            return TOO_FAR;
    }

    if(norm < keyFrame_norm)
    {
        return TOO_CLOSE;
    }

    if( !isLoopCheck )
    {
        g2o::VertexSE3 * vertex = new g2o::VertexSE3();
        vertex->setId(f2.frameID);
        vertex -> setEstimate( Eigen::Isometry3d::Identity());
        optimizer.addVertex(vertex);
    }   

    g2o::EdgeSE3 * edge = new g2o::EdgeSE3();
    edge->setVertex(0, optimizer.vertex(f1.frameID));
    edge->setVertex(1, optimizer.vertex(f2.frameID));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );
    Eigen::Matrix<double, 6, 6> informationMatrix = Eigen::Matrix<double, 6, 6>::Identity();
    for(int i = 0; i < 6; ++i)
    {
        informationMatrix(i, i) = 100;
    }
    edge->setInformation(informationMatrix);
    edge->setMeasurement(cvMat2Eigen(result.rvec, result.tvec).inverse());

    optimizer.addEdge(edge);
    return KEY_FRAME;
}

void checkNearLoop(vector<FRAME> & frames, FRAME & currFrame, g2o::SparseOptimizer & optimizer)
{
    static ParameterReader reader;
    static int checkNum = atoi(reader.getData("nearby_loops").c_str());
    int i = checkNum;
    for(vector<FRAME>::reverse_iterator it = frames.rbegin(); it != frames.rend(); ++it)
    {
        checkKeyFrame(*it, currFrame, optimizer, true);
        --i;
        if(i<=0)
            return;
    }
}

void checkRandomLoop(vector<FRAME> & frames, FRAME & currFrame, g2o::SparseOptimizer & optimizer)
{
    static ParameterReader reader;
    static int checkNum = atoi(reader.getData("random_loops").c_str());
    static int nearCheckNum = atoi(reader.getData("nearby_loops").c_str());
    srand((unsigned int)time(NULL));

    if( int(frames.size())-nearCheckNum<= checkNum)
    {
        for(int i=0; i<int(frames.size())-nearCheckNum; ++i)
        {
            checkKeyFrame(frames[i], currFrame, optimizer, true);
        }
    }
    else
    {
        for(int i=0; i<checkNum; ++i)
        {
            int t = rand()%(frames.size()-nearCheckNum);
            checkKeyFrame(frames[t], currFrame, optimizer, true);
        }
    }
}